"""
store_dvw.py
================
Where the DVW app's matches and knowledge tree live.

Deliberately the same shape of boundary recruiting_data_store.py draws
for the CSV app -- one module is the only thing that knows where bytes
come from -- but the backing store is the LOCAL FILESYSTEM rather than
a private GitHub repo, because .dvw files are already on disk here and
round-tripping ~2MB of match files through the Contents API for every
question would be pure latency.

The knowledge tree persists to a local JSON file. It holds metric
DEFINITIONS only -- skill/code filters and formulas -- and never any
player data, so unlike the CSV world's recruiting_kb_data.json it is
safe to keep next to the code.

Serialization covers the three event-grain payload kinds ("event",
"measure", "formula"). It is intentionally NOT shared with
recruiting_data_store's tree_to_json_dict, which only knows "column"
and "formula": that function would silently drop the where-clause of
every event metric, which is the entire definition.
"""

import json
import os
from typing import Dict, List, Optional

from recruiting_tree import KnowledgeTree, Node, NodeKind

from event_spec import make_event_spec, make_measure_spec, make_metric_formula_spec
from source import SourceSchema

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TREE_PATH = os.path.join(PACKAGE_DIR, "volley_kb_data.json")

# Where the .dvw files live. This is the ONLY place a corpus path is
# written down -- the match files themselves live outside this repo
# (see .gitignore), so nothing here should be welded to one machine's
# layout. Both are overridable by environment variable.
#
# The app loads exactly ONE directory (discover_matches is deliberately
# non-recursive: sibling folders hold duplicate copies of three matches,
# and loading both would double every count). The parser tests decode
# every file in every directory, duplicates included, because there the
# unit is the file rather than the match.
DEFAULT_DVW_DIR = os.environ.get(
    "VOLLEY_DVW_DIR",
    "/fs/vulcan-projects/vlm_motion_benchmark/VolleyballMetrics/player_analysis/downloads/dvw",
)

DVW_SEARCH_DIRS = [
    path for path in os.environ.get("VOLLEY_DVW_DIRS", "").split(os.pathsep) if path
] or [
    DEFAULT_DVW_DIR,
    "/fs/vulcan-projects/vlm_motion_benchmark/VolleyballMetrics/player_analysis/dvw_downloads",
]


def _spec_to_dict(spec) -> dict:
    payload = spec.payload
    kind = payload.get("kind")
    if kind == "event":
        entry = {"kind": "event", "where": payload.get("where", {}),
                 "aggregate": payload.get("aggregate", "count")}
        if payload.get("field"):
            entry["field"] = payload["field"]
        return entry
    if kind == "measure":
        return {"kind": "measure", "measure": payload.get("measure")}
    if kind == "formula":
        return {"kind": "formula", "formula_expr": payload.get("formula_expr"),
                "human_description": payload.get("human_description", "")}
    raise ValueError(f"Unknown spec kind for persistence: {kind!r}")


def _spec_from_dict(entry: dict, schema: SourceSchema, valid_metric_labels):
    kind = entry.get("kind")
    if kind == "event":
        return make_event_spec(
            entry.get("where", {}), entry.get("description", ""), schema,
            aggregate=entry.get("aggregate", "count"), field=entry.get("field"),
        )
    if kind == "measure":
        return make_measure_spec(entry["measure"], entry.get("description", ""))
    if kind == "formula":
        return make_metric_formula_spec(
            entry["formula_expr"], entry.get("human_description", ""), valid_metric_labels,
        )
    raise ValueError(f"Unknown stored spec kind: {kind!r}")


def tree_to_json_dict(tree: KnowledgeTree) -> dict:
    root = tree.committed.get(tree.root_id)
    groups: dict = {}
    if not root:
        return {"volley": groups}

    for branch_id in root.children:
        branch = tree.committed.get(branch_id)
        if not branch:
            continue
        metrics = {}
        for leaf_id in branch.children:
            leaf = tree.committed.get(leaf_id)
            if not leaf or not leaf.spec:
                continue
            entry = _spec_to_dict(leaf.spec)
            entry.update({
                "node_id": leaf.node_id,
                "aliases": leaf.aliases,
                "authored_by": leaf.authored_by,
                "created_at": leaf.created_at,
                "description": leaf.spec.description,
            })
            metrics[leaf.label] = entry
        groups[branch.label] = {"branch_node_id": branch.node_id, "metrics": metrics}
    return {"volley": groups}


def tree_from_json_dict(data: dict, schema: SourceSchema) -> KnowledgeTree:
    """
    Rebuild a tree from stored JSON.

    Branches are created first and leaves added in file order, with
    formula validators closed over the labels committed SO FAR -- the
    same ordering constraint the seed has, since a formula may only
    reference a metric that already exists.
    """
    tree = KnowledgeTree()
    tree.add_root()
    root = tree.committed[tree.root_id]
    known_labels: List[str] = []

    # See recruiting_data_store._tree_from_json_dict: the mirror of this
    # guard. Reading the other world's file must fail loudly, not
    # produce a branchless tree that looks merely empty.
    if "volley" not in data:
        raise ValueError(
            "not an event knowledge base: expected a top-level 'volley' key, "
            f"found {sorted(data) or 'nothing'}"
        )

    for branch_label, branch_data in data.get("volley", {}).items():
        branch_id = branch_data.get("branch_node_id") or tree._new_id(branch_label)
        branch_node = Node(node_id=branch_id, label=branch_label,
                            kind=NodeKind.BRANCH, parent_id=tree.root_id)
        tree.committed[branch_id] = branch_node
        root.children.append(branch_id)

        for leaf_label, entry in branch_data.get("metrics", {}).items():
            spec = _spec_from_dict(entry, schema, list(known_labels))
            leaf_id = entry.get("node_id") or tree._new_id(leaf_label)
            tree.committed[leaf_id] = Node(
                node_id=leaf_id, label=leaf_label, kind=NodeKind.LEAF,
                parent_id=branch_id, spec=spec,
                aliases=entry.get("aliases", []),
                authored_by=entry.get("authored_by", "system"),
                created_at=entry.get("created_at", ""),
            )
            branch_node.children.append(leaf_id)
            known_labels.append(leaf_label)

    tree.seed_from_committed()
    return tree


def save_tree(tree: KnowledgeTree, path: str = DEFAULT_TREE_PATH) -> None:
    """Raises on any failure, deliberately: a silent failure here means
    an edit LOOKS merged in the UI but was never saved."""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(tree_to_json_dict(tree), handle, indent=2)


def load_tree(schema: SourceSchema, path: str = DEFAULT_TREE_PATH) -> Optional[KnowledgeTree]:
    """None on ANY failure (absent, unreadable, malformed, or written
    against a schema this source cannot satisfy) -- the caller then
    seeds a fresh tree, exactly as load_committed_tree does for the CSV
    app. Never raises, so a corrupt file cannot make the app unstartable."""
    try:
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return tree_from_json_dict(data, schema)
    except Exception:
        return None
