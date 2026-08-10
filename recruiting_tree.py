"""
recruiting_tree.py
===================
The recruiting KB's tree structure, in one place: the generic
KnowledgeTree engine, the real CSV column schema it validates against,
the concrete MetricSpec payload/validator, and the seed that builds the
initial committed tree from that schema.

Design decisions locked in during the original design discussion:
  - Branches (categories) and leaves (queryable metrics) are both nodes.
  - Justin has full read/write freedom: add/edit/move/delete anywhere.
  - Two full copies always exist: `committed` (what the live router
    serves queries from) and `staging` (where edits land first).
  - Nothing in staging affects live queries until an explicit merge.
  - Merge requires an explicit diff review -- Justin sees exactly what
    changed before confirming. No silent auto-merge.
  - One fixed executor function (see app.py) reads a leaf's spec and
    computes over the real CSV data -- new leaves never require new code.

Column schema (ground truth): every real column that exists in
Huddle-exported recruiting CSVs, grouped by skill the same way the
file's own column headers already group them (e.g. "Attack K", "Attack
E" -- the prefix IS the grouping, not something imposed). There's no
event-level filtering possible here -- this data is already aggregated
per player, per match. A leaf either references one existing column
directly, or derives a formula from existing columns; nothing can
invent detail that isn't in the export.

Two kinds of leaf:
  - "column": a direct reference to one existing CSV column, no math.
  - "formula": composed from existing columns, e.g.
               "(Attack K - Attack E) / Sets Sets Played".
Validation is mechanical: every column name referenced -- plain
column-ref or inside a formula -- must exist in
RECRUITING_COLUMN_SCHEMA (or be another already-committed metric's
label, when the caller supplies extra_valid_refs). This does NOT judge
whether the formula makes domain sense; that's Justin's call.
"""

from __future__ import annotations
import copy
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable, Dict, List, Optional, Set, Tuple


# ──────────────────────────────────────────────────────────────
# TREE ENGINE
# ──────────────────────────────────────────────────────────────

class NodeKind(Enum):
    BRANCH = "branch"
    LEAF = "leaf"


class ChangeKind(Enum):
    ADD = "add"
    EDIT = "edit"
    MOVE = "move"
    DELETE = "delete"


@dataclass
class MetricSpec:
    """
    Generic container for "what a leaf computes": a human-readable
    description (for the verification view), a payload dict (this KB's
    "column"/"formula" shape, see below), and a validator callable that
    checks the payload against RECRUITING_COLUMN_SCHEMA.
    """
    description: str
    payload: Dict
    validator: Callable[[Dict], List[str]] = field(default=lambda payload: [])

    def validate(self) -> List[str]:
        return self.validator(self.payload)


@dataclass
class Node:
    node_id: str
    label: str
    kind: NodeKind
    parent_id: Optional[str]
    children: List[str] = field(default_factory=list)

    # Only populated for kind == LEAF
    spec: Optional[MetricSpec] = None
    chart_constraints: Optional[Dict[str, List[str]]] = None
    aliases: List[str] = field(default_factory=list)

    # Provenance -- who authored this, when. Matters once Justin is
    # actively growing the tree; also feeds the diff/merge review.
    authored_by: str = "system"
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def is_leaf(self) -> bool:
        return self.kind == NodeKind.LEAF


@dataclass
class ChangeRecord:
    """One entry in a diff between staging and committed."""
    kind: ChangeKind
    node_id: str
    label: str
    before: Optional[Node] = None
    after: Optional[Node] = None
    # For DELETE/MOVE specifically: what currently depends on this node,
    # surfaced so the impact warning has real content instead of being
    # cosmetic. Populated by KnowledgeTree.compute_diff via
    # dependent_query_log (see below).
    dependents: List[str] = field(default_factory=list)


class KnowledgeTree:
    """
    Holds both the `committed` and `staging` copies of the tree, plus a
    lightweight dependency log (which past query patterns/benchmark
    entries touched which node) so diff/merge impact warnings are real,
    not cosmetic.
    """

    def __init__(self):
        self.committed: Dict[str, Node] = {}
        self.staging: Dict[str, Node] = {}
        self.root_id = "root"
        # node_id -> list of query descriptions that have hit this node.
        # In production this should be populated from real usage logs /
        # the benchmark harness; stubbed here as a plain dict for the
        # prototype.
        self.dependent_query_log: Dict[str, List[str]] = {}

    # ── construction ──────────────────────────────────────────

    def _new_id(self, label: str) -> str:
        slug = label.lower().replace(" ", "_").replace("-", "_")
        return f"{slug}_{uuid.uuid4().hex[:6]}"

    def add_root(self) -> None:
        root = Node(node_id=self.root_id, label="Volleyball Metrics",
                    kind=NodeKind.BRANCH, parent_id=None)
        self.committed[self.root_id] = root
        self.staging[self.root_id] = copy.deepcopy(root)

    def add_node(self, *, label: str, kind: NodeKind, parent_id: str,
                 to: str = "staging", spec: Optional[MetricSpec] = None,
                 chart_constraints: Optional[Dict[str, List[str]]] = None,
                 aliases: Optional[List[str]] = None,
                 authored_by: str = "system") -> str:
        """Adds a node to either 'staging' or 'committed' (committed is
        used only for seeding the initial taxonomy; Justin's real edits
        should always target staging)."""
        target = self.staging if to == "staging" else self.committed
        if parent_id not in target:
            raise ValueError(f"Parent '{parent_id}' does not exist in {to}.")

        node_id = self._new_id(label)
        node = Node(
            node_id=node_id, label=label, kind=kind, parent_id=parent_id,
            spec=spec, chart_constraints=chart_constraints,
            aliases=aliases or [], authored_by=authored_by,
        )
        if kind == NodeKind.LEAF and spec is not None:
            errors = spec.validate()
            if errors:
                raise ValueError(f"Invalid spec for '{label}': {'; '.join(errors)}")

        target[node_id] = node
        target[parent_id].children.append(node_id)
        return node_id

    def seed_from_committed(self) -> None:
        """Copies committed -> staging, e.g. at the start of a session
        before Justin starts making edits."""
        self.staging = copy.deepcopy(self.committed)

    # ── editing (always against staging) ─────────────────────

    def edit_leaf_spec(self, node_id: str, new_spec: MetricSpec) -> None:
        errors = new_spec.validate()
        if errors:
            raise ValueError(f"Invalid spec: {'; '.join(errors)}")
        if node_id not in self.staging:
            raise ValueError(f"'{node_id}' not found in staging.")
        self.staging[node_id].spec = new_spec

    def move_node(self, node_id: str, new_parent_id: str) -> None:
        if node_id not in self.staging or new_parent_id not in self.staging:
            raise ValueError("Both node and new parent must exist in staging.")
        node = self.staging[node_id]
        old_parent = self.staging.get(node.parent_id)
        if old_parent:
            old_parent.children.remove(node_id)
        node.parent_id = new_parent_id
        self.staging[new_parent_id].children.append(node_id)

    def delete_node(self, node_id: str) -> None:
        """Deletes a node AND everything beneath it in staging."""
        if node_id not in self.staging:
            return
        node = self.staging[node_id]
        for child_id in list(node.children):
            self.delete_node(child_id)
        parent = self.staging.get(node.parent_id)
        if parent and node_id in parent.children:
            parent.children.remove(node_id)
        del self.staging[node_id]

    # ── diff / merge ──────────────────────────────────────────

    def _subtree_ids(self, tree: Dict[str, Node], node_id: str) -> List[str]:
        if node_id not in tree:
            return []
        ids = [node_id]
        for c in tree[node_id].children:
            ids.extend(self._subtree_ids(tree, c))
        return ids

    def compute_diff(self) -> List[ChangeRecord]:
        """Compares staging against committed. This is what gets shown to
        Justin before merge -- every add/edit/move/delete, with impact
        context for anything that has real dependents."""
        changes: List[ChangeRecord] = []
        committed_ids = set(self.committed)
        staging_ids = set(self.staging)

        for node_id in staging_ids - committed_ids:
            node = self.staging[node_id]
            changes.append(ChangeRecord(
                kind=ChangeKind.ADD, node_id=node_id, label=node.label,
                after=node,
            ))

        for node_id in committed_ids - staging_ids:
            node = self.committed[node_id]
            dependents = self.dependent_query_log.get(node_id, [])
            changes.append(ChangeRecord(
                kind=ChangeKind.DELETE, node_id=node_id, label=node.label,
                before=node, dependents=dependents,
            ))

        for node_id in committed_ids & staging_ids:
            before, after = self.committed[node_id], self.staging[node_id]
            if before.parent_id != after.parent_id:
                dependents = self.dependent_query_log.get(node_id, [])
                changes.append(ChangeRecord(
                    kind=ChangeKind.MOVE, node_id=node_id, label=after.label,
                    before=before, after=after, dependents=dependents,
                ))
            elif before.spec != after.spec or before.label != after.label:
                dependents = self.dependent_query_log.get(node_id, [])
                changes.append(ChangeRecord(
                    kind=ChangeKind.EDIT, node_id=node_id, label=after.label,
                    before=before, after=after, dependents=dependents,
                ))
        return changes

    def merge(self, approved_node_ids: Optional[List[str]] = None) -> None:
        """
        Commits staging -> committed. If approved_node_ids is given, only
        those changed nodes are merged (Justin approved some, rejected
        others, within the same batch); otherwise the entire staged diff
        is merged. Never called without compute_diff() + explicit
        confirmation happening first, per "diff explicitly, always".
        """
        diff = self.compute_diff()
        to_merge = diff if approved_node_ids is None else [
            c for c in diff if c.node_id in approved_node_ids
        ]
        for change in to_merge:
            if change.kind == ChangeKind.DELETE:
                self.committed.pop(change.node_id, None)
                parent = self.committed.get(change.before.parent_id)
                if parent and change.node_id in parent.children:
                    parent.children.remove(change.node_id)
            else:
                self.committed[change.node_id] = copy.deepcopy(self.staging[change.node_id])
                parent = self.committed.get(self.staging[change.node_id].parent_id)
                if parent and change.node_id not in parent.children:
                    parent.children.append(change.node_id)

    # ── query-time lookup ─────────────────────────────────────

    def find_path(self, node_id: str, tree: Optional[Dict[str, Node]] = None) -> List[str]:
        """Root-to-node path of labels, e.g. ['Attacking', 'Out-of-System', 'Bailout Kill %']."""
        tree = tree or self.committed
        path = []
        cur = tree.get(node_id)
        while cur:
            path.insert(0, cur.label)
            cur = tree.get(cur.parent_id) if cur.parent_id else None
        return path

    def render_tree(self, tree: Optional[Dict[str, Node]] = None, node_id: Optional[str] = None, depth: int = 0) -> str:
        """Plain-text indented tree, for quick visual sanity checks and as
        a stand-in for the eventual real UI rendering."""
        tree = tree or self.committed
        node_id = node_id or self.root_id
        node = tree.get(node_id)
        if not node:
            return ""
        marker = "L" if node.is_leaf() else "B"
        line = f"{'  ' * depth}[{marker}] {node.label}"
        if node.is_leaf() and node.spec:
            line += f"  --  {node.spec.description}"
        lines = [line]
        for child_id in node.children:
            lines.append(self.render_tree(tree, child_id, depth + 1))
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# COLUMN SCHEMA -- ground truth: every real column that exists in the
# Huddle-exported recruiting CSVs, grouped by the prefix the export
# itself already uses.
# ──────────────────────────────────────────────────────────────

class ColumnType(Enum):
    COUNT = "count"          # raw counting stat, e.g. Attack K
    PERCENTAGE = "percentage"  # already a ratio/percentage, e.g. Attack Atk%
    RATE = "rate"             # per-set or similar rate, e.g. Attack K/S
    RATING = "rating"         # a composite rating score, e.g. Serve Rtg.


@dataclass(frozen=True)
class ColumnSpec:
    name: str            # exact column header as it appears in the CSV
    skill_group: str      # Attack / Serve / Receive / Set / Dig / Block / Points / Sets
    type: ColumnType
    description: str = ""


RECRUITING_COLUMN_SCHEMA: Dict[str, ColumnSpec] = {
    # ── Attack ──
    "Attack K":     ColumnSpec("Attack K", "Attack", ColumnType.COUNT, "Kills"),
    "Attack E":     ColumnSpec("Attack E", "Attack", ColumnType.COUNT, "Attack errors"),
    "Attack TA":    ColumnSpec("Attack TA", "Attack", ColumnType.COUNT, "Total attack attempts"),
    "Attack Atk%":  ColumnSpec("Attack Atk%", "Attack", ColumnType.PERCENTAGE, "(K - E) / TA, hitting efficiency"),
    "Attack K/S":   ColumnSpec("Attack K/S", "Attack", ColumnType.RATE, "Kills per set played"),

    # ── Serve ──
    "Serve SA":     ColumnSpec("Serve SA", "Serve", ColumnType.COUNT, "Service aces"),
    "Serve SE":     ColumnSpec("Serve SE", "Serve", ColumnType.COUNT, "Service errors"),
    "Serve TA":     ColumnSpec("Serve TA", "Serve", ColumnType.COUNT, "Total serve attempts"),
    "Serve Pct":    ColumnSpec("Serve Pct", "Serve", ColumnType.PERCENTAGE, "Serve success percentage"),
    "Serve Eff":    ColumnSpec("Serve Eff", "Serve", ColumnType.PERCENTAGE, "Serve efficiency"),
    "Serve Rtg.":   ColumnSpec("Serve Rtg.", "Serve", ColumnType.RATING, "Composite serve rating"),

    # ── Receive ──
    "Receive 3":    ColumnSpec("Receive 3", "Receive", ColumnType.COUNT, "Perfect passes (top of quality scale)"),
    "Receive 2":    ColumnSpec("Receive 2", "Receive", ColumnType.COUNT, "Good passes"),
    "Receive 1":    ColumnSpec("Receive 1", "Receive", ColumnType.COUNT, "Poor passes"),
    "Receive 0":    ColumnSpec("Receive 0", "Receive", ColumnType.COUNT, "Errors / lowest quality"),
    "Receive TA":   ColumnSpec("Receive TA", "Receive", ColumnType.COUNT, "Total receive attempts"),
    "Receive Pass%": ColumnSpec("Receive Pass%", "Receive", ColumnType.PERCENTAGE, "Passing quality percentage"),

    # ── Set ──
    "Set Ast":      ColumnSpec("Set Ast", "Set", ColumnType.COUNT, "Assists"),
    "Set TA":       ColumnSpec("Set TA", "Set", ColumnType.COUNT, "Total set attempts"),
    "Set SE":       ColumnSpec("Set SE", "Set", ColumnType.COUNT, "Setting errors"),
    "Set 3":        ColumnSpec("Set 3", "Set", ColumnType.COUNT, "Perfect sets (same 3/2/1/0 scale as Receive)"),
    "Set 2":        ColumnSpec("Set 2", "Set", ColumnType.COUNT, "Good sets"),
    "Set 1":        ColumnSpec("Set 1", "Set", ColumnType.COUNT, "Poor sets"),
    "Set 0":        ColumnSpec("Set 0", "Set", ColumnType.COUNT, "Set errors"),
    "Set Rtg.":     ColumnSpec("Set Rtg.", "Set", ColumnType.RATING, "Composite setting rating"),

    # ── Dig ──
    "Dig DS":       ColumnSpec("Dig DS", "Dig", ColumnType.COUNT, "Digs succeeded"),
    "Dig DE":       ColumnSpec("Dig DE", "Dig", ColumnType.COUNT, "Dig errors"),

    # ── Block ──
    "Block BS":     ColumnSpec("Block BS", "Block", ColumnType.COUNT, "Solo blocks"),
    "Block BA":     ColumnSpec("Block BA", "Block", ColumnType.COUNT, "Assisted blocks"),
    "Block BE":     ColumnSpec("Block BE", "Block", ColumnType.COUNT, "Block errors"),
    "Block B/S":    ColumnSpec("Block B/S", "Block", ColumnType.RATE, "Blocks per set"),

    # ── Points / Sets (match-level, not skill-specific) ──
    "Points Pts +/-": ColumnSpec("Points Pts +/-", "Points", ColumnType.COUNT, "Plus/minus point differential"),
    "Sets Sets Played": ColumnSpec("Sets Sets Played", "Sets", ColumnType.COUNT, "Sets played in this match"),
}


def column_exists(name: str) -> bool:
    return name in RECRUITING_COLUMN_SCHEMA


def columns_in_group(skill_group: str) -> List[str]:
    return [c.name for c in RECRUITING_COLUMN_SCHEMA.values() if c.skill_group == skill_group]


ALL_SKILL_GROUPS: List[str] = ["Attack", "Serve", "Receive", "Set", "Dig", "Block", "Points", "Sets"]


# ──────────────────────────────────────────────────────────────
# METRIC SPEC -- concrete payload/validator for this KB's two leaf kinds,
# plugged into the generic MetricSpec container above.
# ──────────────────────────────────────────────────────────────

# Matches a bracketed column reference in a formula string, e.g.
# "[Attack K] - [Attack E]" -- brackets required because column names
# contain spaces and can't be parsed as bare identifiers.
_COLUMN_REF_PATTERN = re.compile(r"\[([^\[\]]+)\]")


def _extract_referenced_columns(formula_expr: str) -> List[str]:
    return _COLUMN_REF_PATTERN.findall(formula_expr)


def _make_validator(extra_valid_refs: Optional[Set[str]] = None):
    """
    Builds a validator closed over `extra_valid_refs` -- a snapshot of
    labels that should ALSO be accepted as valid bracket-token/column_ref
    targets, on top of the raw RECRUITING_COLUMN_SCHEMA columns. Used so a
    formula can reference another already-committed metric by its label
    (e.g. "[Kills Per Set]" composing on top of "[Attack K]"), not just raw
    CSV columns -- the caller (app.py) passes the current tree's committed
    leaf labels as extra_valid_refs at spec-construction time.
    """
    extra = extra_valid_refs or set()

    def _validate(payload: Dict) -> List[str]:
        errors = []
        kind = payload.get("kind")
        known_names = sorted(set(RECRUITING_COLUMN_SCHEMA) | extra)

        if kind == "column":
            col = payload.get("column_ref")
            if not (column_exists(col) or col in extra):
                errors.append(
                    f"'{col}' is not a real column or an existing metric. "
                    f"Known: {', '.join(known_names)}"
                )

        elif kind == "formula":
            expr = payload.get("formula_expr", "")
            refs = _extract_referenced_columns(expr)
            if not refs:
                errors.append(
                    f"Formula '{expr}' references no bracketed columns -- "
                    f"expected at least one [Column Name] or [Existing Metric Label]."
                )
            for col in refs:
                if not (column_exists(col) or col in extra):
                    errors.append(
                        f"'{col}' referenced in formula is neither a real column nor "
                        f"an existing metric. Known: {', '.join(known_names)}"
                    )

        else:
            errors.append(f"Unknown spec kind '{kind}' -- expected 'column' or 'formula'.")

        return errors

    return _validate


def make_column_spec(column_ref: str, extra_valid_refs: Optional[Set[str]] = None) -> MetricSpec:
    """Leaf that's just a direct reference to an existing column (or,
    if extra_valid_refs is given, another existing metric's label)."""
    col = RECRUITING_COLUMN_SCHEMA.get(column_ref)
    desc = col.description if col else "(unknown column)"
    return MetricSpec(
        description=f"Direct column: {column_ref} -- {desc}",
        payload={"kind": "column", "column_ref": column_ref},
        validator=_make_validator(extra_valid_refs),
    )


def make_formula_spec(formula_expr: str, human_description: str,
                       extra_valid_refs: Optional[Set[str]] = None) -> MetricSpec:
    """
    Leaf that's a derived formula over existing columns and/or other
    existing metrics (when extra_valid_refs is given), e.g.:
        make_formula_spec(
            "([Attack K] - [Attack E]) / [Sets Sets Played]",
            "Net kills per set played",
        )
    This is Justin's authoring path for a genuinely new metric.
    """
    return MetricSpec(
        description=f"Formula: {formula_expr} -- {human_description}",
        payload={"kind": "formula", "formula_expr": formula_expr,
                 "human_description": human_description},
        validator=_make_validator(extra_valid_refs),
    )


# ──────────────────────────────────────────────────────────────
# TREE SEED -- builds the initial COMMITTED tree from every real column
# in RECRUITING_COLUMN_SCHEMA, organized under one branch per skill
# group (that grouping is already implicit in the CSV's own
# column-prefix convention, not invented).
# ──────────────────────────────────────────────────────────────

def seed_recruiting_tree() -> Tuple[KnowledgeTree, Dict[str, str]]:
    """
    Builds a brand-new, independent KnowledgeTree seeded from every real
    column in the recruiting CSV export, one branch per skill group.
    Each column becomes a plain "column" leaf under its group, seeded
    straight into committed (to="committed" is for initial-taxonomy
    seeding only -- see add_node above).

    Ends with seed_from_committed() so the returned tree's staging already
    mirrors committed and is immediately ready for a caller to start
    staging edits against.

    Returns (tree, branch_ids) -- branch_ids maps each skill group name
    (e.g. "Attack") to its branch node_id, so callers don't have to
    re-derive that mapping by walking the tree or matching on label text.
    """
    tree = KnowledgeTree()
    tree.add_root()

    branch_ids = {}
    for group in ALL_SKILL_GROUPS:
        branch_ids[group] = tree.add_node(
            label=group, kind=NodeKind.BRANCH, parent_id=tree.root_id, to="committed",
        )
        for col_name in columns_in_group(group):
            col = RECRUITING_COLUMN_SCHEMA[col_name]
            tree.add_node(
                label=col.description or col_name,
                kind=NodeKind.LEAF,
                parent_id=branch_ids[group],
                to="committed",
                spec=make_column_spec(col_name),
                aliases=[col_name.lower()],
                authored_by="system:csv_import",
            )

    tree.seed_from_committed()
    return tree, branch_ids


if __name__ == "__main__":
    tree, branch_ids = seed_recruiting_tree()

    print("=" * 70)
    print("SEEDED RECRUITING TREE (from real CSV columns)")
    print("=" * 70)
    print(tree.render_tree())

    # ──────────────────────────────────────────────────────────────
    # DEMO: Justin adds a derived formula
    # "Net kills per set played" = (Attack K - Attack E) / Sets Sets Played
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("DEMO: Justin derives a new metric from existing columns")
    print("=" * 70)

    new_leaf_id = tree.add_node(
        label="Net Kills Per Set",
        kind=NodeKind.LEAF,
        parent_id=branch_ids["Attack"],
        to="staging",
        spec=make_formula_spec(
            "([Attack K] - [Attack E]) / [Sets Sets Played]",
            "Net kills (kills minus errors) per set played -- Justin's own "
            "efficiency-adjusted volume metric, not a raw export column.",
        ),
        aliases=["net kills per set", "adjusted kill rate"],
        authored_by="justin",
    )

    diff = tree.compute_diff()
    for c in diff:
        print(f"  [{c.kind.value.upper()}] {c.label}")
        if c.after and c.after.spec:
            print(f"      {c.after.spec.description}")

    tree.merge()
    print("\nMerged. New leaf now committed and routable.")

    # ──────────────────────────────────────────────────────────────
    # DEMO: mechanical rejection of an invalid formula
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("DEMO: Justin references a column that doesn't exist in this export")
    print("=" * 70)
    try:
        tree.add_node(
            label="Bad Metric", kind=NodeKind.LEAF, parent_id=branch_ids["Attack"], to="staging",
            spec=make_formula_spec("[Attack K] / [Attack Zone Rating]", "invalid -- no such column"),
            authored_by="justin",
        )
    except ValueError as e:
        print(f"Rejected mechanically: {e}")
