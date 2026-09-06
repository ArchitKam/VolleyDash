"""
workspace.py
=============
A Workspace is everything the UI needs that depends on WHICH data you
are looking at: the source, its knowledge base, where that knowledge
base is saved, and which capabilities the combination supports.

This is the seam the source dropdown switches. Everything else in the
app -- the router, the cube algebra, the chart encoder, the tree editor,
the diff/merge review -- takes a Workspace and behaves identically
either way. Adding a third format means adding a builder here, not
touching the UI.

WHY THE KNOWLEDGE BASES ARE SEPARATE
Because the primitives genuinely differ. A recruiting metric is a
Huddle export COLUMN ("Attack K"); an event metric is a FILTER over
actions (skill=Attack, evaluation_code=#). Neither vocabulary can
validate against the other's data, and a single merged tree would be a
tree where most metrics are broken most of the time. They share the
KnowledgeTree machinery -- staging, diffing, merging, provenance -- and
nothing else, which is exactly the split the two seed functions already
described.

WHY BOTH PERSIST TO THE PRIVATE REPO
Streamlit Community Cloud deploys the public code repo and keeps no
writable disk between restarts, so a knowledge base saved locally is a
knowledge base that silently disappears. Both trees go to VolleyData,
each to its own file, through the same two-step Contents API update.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import recruiting_data_store as data_store
import store_dvw
from recruiting_tree import KnowledgeTree, seed_recruiting_tree
from seed_dvw import seed_dvw_tree
from source import SET_AXIS, Source
from source_csv import CsvSource
from source_dvw import DvwSource

#: Where each workspace's committed tree lives inside the private repo.
RECRUITING_TREE_PATH = data_store.TREE_JSON_PATH
DVW_TREE_PATH = "volley_kb_data.json"

RECRUITING = "recruiting"
PLAYER_ANALYSIS = "player_analysis"


@dataclass
class Workspace:
    """One data world, fully assembled."""

    key: str
    label: str
    caption: str
    source: Source
    tree: KnowledgeTree
    tree_path: str
    save_tree: Callable[[KnowledgeTree], None]
    #: Non-fatal problems encountered while assembling (missing secrets,
    #: an unreachable repo). Surfaced by the UI rather than raised, so a
    #: half-available workspace still shows what it can.
    warnings: List[str] = field(default_factory=list)

    @property
    def supports_sets(self) -> bool:
        """Whether this world can answer a per-set question at all. Read
        off the source rather than the workspace key, so it stays true
        by construction if a format's capabilities change."""
        return SET_AXIS in self.source.identity_fields()

    def games(self) -> List[str]:
        return self.source.game_labels()

    def players(self) -> List[str]:
        facts = self.source.facts()
        column = self.source.identity_fields()["Player"]
        if facts.empty or column not in facts.columns:
            return []
        return sorted(facts[column].dropna().unique())

    def set_labels(self) -> List[str]:
        """Every set present in the loaded data, in set order. Empty for
        a source without the axis, which is what the UI checks before
        offering the control."""
        if not self.supports_sets:
            return []
        facts = self.source.facts()
        column = self.source.identity_fields()[SET_AXIS]
        if facts.empty or column not in facts.columns:
            return []
        from source_dvw import set_sort_key

        return sorted((v for v in facts[column].dropna().unique() if v), key=set_sort_key)


# ──────────────────────────────────────────────────────────────
# BUILDERS
# ──────────────────────────────────────────────────────────────

def build_recruiting(selected_game_paths: Optional[List[str]] = None) -> Workspace:
    """
    The Huddle CSV world, unchanged in behaviour from before
    unification: same store, same tree file, same seed.
    """
    warnings: List[str] = []

    try:
        games = data_store.get_games()
    except Exception as error:
        games = []
        warnings.append(f"Couldn't list matches in the data repo: {error}")

    if selected_game_paths is not None:
        wanted = set(selected_game_paths)
        games = [game for game in games if game.path in wanted]

    frames = []
    for game in games:
        try:
            frames.append((game, data_store.load_game_df(game.path)))
        except Exception as error:
            warnings.append(f"Couldn't load {game.filename}: {error}")

    tree = data_store.load_committed_tree()
    if tree is None:
        tree, _ = seed_recruiting_tree()
        try:
            data_store.save_committed_tree(tree)
        except RuntimeError as error:
            warnings.append(f"Couldn't save the seeded tree: {error}")

    return Workspace(
        key=RECRUITING,
        label="Recruiting (match exports)",
        caption="One row per player per match, from Huddle CSV exports.",
        source=CsvSource(frames),
        tree=tree,
        tree_path=RECRUITING_TREE_PATH,
        save_tree=data_store.save_committed_tree,
        warnings=warnings,
    )


def build_player_analysis(team: Optional[str] = None) -> Workspace:
    """
    The DataVolley world. Match files come from the private repo unless
    a local corpus is configured (data_store.dvw_paths), so this works
    both on a machine that already has the files and on Streamlit Cloud,
    which has neither the files nor a writable disk.
    """
    warnings: List[str] = []

    try:
        paths = data_store.dvw_paths()
    except Exception as error:
        paths = []
        warnings.append(f"Couldn't reach the match files: {error}")

    if not paths:
        warnings.append(
            "No .dvw match files found. Upload them to the private data repo's "
            f"{data_store.DVW_DIR_PATH}/ folder, or set "
            f"{data_store.DVW_LOCAL_DIR_ENV} to a local directory."
        )

    source = DvwSource(paths, team_of_interest=team)

    def _save(tree: KnowledgeTree) -> None:
        data_store.save_committed_tree(
            tree, path=DVW_TREE_PATH, serializer=store_dvw.tree_to_json_dict,
        )

    tree = _load_dvw_tree(source, warnings)
    if tree is None:
        tree, _ = seed_dvw_tree(source.schema)
        try:
            _save(tree)
        except RuntimeError as error:
            warnings.append(f"Couldn't save the seeded tree: {error}")

    return Workspace(
        key=PLAYER_ANALYSIS,
        label="Player analysis (play-by-play)",
        caption="One row per action, from DataVolley .dvw files. Supports per-set drill-down.",
        source=source,
        tree=tree,
        tree_path=DVW_TREE_PATH,
        save_tree=_save,
        warnings=warnings,
    )


def _load_dvw_tree(source: DvwSource, warnings: List[str]) -> Optional[KnowledgeTree]:
    """
    Committed tree out of the private repo, validated against the schema
    the loaded matches actually have.

    None on ANY failure, so a missing file or a tree written against a
    different corpus seeds a fresh one instead of making the app
    unstartable -- the same contract load_committed_tree has for the CSV
    side.
    """
    try:
        raw = data_store.load_json_file(DVW_TREE_PATH)
    except Exception as error:
        warnings.append(f"Couldn't load the saved knowledge base: {error}")
        return None
    if raw is None:
        return None
    try:
        return store_dvw.tree_from_json_dict(raw, source.schema)
    except Exception as error:
        warnings.append(
            f"The saved knowledge base doesn't fit the loaded matches ({error}); "
            "seeding a fresh one."
        )
        return None


BUILDERS: Dict[str, Callable[..., Workspace]] = {
    RECRUITING: build_recruiting,
    PLAYER_ANALYSIS: build_player_analysis,
}

LABELS: Dict[str, str] = {
    RECRUITING: "Recruiting (match exports)",
    PLAYER_ANALYSIS: "Player analysis (play-by-play)",
}
