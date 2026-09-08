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

#: Where each workspace's committed tree lives inside the private repo.
RECRUITING_TREE_PATH = data_store.TREE_JSON_PATH
DVW_TREE_PATH = "volley_kb_data.json"

RECRUITING = "recruiting"
PLAYER_ANALYSIS = "player_analysis"


def dvw_unavailable_reason() -> Optional[str]:
    """
    Why the play-by-play workspace cannot be offered, or None if it can.

    The DVW stack has a hard dependency bound (pydatavolley predates
    pandas 3 and numpy 2), so importing it is allowed to FAIL. Asking
    first -- rather than importing at module scope and hoping -- is what
    keeps a .dvw dependency problem from taking down the recruiting
    dashboard, which needs none of it.
    """
    try:
        import source_dvw  # noqa: F401
    except Exception as error:
        return f"{type(error).__name__}: {error}"
    return None


def available_keys() -> List[str]:
    """Workspaces that can actually be built. Recruiting is always
    available; it depends on none of the .dvw stack."""
    keys = [RECRUITING]
    if dvw_unavailable_reason() is None:
        keys.append(PLAYER_ANALYSIS)
    return keys


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

    def worked_example(self):
        """
        A frame the metric editor can compute a real example against.

        For the CSV world that is one match's export, which is what the
        editor has always used -- a spec there is a column reference, so
        it needs the columns. For the event world there is no such
        frame: an event metric is a filter over actions and its worked
        example would have to be computed, not read off a row. Returning
        None makes the editor omit the example rather than invent one.
        """
        frames = getattr(self.source, "frames", None)
        if frames is None:
            return None
        loaded = frames()
        return loaded[0][1] if loaded else None

    def teams(self) -> List[str]:
        """Teams selectable as the point of view, most-played first.
        Empty for a source that has no such notion."""
        return list(getattr(self.source, "teams", list)())

    @property
    def team(self) -> Optional[str]:
        return getattr(self.source, "team_of_interest", None)

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
    from source_dvw import DvwSource

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

    if team is None and paths:
        # Whose season is this? Without an answer, MatchInfo.opponent
        # falls back to the visiting team, so every away match gets
        # labelled with the scouted team's own name.
        #
        # Guarded because this is the first thing that actually PARSES a
        # file, so any parser incompatibility surfaces here -- and a
        # parser problem must disable one workspace, never take down the
        # whole app and the recruiting side with it.
        try:
            team = DvwSource(paths).default_team()
        except Exception as error:
            paths = []
            warnings.append(f"Couldn't read the match files: {type(error).__name__}: {error}")

    source = DvwSource(paths, team_of_interest=team)

    def _save(tree: KnowledgeTree) -> None:
        data_store.save_committed_tree(
            tree, path=DVW_TREE_PATH, serializer=store_dvw.tree_to_json_dict,
        )

    tree = _load_dvw_tree(source, warnings)
    if tree is None:
        tree, _ = seed_dvw_tree(source.schema)
        # Only PERSIST a seeded tree when it was seeded against real
        # matches. With no readable files every primitive fails
        # validation, so the seed is empty -- saving that would
        # overwrite a good knowledge base with nothing.
        if paths:
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


def _load_dvw_tree(source: "Source", warnings: List[str]) -> Optional[KnowledgeTree]:
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


# ──────────────────────────────────────────────────────────────
# SWITCHING
# ──────────────────────────────────────────────────────────────

#: Session keys scoped to ONE workspace, and their value on reset.
#: Everything here names something -- a player, a game, a metric, a tree
#: node id -- that may simply not exist in the world being switched to.
SCOPED_STATE = {
    "qa_last_decomposition": None,
    "qa_action_results": [],
    "qa_encodings": {},
    "qa_player_filter": [],
    "expanded_branches": set(),
    "selected_node_id": None,
    "last_graph_click": None,
    "editing_node_id": None,
    "pending_delete_node_id": None,
    "new_metric_step": "describe",
    "new_metric_worked_example": None,
}

#: Widget keys that are generated per player / per chart slot and so
#: cannot be listed literally. Streamlit keeps widget values under these
#: keys across reruns, which is exactly why they have to be deleted
#: rather than merely reset.
SCOPED_KEY_PREFIXES = ("qa_enc_", "qa_playerfilter_")

#: Widget keys that are keyed by LABEL, and would carry one world's
#: opponents into the other world's picker.
SCOPED_WIDGET_KEYS = ("qa_games_multiselect", "qa_sets_multiselect")


def clear_scoped_state(session_state) -> None:
    """
    Drop everything belonging to the workspace being left.

    Lives here rather than in app.py so it is testable without executing
    the Streamlit script -- importing app.py runs it, which in a bare
    context leaves an st.form block open and breaks the next AppTest.
    Session hygiene is not layout, and it should not need a UI to check.
    """
    for key, value in SCOPED_STATE.items():
        session_state[key] = set() if isinstance(value, set) else (
            list(value) if isinstance(value, list) else
            dict(value) if isinstance(value, dict) else value
        )
    for key in [k for k in list(session_state) if k.startswith(SCOPED_KEY_PREFIXES)]:
        del session_state[key]
    for key in SCOPED_WIDGET_KEYS:
        if key in session_state:
            del session_state[key]
