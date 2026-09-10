#omsairam omsairam omsairam 
"""
app.py
======
Interactive prototype for growing the recruiting knowledge-base tree AND
asking natural-language questions over it. Calls directly into the real
recruiting_tree.py engine (KnowledgeTree, schema, MetricSpec) and
recruiting_llm.py (LLM-backed authoring/routing, with a rule-based
fallback) -- UI layer plus CSV-facing glue.

Run with: streamlit run app.py
"""

import ast
import copy
import difflib
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_agraph import agraph, Node as AGNode, Edge as AGEdge, Config as AGConfig

from recruiting_tree import (
    ALL_SKILL_GROUPS, ChangeKind, ColumnType, KnowledgeTree, MetricSpec, Node, NodeKind,
    RECRUITING_COLUMN_SCHEMA, make_column_spec, make_formula_spec, seed_recruiting_tree,
)
from recruiting_llm import (
    EXAMPLE_PHRASES, LLMUnavailableError, decompose_recruiting_query,
    parse_phrase_to_formula, parse_phrase_to_formula_llm,
)
from recruiting_operations import Operation, Rank, Slice, run_pipeline, describe_pipeline
from recruiting_data_store import (
    save_committed_tree, load_committed_tree,
)
from recruiting_encoding import (
    EncodingAssignment, default_encoding, reconcile_encoding, resolve_clicked_point,
    set_game_order, slot_options, render as render_encoded_panels,
)
import workspace

try:
    if "GROQ_API_KEY" not in os.environ and "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

st.set_page_config(page_title="Recruiting KB", page_icon="🏐", layout="wide")

_SINGLE_COL_RE = re.compile(r"^\[([^\[\]]+)\]$")

UMD_RED = "#E21833"
UMD_BLACK = "#000000"
UMD_GOLD = "#B8860B"
UMD_WHITE = "#FFFFFF"
UMD_CYCLE = [UMD_RED, UMD_GOLD, UMD_WHITE]

# Categorical hues for the Player axis, in fixed order, stepped for a DARK
# chart surface. Validated against this app's actual surface (#000000)
# rather than assumed:
#
#   PASS lightness band · PASS chroma floor · PASS CVD separation
#   (worst adjacent dE 8.4) · PASS normal-vision floor (19.3) · PASS contrast
#
# The palette these replace failed outright: four near-identical reds and
# golds plus white and grey, with red<->gold at CVD dE 6.6 -- two players a
# colourblind reader could not tell apart -- and two entries with no chroma
# at all, which read as "no series" rather than as a player.
#
# ORDER IS FIXED AND ASSIGNMENT IS BY ROSTER POSITION, not by who happens to
# be on screen. A player keeps their colour when the filter changes; a chart
# that repainted its survivors every time you unticked someone would make
# colour meaningless as identity.
PLAYER_PALETTE = [
    "#3987e5",  # blue
    "#d95926",  # orange
    "#199e70",  # aqua
    "#c98500",  # yellow
    "#d55181",  # magenta
    "#008300",  # green
    "#9085e9",  # violet
    "#e66767",  # red
]

PLOTLY_BASE = dict(
    plot_bgcolor=UMD_BLACK, paper_bgcolor=UMD_BLACK,
    font=dict(family="sans-serif", color=UMD_WHITE),
    margin=dict(l=40, r=20, t=40, b=40),
    xaxis=dict(color=UMD_WHITE, gridcolor="#333333"),
    yaxis=dict(color=UMD_WHITE, gridcolor="#333333"),
)

st.markdown(f"""
<style>
h1, h2, h3 {{ border-bottom: 2px solid {UMD_RED}; padding-bottom: 0.2em; }}
hr {{ border-top: 1px solid {UMD_GOLD}; }}
</style>
""", unsafe_allow_html=True)


def get_player_color_map(known_player_pool: List[str]) -> dict:
    """
    Player -> colour, fixed by the player's position in the sorted ROSTER.

    Sorted roster rather than the players in the current result, so a
    player's colour is a property of the player and survives every
    filter, question and re-run.

    Eight hues is the validated set. A roster longer than eight reuses
    them, which is a real limit rather than a hidden one: past eight
    players ON ONE CHART two of them share a hue, and the Player Key is
    what disambiguates. Filtering to the players being compared is the
    intended way to work, and is why the key is checkboxes.
    """
    return {
        player: PLAYER_PALETTE[i % len(PLAYER_PALETTE)]
        for i, player in enumerate(sorted(known_player_pool))
    }


def players_sharing_a_colour(players: List[str], color_map: dict) -> List[str]:
    """Players on screen who collide with another on screen. Empty
    almost always; non-empty is worth saying out loud rather than
    letting two lines quietly look like one."""
    seen: dict = {}
    clashing = set()
    for player in players:
        colour = color_map.get(player)
        if colour is None:
            continue
        if colour in seen:
            clashing.update({seen[colour], player})
        seen[colour] = player
    return sorted(clashing)


def _clear_chart_encoding_widgets() -> None:
    """The chart encoding controls (Group by/Color by/Split into panels/
    order) are Streamlit widgets keyed only by action INDEX
    (qa_enc_position_i etc.), not by question -- so once a widget with a
    given key has been rendered, Streamlit keeps returning whatever the
    coach last picked for that key on every future rerun, regardless of
    the `index=` this code passes in. Resetting qa_encodings = {} alone
    only clears OUR OWN cache dict; it does nothing to that underlying
    widget state. Without also wiping it here, a manual override on some
    earlier question's action 0 chart would keep silently overriding
    default_encoding() (Game -> position, Player -> color, Metric ->
    facet) for every later question's action 0 too. Called every time
    qa_encodings is reset so the documented default is the REAL default,
    not one a stale click can quietly override."""
    for key in [k for k in st.session_state if k.startswith("qa_enc_")]:
        del st.session_state[key]

QA_SAMPLE_QUESTIONS = [
    "What is Sloan's Kills Per Set in the Vegas Aces game?",
    "How was Sloan's serving throughout all games?",
    "Who has the highest Kills Per Set in the Mavs 816 game?",
]


# ──────────────────────────────────────────────────────────────
# QUERY LAYER -- now shared, see query.py
# ──────────────────────────────────────────────────────────────
# Name resolution, pipeline preparation, action execution and the
# tidy/consolidate step are format-agnostic: they work against a Source
# and a KnowledgeTree, so the same code answers a question about a
# Huddle export and about a play-by-play file. They were duplicated
# here and in the DVW app; this is the surviving copy's import.
#
# Re-exported under the names app.py already published, because the test
# suite reaches for app.X and moving the bodies out must not move the
# names.
from query import (  # noqa: E402,F401
    _pipeline_referenced_metrics, _word_prefix_match, collect_leaf_labels,
    consolidate_action_results, execute_query_actions, find_leaf_by_exact_label,
    format_table as format_tidy_table, get_branches,
    metric_format_pattern as get_metric_format_pattern, prepare_pipeline_frame,
    resolve_game_hint, resolve_player_name, tidy_data,
)

# ──────────────────────────────────────────────────────────────
# SPEC EXECUTOR -- now shared, see evaluate_csv.py
# ──────────────────────────────────────────────────────────────
# Re-exported rather than re-implemented: these names were app.py's
# public surface for the CSV evaluator and are imported by name across
# the test suite, so moving the bodies out must not move the names.
from evaluate_csv import (  # noqa: E402,F401
    _ALLOWED_AST_NODES, _COLUMN_REF_PATTERN, _evaluate_spec_for_row,
    _find_leaf_by_label, _resolve_token_value, EvalResult, RowPick,
    build_substituted_expr, coerce_cell_to_float, compute_for_all_players,
    evaluate_spec, pick_example_row, referenced_columns, run_category_query,
    run_metric_query, safe_eval_arithmetic,
)



# ──────────────────────────────────────────────────────────────
# HELPERS 
# ──────────────────────────────────────────────────────────────

def get_branches(tree: KnowledgeTree) -> dict:
    root = tree.committed.get(tree.root_id)
    if not root:
        return {}
    branches = {}
    for child_id in root.children:
        node = tree.committed.get(child_id)
        if node and node.kind == NodeKind.BRANCH:
            branches[node.label] = child_id
    return branches


def collect_leaf_labels(tree: KnowledgeTree) -> Set[str]:
    return {n.label for n in tree.committed.values() if n.kind == NodeKind.LEAF}


def build_spec_from_expr(expr: str, description: str, extra_valid_refs: Optional[Set[str]] = None):
    match = _SINGLE_COL_RE.match(expr.strip())
    if match:
        return make_column_spec(match.group(1), extra_valid_refs=extra_valid_refs)
    return make_formula_spec(expr, description, extra_valid_refs=extra_valid_refs)


def find_leaves_by_label(tree: KnowledgeTree, name: str) -> List[Node]:
    needle = name.strip().lower()
    return [n for n in tree.committed.values()
            if n.kind == NodeKind.LEAF and n.label.strip().lower() == needle]


def find_leaf_by_exact_label(tree: KnowledgeTree, label: str) -> Optional[Node]:
    return next((n for n in tree.committed.values()
                 if n.kind == NodeKind.LEAF and n.label == label), None)


def reset_wizard() -> None:
    st.session_state.new_metric_step = "describe"
    st.session_state.new_metric_worked_example = None
    st.session_state.pop("review_parser_note", None)


def render_dependents(dependents) -> None:
    if dependents:
        st.caption("⚠️ Dependents: " + ", ".join(dependents))
    else:
        st.caption("No known dependents (usage logging not yet wired up).")


def render_worked_example(worked) -> None:
    if worked is None:
        st.info("No worked example yet.")
    elif worked.error:
        st.warning(f"Couldn't compute a worked example: {worked.error}")
    else:
        st.code(f"{worked.substituted_expr}  =  {worked.value}")
        st.caption(f"Computed against: {worked.row_label}")
        if worked.used_fallback_zero:
            st.warning(
                f"Note: {', '.join(worked.blank_columns)} was blank for this player and treated "
                "as 0 -- no single player in this data had every referenced stat recorded, so "
                "this is the closest available real example, not a fully clean one."
            )


def build_agraph_elements(tree: KnowledgeTree, expanded_branches: Set[str]):
    root = tree.committed.get(tree.root_id)
    nodes: List[AGNode] = []
    edges: List[AGEdge] = []
    if not root:
        return nodes, edges

    nodes.append(AGNode(id=root.node_id, label=root.label, shape="hexagon",
                         color=UMD_RED, size=30, font={"color": UMD_WHITE, "size": 16},
                         title=root.label))
    for branch_id in root.children:
        branch = tree.committed.get(branch_id)
        if not branch:
            continue
        is_expanded = branch.node_id in expanded_branches
        nodes.append(AGNode(
            id=branch.node_id, label=branch.label, shape="box", color=UMD_GOLD, size=24,
            font={"color": UMD_WHITE, "size": 14}, widthConstraint={"maximum": 140},
            title=f"{branch.label} — {len(branch.children)} metric(s) — click to "
                  f"{'collapse' if is_expanded else 'expand'}",
        ))
        edges.append(AGEdge(source=root.node_id, target=branch.node_id, color=UMD_RED))
        if not is_expanded:
            continue
        for leaf_id in branch.children:
            leaf = tree.committed.get(leaf_id)
            if not leaf:
                continue
            tooltip = leaf.spec.description if leaf.spec else leaf.label
            nodes.append(AGNode(id=leaf.node_id, label=leaf.label, shape="box",
                                 color=UMD_BLACK, size=18, font={"color": UMD_WHITE, "size": 12},
                                 widthConstraint={"maximum": 140}, title=tooltip))
            edges.append(AGEdge(source=branch.node_id, target=leaf.node_id, color=UMD_RED))
    return nodes, edges


def render_leaf_panel(tree: KnowledgeTree, leaf: Node, worked_example_df: Optional[pd.DataFrame]) -> None:
    path = " / ".join(tree.find_path(leaf.node_id, tree=tree.committed))
    st.markdown(f"#### 📊 {leaf.label}")
    st.caption(f"Path: {path}  ·  Authored by: {leaf.authored_by}")

    # A metric can be unusable in two different ways, and conflating
    # them was actively unhelpful: "no definition" is not the same
    # statement as "a definition this data cannot satisfy", and only the
    # second one can tell you what is actually missing.
    if leaf.spec is None:
        st.warning(
            "This metric has no stored definition at all. It was saved in a format "
            "this data source doesn't understand -- an event metric in the recruiting "
            "world, or the reverse."
        )
        return

    _errors = leaf.spec.validate()
    if _errors:
        st.error(
            "**This metric can't be evaluated against the currently loaded matches.**\n\n"
            + "\n".join(f"- {problem}" for problem in _errors)
        )
        st.caption(
            "The definition itself is intact and still saved. This is about the DATA "
            "loaded right now -- most often no matches loaded at all, in which case "
            "every metric here will report the same thing."
        )

    st.write(f"**Description:** {leaf.spec.description}")

    if worked_example_df is not None:
        st.markdown("**Quick example** (one real row):")
        render_worked_example(evaluate_spec(leaf.spec, worked_example_df, tree=tree))

    payload = leaf.spec.payload
    current_expr = f"[{payload['column_ref']}]" if payload.get("kind") == "column" else payload.get("formula_expr", "")
    is_editing = st.session_state.get("editing_node_id") == leaf.node_id

    if is_editing:
        st.text_input("Formula", key="edit_formula_expr")
        st.text_input("Description", key="edit_description")
        col_save, col_cancel = st.columns(2)
        if col_save.button("Save Edit", type="primary", key=f"save_{leaf.node_id}"):
            extra_refs = collect_leaf_labels(tree) - {leaf.label}
            new_spec = build_spec_from_expr(
                st.session_state.edit_formula_expr, st.session_state.edit_description,
                extra_valid_refs=extra_refs,
            )
            try:
                tree.edit_leaf_spec(leaf.node_id, new_spec)
                st.session_state.editing_node_id = None
                st.session_state.flash_message = (
                    "success", f"Staged edit to {leaf.label}. Review it in the diff below before merging.",
                )
                st.rerun()
            except ValueError as e:
                st.error(str(e))
        if col_cancel.button("Cancel", key=f"cancel_{leaf.node_id}"):
            st.session_state.editing_node_id = None
            st.rerun()
    else:
        st.code(current_expr)
        col_edit, col_delete = st.columns(2)
        if col_edit.button("Edit this metric", key=f"edit_btn_{leaf.node_id}"):
            st.session_state.editing_node_id = leaf.node_id
            st.session_state["edit_formula_expr"] = current_expr
            st.session_state["edit_description"] = payload.get("human_description", "")
            st.rerun()
        if col_delete.button("Delete this metric", key=f"delete_btn_{leaf.node_id}"):
            st.session_state.pending_delete_node_id = leaf.node_id
            st.session_state.pending_delete_label = leaf.label
            st.rerun()


def render_node_panel(tree: KnowledgeTree, selected_id: Optional[str],
                       worked_example_df: Optional[pd.DataFrame]) -> None:
    if not selected_id:
        st.caption("Click a branch to expand/collapse its metrics. Click a metric to view, edit, or delete it.")
        return
    node = tree.committed.get(selected_id)
    if node is None:
        st.info("That node isn't in the committed tree anymore. Click another node.")
    elif node.node_id == tree.root_id:
        st.markdown("#### Volleyball Metrics (root)")
        st.caption(f"{len(node.children)} skill-group branch(es). Click one to expand its metrics.")
    elif node.kind == NodeKind.BRANCH:
        st.markdown(f"#### 📁 {node.label}")
        st.caption(f"{len(node.children)} metric(s) in this category.")
        col_add, col_collapse = st.columns(2)
        if col_add.button(f"➕ Add a metric under {node.label}", key=f"add_child_{node.node_id}"):
            st.session_state.preferred_branch_for_new_metric = node.label
            st.session_state.new_metric_step = "describe"
            st.session_state.flash_message = (
                "success", f"Ready -- scroll down to \"Add a New Metric\" (defaults to \"{node.label}\").",
            )
            st.rerun()
        if col_collapse.button("❌ Collapse", key=f"collapse_{node.node_id}"):
            st.session_state.expanded_branches.discard(node.node_id)
            st.session_state.selected_node_id = None
            st.rerun()
    else:
        render_leaf_panel(tree, node, worked_example_df)


@st.fragment
def render_tree_graph(tree: KnowledgeTree, ws) -> None:
    col_l, col_mid, col_r = st.columns([1, 10, 1])
    with col_mid:
        ag_nodes, ag_edges = build_agraph_elements(tree, st.session_state.expanded_branches)
        ag_config = AGConfig(
            width=1300, height=850, directed=True, physics=False, hierarchical=True,
            direction="UD", sortMethod="directed", nodeSpacing=170, levelSeparation=190,
            interaction={"zoomView": False, "navigationButtons": True, "keyboard": True, "dragView": True},
        )
        clicked_node_id = agraph(nodes=ag_nodes, edges=ag_edges, config=ag_config)

    if clicked_node_id and clicked_node_id != st.session_state.last_graph_click:
        st.session_state.last_graph_click = clicked_node_id
        clicked_node = tree.committed.get(clicked_node_id)
        if clicked_node and clicked_node.kind == NodeKind.BRANCH:
            if clicked_node_id in st.session_state.expanded_branches:
                st.session_state.expanded_branches.discard(clicked_node_id)
                st.session_state.selected_node_id = None
            else:
                st.session_state.expanded_branches.add(clicked_node_id)
                st.session_state.selected_node_id = clicked_node_id
        else:
            st.session_state.selected_node_id = clicked_node_id
        st.rerun(scope="fragment")

    worked_example_df = ws.worked_example()
    render_node_panel(tree, st.session_state.selected_node_id, worked_example_df)


# ──────────────────────────────────────────────────────────────
# SESSION INITIALIZATION
# ──────────────────────────────────────────────────────────────

# ── which data world are we in ────────────────────────────────
# The workspace key is the ONLY thing the dropdown sets; everything
# else -- source, knowledge base, where that base is saved, whether
# per-set questions are answerable -- is rebuilt from it. Keeping one
# switch means the UI can never end up showing one world's metrics
# against another world's data.
if "workspace_key" not in st.session_state:
    st.session_state.workspace_key = workspace.RECRUITING


@st.cache_resource(show_spinner="Loading data...")
def _build_workspace(key: str, team: Optional[str]):
    """Cached on the key so switching back and forth does not re-fetch
    and re-parse. cache_resource rather than cache_data because a
    Workspace holds a live Source with its own parsed frames, which
    must not be copied per call."""
    if key == workspace.PLAYER_ANALYSIS:
        return workspace.build_player_analysis(team=team)
    return workspace.build_recruiting()


def _shorten_team(name: str) -> str:
    """Imported lazily: the .dvw stack has a hard dependency bound and
    importing it may fail, which must not matter to a workspace that
    never touches it. Falls back to the full name."""
    try:
        from source_dvw import shorten_team_name

        return shorten_team_name(name)
    except Exception:
        return name


def _switch_workspace() -> None:
    """
    Everything scoped to the OLD world has to go.

    A result set, a chart encoding, a player filter and a selected tree
    node all name things -- players, games, metrics, node ids -- that
    may not exist in the world being switched to. Leaving any of them
    behind produces a chart drawn against the wrong roster or an editor
    pointed at a node from another tree, which is worse than an empty
    screen because it looks like an answer.

    The list itself lives in workspace.py so it can be tested without
    executing this script.
    """
    workspace.clear_scoped_state(st.session_state)


def current_workspace():
    return _build_workspace(
        st.session_state.workspace_key, st.session_state.get("team_of_interest"),
    )


if "qa_trigger" not in st.session_state:
    st.session_state.new_metric_step = "describe"
    st.session_state.new_metric_worked_example = None
    st.session_state.pending_review_prefill = None
    st.session_state.pending_delete_node_id = None
    st.session_state.pending_delete_label = None
    st.session_state.editing_node_id = None
    st.session_state.flash_message = None
    st.session_state.qa_pending_query = None
    st.session_state.qa_trigger = False
    st.session_state.qa_last_decomposition = None
    st.session_state.qa_action_results = []
    st.session_state.qa_encodings = {}
    st.session_state.qa_player_filter = []
    st.session_state.expanded_branches = set()
    st.session_state.last_graph_click = None
    st.session_state.selected_node_id = None

if st.session_state.pending_review_prefill is not None:
    prefill = st.session_state.pending_review_prefill
    st.session_state["review_formula_expr"] = prefill["formula_expr"]
    st.session_state["review_human_description"] = prefill["human_description"]
    st.session_state["review_label"] = prefill["label"]
    st.session_state["review_branch_label"] = prefill["branch_label"]
    st.session_state["review_aliases"] = ", ".join(prefill.get("aliases", []))
    st.session_state["review_parser_note"] = prefill.get("parser_note", "")
    st.session_state.pending_review_prefill = None

if st.session_state.qa_pending_query is not None:
    st.session_state["qa_query_text"] = st.session_state.qa_pending_query
    st.session_state.qa_pending_query = None
    st.session_state.qa_trigger = True

# Computed once, before anything reads it: a workspace whose dependencies
# cannot be imported must not appear in the picker, and a stale
# workspace_key pointing at it must fall back rather than crash.
_AVAILABLE_KEYS = workspace.available_keys()
_DVW_REASON = workspace.dvw_unavailable_reason()
if st.session_state.workspace_key not in _AVAILABLE_KEYS:
    st.session_state.workspace_key = _AVAILABLE_KEYS[0]

ws = current_workspace()
tree = ws.tree
source = ws.source
known_games = ws.games()
known_player_pool = ws.players()
set_labels = ws.set_labels()

# Kept in session state because the tree editor mutates it in place and
# then asks the workspace to persist it.
st.session_state.tree = tree


# ──────────────────────────────────────────────────────────────
# HEADER + FLASH MESSAGE
# ──────────────────────────────────────────────────────────────

st.title("🏐 Volleyball Knowledge Base")

_teams = ws.teams()
col_source, col_team, col_caption = st.columns([2, 2, 4] if _teams else [2, 0.01, 6])
with col_source:
    st.selectbox(
        "Data source",
        _AVAILABLE_KEYS,
        format_func=lambda key: workspace.LABELS[key],
        key="workspace_key",
        on_change=_switch_workspace,
        help="Each source has its own knowledge base, because their metrics "
             "are built from different things -- export columns on one side, "
             "individual actions on the other.",
    )
if _teams:
    with col_team:
        # A .dvw records BOTH teams' actions, so "whose season is this"
        # is a real question rather than a preference: it decides whose
        # roster is analysed and which name each match is labelled with.
        st.selectbox(
            "Team", _teams, index=_teams.index(ws.team) if ws.team in _teams else 0,
            key="team_of_interest", on_change=_switch_workspace,
            format_func=_shorten_team,
            help="Scouting files contain both teams. This picks whose players "
                 "are analysed; the other side becomes the opponent.",
        )
with col_caption:
    st.caption(ws.caption)
    st.caption("Mixed-initiative knowledge base for volleyball analytics -- "
               "mechanically validated with EUD staging & diffs.")

for _warning in ws.warnings:
    st.warning(_warning)

if _DVW_REASON:
    st.info(
        "Play-by-play (.dvw) analysis is unavailable in this deployment, so only "
        f"the recruiting source is listed. Reason: {_DVW_REASON}"
    )

if st.session_state.flash_message:
    level, text = st.session_state.flash_message
    getattr(st, level)(text)
    st.session_state.flash_message = None

tab_qa, tab_kb = st.tabs(["Ask a Question", "Knowledge Base"])


# ──────────────────────────────────────────────────────────────
# TAB 1: ASK A QUESTION
# ──────────────────────────────────────────────────────────────

with tab_qa:
    st.header("Ask a Question")

    game_opponents = known_games
    st.multiselect(
        "Games (default scope for any part of your question that doesn't name one)",
        game_opponents, default=game_opponents, key="qa_games_multiselect",
    )

    # Offered only where the data can answer it. The CSV source declares
    # no Set axis (source.axes()), so this whole block is absent there
    # rather than disabled -- a control that cannot do anything is worse
    # than no control, because it implies the answer exists.
    if set_labels:
        st.multiselect(
            "Sets (leave empty for all sets together)",
            set_labels, default=[], key="qa_sets_multiselect",
            help="Narrowing to a set also rebases per-set rates: "
                 "\"Kills Per Set\" inside one set is that set's kills.",
        )


    with st.form("qa_query_form", clear_on_submit=False):
        st.text_input(
            "Ask about a metric, a player, and/or a game",
            key="qa_query_text",
            placeholder='e.g. "What is Sloan\'s Kills Per Set in the Vegas Aces game?"',
        )
        qa_submit = st.form_submit_button("Ask 🏐", type="primary")

    st.caption("Try one of these:")
    sample_cols = st.columns(len(QA_SAMPLE_QUESTIONS))
    for col, question in zip(sample_cols, QA_SAMPLE_QUESTIONS):
        if col.button(question, key=f"qa_sample_{question}"):
            st.session_state.qa_pending_query = question
            st.rerun()

    if qa_submit:
        st.session_state.qa_trigger = True

    if st.session_state.qa_trigger:
        st.session_state.qa_trigger = False
        query_text = st.session_state.get("qa_query_text", "")
        if not query_text or not query_text.strip():
            st.warning("Type a question first.")
        else:
            try:
                with st.spinner("Routing your question via local LLM..."):
                    decomposition = decompose_recruiting_query(
                        query_text, tree, game_opponents, supports_sets=ws.supports_sets,
                    )
            except LLMUnavailableError as e:
                st.session_state.qa_last_decomposition = None
                st.session_state.qa_action_results = []
                st.session_state.flash_message = ("error", f"LLM unreachable -- can't process your question. ({e})")
                st.rerun()

            selected_games = [g for g in known_games
                              if g in st.session_state.qa_games_multiselect]
            action_results = execute_query_actions(
                decomposition, tree, source, selected_games,
                selected_sets=st.session_state.get("qa_sets_multiselect") or None,
            )

            st.session_state.qa_last_decomposition = decomposition
            st.session_state.qa_action_results = action_results
            st.session_state.qa_encodings = {}
            _clear_chart_encoding_widgets()

            # Pre-tick whoever the question named, so the Player Key opens
            # on the player being asked about and comparing is one click --
            # tick a second player -- instead of finding and ticking the
            # subject first. Clearing to empty made the key say "no one
            # selected" for a question that was explicitly about someone.
            _asked_for = sorted({
                r["resolved_player"] for r in action_results
                if r.get("resolved_player")
            })
            for _player in known_player_pool:
                st.session_state.pop(f"qa_playerfilter_{_player}", None)
            for _player in _asked_for:
                st.session_state[f"qa_playerfilter_{_player}"] = True

            # Recorded as ALREADY applied: the filter-changed check below
            # compares the ticks against this and re-runs the whole query
            # when they differ. Leaving it empty would make every question
            # about a named player immediately re-run itself.
            st.session_state.qa_player_filter = _asked_for

    decomposition = st.session_state.qa_last_decomposition
    action_results = st.session_state.qa_action_results

    if decomposition is not None:
        st.divider()
        if decomposition.get("limitations"):
            st.warning(decomposition["limitations"])

        current_player_filter = sorted(
            p for p in known_player_pool if st.session_state.get(f"qa_playerfilter_{p}", False)
        )
        if current_player_filter != st.session_state.qa_player_filter:
            st.session_state.qa_player_filter = current_player_filter
            selected_games = [g for g in known_games
                              if g in st.session_state.qa_games_multiselect]
            action_results = execute_query_actions(
                decomposition, tree, source, selected_games,
                player_filter=current_player_filter or None,
                selected_sets=st.session_state.get("qa_sets_multiselect") or None,
            )
            st.session_state.qa_action_results = action_results
            st.session_state.qa_encodings = {}
            _clear_chart_encoding_widgets()

        valid_action_results = [item for item in action_results if "error" not in item and item.get("status") != "missing_metric"]
        missing_metric_actions = [item for item in action_results if item.get("status") == "missing_metric"]

        # ── 1. CONSOLIDATED TIDY TABLE (For valid metrics) ──
        st.subheader("📊 Consolidated Analytical Matrix")
        st.caption("💡 Click any column header to sort by it.")
        consolidated_df = consolidate_action_results(valid_action_results)

        if not consolidated_df.empty:
            formatted_table = format_tidy_table(consolidated_df, tree)
            st.dataframe(formatted_table, use_container_width=True, hide_index=True)

            with st.expander("💡 Table Provenance & Metric Contracts"):
                st.markdown("This view consolidates data computed from the **Committed Knowledge Base**: ")
                for col in consolidated_df.columns:
                    if col in ["Player", "Game"]:
                        continue
                    leaf = find_leaf_by_exact_label(tree, str(col))
                    if leaf and leaf.spec:
                        payload = leaf.spec.payload
                        expr = f"[{payload['column_ref']}]" if payload.get("kind") == "column" else payload.get("formula_expr")
                        st.markdown(f"- **{col}**: `{expr}`  *(Authored by: {leaf.authored_by})*")
                        st.caption(f"  Description: {leaf.spec.description}")
        elif not missing_metric_actions:
            unrecognized = decomposition.get("unrecognized_terms") or []
            if unrecognized:
                terms = ", ".join(f"'{t}'" for t in unrecognized)
                st.error(f"I didn't recognize {terms} as a stat or metric -- did you mean something else?")
            else:
                # Every action was dropped, or every one produced an
                # empty frame. The router's own account of what it did
                # is the only thing that can distinguish those, and it
                # was previously discarded -- leaving a message that
                # says something went wrong and nothing about what.
                st.info("No computable values returned for this combination.")
                _limitations = (decomposition.get("limitations") or "").strip()
                if _limitations:
                    st.caption(f"Why: {_limitations}")
                if not decomposition.get("actions"):
                    st.caption(
                        "The question produced no runnable actions at all — so this is "
                        "about how it was interpreted, not about the data."
                    )

        # ── 2. IN-SITU KNOWLEDGE BASE AUTHORING (For missing metrics) ──
        for item in missing_metric_actions:
            action = item["action"]
            missing_label = item["raw_metric_name"]

            st.divider()

            if item["is_gibberish"]:
                st.error(f"I didn't recognize **'{missing_label}'** as a stat or metric -- did you mean something else?")
                st.caption("This didn't look like a plausible volleyball stat, so no formula was drafted for it.")
                continue

            st.warning(f"⚠️ **Missing Metric Detected:** The metric **'{missing_label}'** is not in your Knowledge Base.")

            try:
                draft_proposal = parse_phrase_to_formula_llm(missing_label, tree)
            except LLMUnavailableError:
                draft_proposal = parse_phrase_to_formula(missing_label)

            col_decomp, col_draft = st.columns([1, 1])

            with col_decomp:
                st.markdown("**🔍 LLM Query Intent Decomposition:**")
                st.write(f"- **Requested Stat:** `{missing_label}`")
                st.write(f"- **Target Player:** {action.get('player') or 'All Players'}")
                st.write(f"- **Target Game:** {action.get('game_hint') or 'All Games'}")
                st.caption("The router recognized what you were asking for, but found no matching metric contract in `tree.committed`.")

            with col_draft:
                st.markdown("**🤖 AI Proposed Candidate Contract:**")
                if draft_proposal.matched:
                    proposed_expr = f"[{draft_proposal.column_ref}]" if draft_proposal.spec_kind == "column" else draft_proposal.formula_expr
                    st.code(f"Formula: {proposed_expr}")
                    st.write(f"**Description:** {draft_proposal.human_description}")
                    st.write(f"**Suggested Group:** {draft_proposal.suggested_branch_group or 'Attack'}")

                    if st.button(f"🚀 Pre-fill '{draft_proposal.suggested_label}' in Knowledge Base Tab", type="primary", key=f"prefill_{missing_label}"):
                        branches = get_branches(tree)
                        target_branch = draft_proposal.suggested_branch_group if draft_proposal.suggested_branch_group in branches else next(iter(branches), None)

                        st.session_state.pending_review_prefill = {
                            "formula_expr": proposed_expr,
                            "human_description": draft_proposal.human_description,
                            "label": draft_proposal.suggested_label,
                            "branch_label": target_branch,
                            "aliases": draft_proposal.suggested_aliases,
                            "parser_note": "Generated from unmatched Q&A query intent.",
                        }
                        st.session_state.new_metric_step = "review"
                        st.session_state.flash_message = (
                            "info",
                            f"Pre-filled '{draft_proposal.suggested_label}'! Switch to the 'Knowledge Base' tab below to review and stage it."
                        )
                        st.rerun()
                else:
                    st.error(f"Could not automatically draft a formula: {draft_proposal.message}")
                    st.caption("You can manually define this metric in the Knowledge Base tab.")

        # ── 3. VISUALIZATION CHARTS ──
        if valid_action_results:
            with st.expander("📈 Interactive Visualizations", expanded=True):
                qa_encodings = st.session_state.qa_encodings
                player_color_map = get_player_color_map(known_player_pool)

                col_charts, col_key = st.columns([5, 1])
                with col_key:
                    st.markdown("**Player Key**")
                    _clashing = players_sharing_a_colour(
                        sorted(player_color_map), player_color_map
                    )
                    if _clashing:
                        st.caption(
                            f"⚠️ {len(_clashing)} players share a colour with another "
                            "(eight distinct hues are available). Tick only the players "
                            "you're comparing to keep them apart."
                        )
                    st.caption(
                        "Check a player to re-run this question for just them "
                        "(check more to compare several) -- no new question needed."
                    )
                    active_filter = st.session_state.qa_player_filter
                    if active_filter:
                        st.caption(f"🔎 Filtered to: **{', '.join(active_filter)}**")
                        if st.button("Clear player filter", key="qa_playerfilter_clear"):
                            for player in known_player_pool:
                                st.session_state.pop(f"qa_playerfilter_{player}", None)
                            st.rerun()
                    for player in sorted(player_color_map):
                        swatch = player_color_map[player]
                        col_check, col_label = st.columns([1, 5])
                        with col_check:
                            st.checkbox(
                                player, key=f"qa_playerfilter_{player}",
                                value=(player in active_filter), label_visibility="collapsed",
                            )
                        with col_label:
                            st.markdown(
                                f'<div style="display:flex;align-items:center;margin-bottom:4px;">'
                                f'<span style="display:inline-block;width:14px;height:14px;'
                                f'background-color:{swatch};border:1px solid {UMD_WHITE};margin-right:6px;">'
                                f'</span><span style="font-size:0.85em;color:{UMD_WHITE};">{player}</span></div>',
                                unsafe_allow_html=True,
                            )

                with col_charts:
                    for i, item in enumerate(valid_action_results):
                        action = item["action"]
                        title = action.get("title") or action.get("metric_of_interest") or action.get("skill_group") or f"Chart {i + 1}"
                        result_df = item["result_df"]

                        if item.get("pipeline"):
                            st.caption(f"🔧 Pipeline: {describe_pipeline(item['pipeline'])}")
                        for note in item.get("pipeline_notes", []):
                            st.warning(note)

                        if result_df.empty or result_df["Value"].notna().sum() == 0:
                            continue

                        st.markdown(f"**{title}**")
                        encoding = qa_encodings.get(i) or default_encoding(result_df)
                        options = slot_options(result_df)

                        def _fmt(axis):
                            return axis or "(none)"

                        col_pos, col_color, col_facet = st.columns(3)
                        with col_pos:
                            chosen_position = st.selectbox(
                                "Group by", options, index=options.index(encoding.position),
                                format_func=_fmt, key=f"qa_enc_position_{i}",
                            )
                        with col_color:
                            chosen_color = st.selectbox(
                                "Color by", options, index=options.index(encoding.color),
                                format_func=_fmt, key=f"qa_enc_color_{i}",
                            )
                        with col_facet:
                            chosen_facet = st.selectbox(
                                "Split into panels by", options, index=options.index(encoding.facet),
                                format_func=_fmt, key=f"qa_enc_facet_{i}",
                            )

                        if chosen_position != encoding.position:
                            encoding = reconcile_encoding(encoding, "position", chosen_position)
                        if chosen_color != encoding.color:
                            encoding = reconcile_encoding(encoding, "color", chosen_color)
                        if chosen_facet != encoding.facet:
                            encoding = reconcile_encoding(encoding, "facet", chosen_facet)

                        if encoding.position is not None:
                            order_labels = ["value", "original"]
                            chosen_order = st.radio(
                                f"{encoding.position} order", order_labels, index=order_labels.index(encoding.game_order),
                                horizontal=True, key=f"qa_enc_gameorder_{i}",
                                format_func=lambda o: (
                                    "By value" if o == "value"
                                    else ("Chronological" if encoding.position == "Game"
                                          else "Natural order")
                                ),
                            )
                            if chosen_order != encoding.game_order:
                                encoding = set_game_order(encoding, chosen_order)

                        qa_encodings[i] = encoding
                        action_metric = action.get("metric_of_interest")

                        panels = render_encoded_panels(
                            result_df, encoding, value_col="Value",
                            color_map=player_color_map,
                            # The source's own game order is chronological;
                            # the result frame's is not (rows come out
                            # sorted by identity).
                            position_order=known_games if encoding.position == "Game" else None,
                        )
                        if not panels:
                            st.info("No computable values for this chart.")
                            continue

                        panels_per_row = 2 if len(panels) > 1 else 1
                        for row_start in range(0, len(panels), panels_per_row):
                            row_panels = panels[row_start:row_start + panels_per_row]
                            row_cols = st.columns(panels_per_row)
                            for col, (panel_title, fig) in zip(row_cols, row_panels):
                                panel_key = f"qa_chart_{i}" if len(panels) == 1 else f"qa_chart_{i}_{panel_title}"
                                panel_title_display = title if len(panels) == 1 else f"{title} — {panel_title}"
                                fig.update_layout(**PLOTLY_BASE, title=panel_title_display, showlegend=False)
                                
                                with col:
                                    st.plotly_chart(fig, use_container_width=True, key=panel_key)

        # ── 4. ROUTING DETAILS ──
        with st.expander("⚙️ LLM Router Decomposition"):
            st.write(f"**Intent Summary:** {decomposition.get('intent_summary', '')}")
            st.write(f"**Reasoning:** {decomposition.get('reasoning', '')}")
            st.json(decomposition)


# ──────────────────────────────────────────────────────────────
# TAB 2: KNOWLEDGE BASE
# ──────────────────────────────────────────────────────────────

with tab_kb:
    st.header(f"Current Metrics Tree — {workspace.LABELS[ws.key]}")

    # WHICH knowledge base this is, stated rather than implied. The two
    # trees deliberately share branch NAMES ("Attack", "Serve",
    # "Receive"...) so the router's synonyms resolve a category question
    # in either world -- which means they look nearly identical on
    # screen. Naming the source, the file and the metric count is what
    # makes "am I editing the right one" answerable at a glance instead
    # of by opening a metric and inspecting its definition.
    _leaf_count = sum(1 for _n in tree.committed.values() if _n.kind == NodeKind.LEAF)
    st.caption(
        f"{_leaf_count} committed metrics · saved to `{ws.tree_path}` in the private "
        f"VolleyData repo · updates on merge"
    )
    if ws.supports_sets:
        st.caption(
            "Metrics here are **filters over individual actions** "
            "(e.g. skill=Attack, evaluation_code=#), so they can be broken down by set."
        )
    else:
        st.caption(
            "Metrics here are **match-export columns and arithmetic over them**, "
            "so there is no per-set detail to break down."
        )

    render_tree_graph(tree, ws)

    with st.expander("Delete a metric by name"):
        delete_query = st.text_input("Metric name (exact label)", key="delete_by_name_query")
        if st.button("Find"):
            matches = find_leaves_by_label(tree, delete_query)
            if not matches:
                st.session_state.delete_by_name_matches = []
                st.error(f"No committed metric named '{delete_query}'.")
            else:
                st.session_state.delete_by_name_matches = [m.node_id for m in matches]

        for match_id in st.session_state.get("delete_by_name_matches", []):
            match_node = tree.committed.get(match_id)
            if not match_node:
                continue
            match_path = " / ".join(tree.find_path(match_id, tree=tree.committed))
            col_match_label, col_match_delete = st.columns([4, 1])
            col_match_label.markdown(f"- **{match_node.label}** (`{match_path}`)")
            if col_match_delete.button("Delete", key=f"byname_delete_{match_id}"):
                st.session_state.pending_delete_node_id = match_id
                st.session_state.pending_delete_label = match_node.label
                st.session_state.delete_by_name_matches = []
                st.rerun()

    with st.expander("Browse as a plain list"):
        root = tree.committed.get(tree.root_id)
        if root:
            for branch_id in root.children:
                branch = tree.committed.get(branch_id)
                if not branch:
                    continue
                st.markdown(f"**{branch.label}**  ({len(branch.children)} metrics)")
                for leaf_id in branch.children:
                    leaf = tree.committed.get(leaf_id)
                    if not leaf:
                        continue
                    desc = leaf.spec.description if leaf.spec else ""
                    col_label, col_delete = st.columns([5, 1])
                    col_label.markdown(f"- **{leaf.label}** — {desc}")
                    if col_delete.button("Delete", key=f"delete_{leaf_id}"):
                        st.session_state.pending_delete_node_id = leaf_id
                        st.session_state.pending_delete_label = leaf.label
                        st.rerun()

    if st.session_state.pending_delete_node_id:
        st.warning(
            f"Delete **{st.session_state.pending_delete_label}**? This stages removal -- "
            "you must click **Merge All Staged Changes** below to finalize."
        )
        col_yes, col_cancel = st.columns(2)
        if col_yes.button("Yes, stage this deletion", type="primary"):
            tree.delete_node(st.session_state.pending_delete_node_id)
            st.session_state.flash_message = (
                "success", f"Staged deletion of {st.session_state.pending_delete_label}."
            )
            st.session_state.pending_delete_node_id = None
            st.session_state.pending_delete_label = None
            st.rerun()
        if col_cancel.button("Cancel"):
            st.session_state.pending_delete_node_id = None
            st.session_state.pending_delete_label = None
            st.rerun()

    with st.expander("Raw text view (tree.render_tree())"):
        st.code(tree.render_tree())

    st.divider()

    st.header("Add a New Metric")

    if st.session_state.new_metric_step == "describe":
        phrase = st.text_input(
            "Describe the metric in plain language",
            key="describe_phrase",
            placeholder='e.g. "net kills per set"',
        )
        if st.button("Interpret", type="primary"):
            parser_note = ""
            try:
                result = parse_phrase_to_formula_llm(phrase, tree)
                parser_note = "Parsed via LLM."
            except LLMUnavailableError:
                result = parse_phrase_to_formula(phrase)
                parser_note = "⚠️ LLM unreachable -- used rule-based fallback."

            if not result.matched:
                st.error(result.message)
                if parser_note:
                    st.caption(parser_note)
                st.caption("Rule-based fallback understands: " + "; ".join(f'"{p}"' for p in EXAMPLE_PHRASES))
            else:
                expr = f"[{result.column_ref}]" if result.spec_kind == "column" else result.formula_expr
                spec = build_spec_from_expr(expr, result.human_description, extra_valid_refs=collect_leaf_labels(tree))
                worked_example_df = ws.worked_example()
                st.session_state.new_metric_worked_example = (
                    evaluate_spec(spec, worked_example_df, tree=tree) if worked_example_df is not None else None
                )

                branches = get_branches(tree)
                preferred_branch = st.session_state.pop("preferred_branch_for_new_metric", None)
                if preferred_branch in branches:
                    branch_label = preferred_branch
                elif result.suggested_branch_group in branches:
                    branch_label = result.suggested_branch_group
                else:
                    branch_label = next(iter(branches), None)
                st.session_state.pending_review_prefill = {
                    "formula_expr": expr,
                    "human_description": result.human_description,
                    "label": result.suggested_label,
                    "branch_label": branch_label,
                    "aliases": result.suggested_aliases,
                    "parser_note": parser_note,
                }
                st.session_state.new_metric_step = "review"
                st.rerun()

    elif st.session_state.new_metric_step == "review":
        branches = get_branches(tree)
        branch_labels = list(branches.keys())

        if st.session_state.get("review_parser_note"):
            st.caption(st.session_state.review_parser_note)

        st.text_input("Formula", key="review_formula_expr")
        st.text_input("Description", key="review_human_description")
        st.text_input("Metric label", key="review_label")
        st.text_input("Aliases (comma-separated)", key="review_aliases")
        st.selectbox("Target branch", branch_labels, key="review_branch_label")

        st.markdown("**Worked Example**")
        render_worked_example(st.session_state.new_metric_worked_example)

        col_recompute, col_confirm, col_cancel = st.columns(3)

        if col_recompute.button("Recompute Worked Example"):
            spec = build_spec_from_expr(st.session_state.review_formula_expr, st.session_state.review_human_description,
                                         extra_valid_refs=collect_leaf_labels(tree))
            worked_example_df = ws.worked_example()
            st.session_state.new_metric_worked_example = (
                evaluate_spec(spec, worked_example_df, tree=tree) if worked_example_df is not None else None
            )
            st.rerun()

        if col_confirm.button("Confirm & Stage", type="primary"):
            spec = build_spec_from_expr(st.session_state.review_formula_expr, st.session_state.review_human_description,
                                         extra_valid_refs=collect_leaf_labels(tree))
            branch_id = branches.get(st.session_state.review_branch_label)
            aliases = [a.strip() for a in st.session_state.review_aliases.split(",") if a.strip()]
            try:
                tree.add_node(
                    label=st.session_state.review_label,
                    kind=NodeKind.LEAF,
                    parent_id=branch_id,
                    to="staging",
                    spec=spec,
                    aliases=aliases,
                    authored_by="justin",
                )
                staged_label = st.session_state.review_label
                reset_wizard()
                st.session_state.flash_message = ("success", f"Staged: {staged_label}. Review below before merging.")
                st.rerun()
            except ValueError as e:
                st.error(str(e))

        if col_cancel.button("Cancel / Start Over"):
            reset_wizard()
            st.rerun()

    st.divider()

    st.header("Review Staged Changes (Diff)")

    diff = tree.compute_diff()

    if not diff:
        st.info("No staged changes.")
    else:
        order = [ChangeKind.ADD, ChangeKind.EDIT, ChangeKind.MOVE, ChangeKind.DELETE]
        labels = {ChangeKind.ADD: "Added", ChangeKind.EDIT: "Edited", ChangeKind.MOVE: "Moved", ChangeKind.DELETE: "Deleted"}
        for kind in order:
            group = [c for c in diff if c.kind == kind]
            if not group:
                continue
            st.subheader(f"{labels[kind]} ({len(group)})")
            for change in group:
                if kind == ChangeKind.ADD:
                    desc = change.after.spec.description if (change.after and change.after.spec) else "(branch)"
                    st.markdown(f"- **{change.label}** — {desc}")
                elif kind == ChangeKind.EDIT:
                    before_desc = change.before.spec.description if (change.before and change.before.spec) else change.before.label
                    after_desc = change.after.spec.description if (change.after and change.after.spec) else change.after.label
                    st.markdown(f"- **{change.label}**")
                    st.markdown(f"  - before: `{before_desc}`")
                    st.markdown(f"  - after: `{after_desc}`")
                    render_dependents(change.dependents)
                elif kind == ChangeKind.MOVE:
                    before_path = " / ".join(tree.find_path(change.node_id, tree=tree.committed))
                    after_path = " / ".join(tree.find_path(change.node_id, tree=tree.staging))
                    st.markdown(f"- **{change.label}**: `{before_path}` → `{after_path}`")
                    render_dependents(change.dependents)
                elif kind == ChangeKind.DELETE:
                    desc = change.before.spec.description if (change.before and change.before.spec) else "(branch)"
                    st.markdown(f"- **{change.label}** — {desc}")
                    render_dependents(change.dependents)

        if st.button("Merge All Staged Changes", type="primary", disabled=len(diff) == 0):
            n_changes = len(diff)
            tree.merge()
            flash_level, flash_text = "success", f"Merged {n_changes} change(s) into committed tree."
            try:
                ws.save_tree(tree)
            except RuntimeError as e:
                flash_level, flash_text = "warning", (
                    f"Merged for this session, but couldn't save to VolleyData "
                    f"(edit won't survive a redeploy): {e}"
                )
            st.session_state.flash_message = (flash_level, flash_text)
            st.rerun()