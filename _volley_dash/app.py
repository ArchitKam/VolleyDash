"""
_volley_dash/app.py
====================
The DataVolley (.dvw) twin of the CSV recruiting dashboard: browse an
event-grain knowledge tree, ask plain-language questions over it, and
author new metrics on the fly.

Run with:
    cd .../recruiting_analysis
    streamlit run _volley_dash/app.py

Feature-for-feature with the CSV app -- router-driven Q&A, consolidated
matrix, adaptive charts, in-situ metric authoring, staging/diff/merge --
because the parts that are format-agnostic are IMPORTED rather than
reimplemented: recruiting_tree (tree engine), recruiting_operations
(pipeline algebra), recruiting_encoding (chart encoding), and
recruiting_llm's query router. What is genuinely different -- the
source, the event metric kind, the evaluator, the metric author -- lives
in this package.

UI only. Every decision it makes is delegated to volley_query.py so the
logic is testable without a Streamlit runtime.
"""

import os
import sys

# This package's own directory, before any local import. `streamlit run`
# usually adds the script's directory to sys.path, but not every runner
# does (Streamlit's own AppTest harness does not), and relying on it
# made the app fail to start with ModuleNotFoundError depending purely
# on how it was launched.
_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
if _PACKAGE_DIR not in sys.path:
    sys.path.insert(0, _PACKAGE_DIR)

from typing import List, Optional

import pandas as pd
import streamlit as st
from streamlit_agraph import agraph, Node as AGNode, Edge as AGEdge, Config as AGConfig

import _parent_path  # noqa: F401
from recruiting_encoding import (
    default_encoding, reconcile_encoding, set_game_order, slot_options,
    render as render_encoded_panels,
)
from recruiting_llm import LLMUnavailableError, decompose_recruiting_query
from recruiting_operations import describe_pipeline
from recruiting_tree import ChangeKind, KnowledgeTree, NodeKind

from volley_llm import build_spec, parse_phrase_to_event_spec
from volley_query import (
    collect_leaf_labels, consolidate_action_results, execute_query_actions, find_leaf_by_exact_label,
    format_table, get_branches, known_games, known_players,
)
from volley_seed import seed_volley_tree
from volley_source_dvw import DvwSource, discover_matches
from volley_store import DEFAULT_DVW_DIR, DEFAULT_TREE_PATH, load_tree, save_tree

# Bridge Streamlit secrets into the environment before recruiting_llm
# reads them -- that module is Streamlit-agnostic by design and reads
# GROQ_API_KEY from os.environ only. Wrapped because st.secrets raises
# rather than behaving like an empty dict when no secrets.toml exists.
try:
    if "GROQ_API_KEY" not in os.environ and "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

st.set_page_config(page_title="VolleyDash — DataVolley", page_icon="🏐", layout="wide")

UMD_RED, UMD_BLACK, UMD_GOLD, UMD_WHITE = "#E21833", "#000000", "#B8860B", "#FFFFFF"
PLAYER_PALETTE = [UMD_RED, UMD_GOLD, "#8B0000", "#DAA520", UMD_WHITE, "#A9A9A9",
                   "#FF6B6B", "#F0C300", "#6B8E23", "#4682B4"]
PLOTLY_BASE = dict(
    plot_bgcolor=UMD_BLACK, paper_bgcolor=UMD_BLACK,
    font=dict(family="sans-serif", color=UMD_WHITE),
    margin=dict(l=40, r=20, t=40, b=40),
    xaxis=dict(color=UMD_WHITE, gridcolor="#333333"),
    yaxis=dict(color=UMD_WHITE, gridcolor="#333333"),
)

SAMPLE_QUESTIONS = [
    "Who has the highest Kills Per Set?",
    "How was Ally Williams' passing across all matches?",
    "What is Ajack Malual's Hitting Efficiency in the Minnesota game?",
]

st.markdown(f"""
<style>
h1, h2, h3 {{ border-bottom: 2px solid {UMD_RED}; padding-bottom: 0.2em; }}
hr {{ border-top: 1px solid {UMD_GOLD}; }}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# SOURCE + TREE
# ──────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Parsing DataVolley files…")
def build_source(directory: str, team: Optional[str]) -> DvwSource:
    """Cached on (directory, team): parsing 18 matches costs ~2.5s, and
    the script re-runs top to bottom on every widget interaction."""
    source = DvwSource(discover_matches(directory), team_of_interest=team)
    source.facts()  # force the parse inside the cached call
    return source


@st.cache_resource(show_spinner=False)
def available_teams(directory: str) -> List[str]:
    return DvwSource(discover_matches(directory)).teams()


def player_color_map(players: List[str]) -> dict:
    """Deterministic player -> colour so the same player is the same
    colour on every chart, rather than being re-picked per figure."""
    return {p: PLAYER_PALETTE[i % len(PLAYER_PALETTE)] for i, p in enumerate(sorted(players))}


def clear_chart_encoding_widgets() -> None:
    """The encoding dropdowns are widgets keyed by action index, so a
    manual override on one question would otherwise persist onto the
    next question's chart and silently beat default_encoding()."""
    for key in [k for k in st.session_state if k.startswith("vd_enc_")]:
        del st.session_state[key]


st.title("🏐 VolleyDash — DataVolley")
st.caption("Event-grain recruiting knowledge base over DataVolley .dvw play-by-play.")

dvw_dir = st.sidebar.text_input("DataVolley folder", value=DEFAULT_DVW_DIR, key="vd_dir")
match_paths = discover_matches(dvw_dir)
if not match_paths:
    st.error(f"No .dvw files found in `{dvw_dir}`.")
    st.stop()

teams = available_teams(dvw_dir)
default_team_index = teams.index("University of Maryland") if "University of Maryland" in teams else 0
team = st.sidebar.selectbox(
    "Team of interest", teams, index=default_team_index, key="vd_team",
    help="A .dvw records BOTH teams' actions; without a team, 'every player' means both benches.",
)
st.sidebar.caption(f"{len(match_paths)} match file(s) found.")

source = build_source(dvw_dir, team)

if "vd_tree" not in st.session_state or st.session_state.get("vd_tree_team") != team:
    loaded = load_tree(source.schema)
    if loaded is not None:
        tree = loaded
    else:
        tree, _ = seed_volley_tree(source.schema)
    st.session_state.vd_tree = tree
    st.session_state.vd_tree_team = team
    st.session_state.vd_decomposition = None
    st.session_state.vd_results = []
    st.session_state.vd_encodings = {}
    st.session_state.vd_player_filter = []
    st.session_state.vd_step = "describe"
    st.session_state.vd_prefill = None
    st.session_state.vd_flash = None
    st.session_state.vd_expanded = set()
    st.session_state.vd_selected_node = None
    st.session_state.vd_last_click = None
    st.session_state.vd_pending_delete = None

tree: KnowledgeTree = st.session_state.vd_tree
roster = known_players(source)
all_games = known_games(source)

if st.session_state.vd_prefill is not None:
    prefill = st.session_state.vd_prefill
    st.session_state["vd_review_label"] = prefill["label"]
    st.session_state["vd_review_description"] = prefill["description"]
    st.session_state["vd_review_branch"] = prefill["branch"]
    st.session_state["vd_review_aliases"] = ", ".join(prefill.get("aliases", []))
    st.session_state["vd_review_payload"] = prefill["payload"]
    st.session_state.vd_prefill = None

if st.session_state.vd_flash:
    level, text = st.session_state.vd_flash
    getattr(st, level)(text)
    st.session_state.vd_flash = None

tab_qa, tab_kb = st.tabs(["Ask a Question", "Knowledge Base"])


# ──────────────────────────────────────────────────────────────
# TAB 1 — ASK A QUESTION
# ──────────────────────────────────────────────────────────────

with tab_qa:
    st.header("Ask a Question")

    selected_games = st.multiselect(
        "Matches (default scope for any part of your question that doesn't name one)",
        all_games, default=all_games, key="vd_games",
    )

    with st.form("vd_query_form"):
        st.text_input("Ask about a metric, a player, and/or a match", key="vd_query",
                       placeholder='e.g. "Who has the highest Kills Per Set?"')
        submitted = st.form_submit_button("Ask 🏐", type="primary")

    st.caption("Try one of these:")
    for column, question in zip(st.columns(len(SAMPLE_QUESTIONS)), SAMPLE_QUESTIONS):
        if column.button(question, key=f"vd_sample_{question}"):
            st.session_state["vd_query"] = question
            submitted = True

    if submitted:
        query = st.session_state.get("vd_query", "")
        if not query.strip():
            st.warning("Type a question first.")
        else:
            try:
                with st.spinner("Routing your question…"):
                    decomposition = decompose_recruiting_query(query, tree, all_games)
            except LLMUnavailableError as error:
                decomposition = None
                st.error(f"LLM unreachable — can't process your question. ({error})")

            if decomposition is not None:
                st.session_state.vd_decomposition = decomposition
                st.session_state.vd_results = execute_query_actions(
                    decomposition, tree, source, selected_games=selected_games,
                )
                st.session_state.vd_encodings = {}
                st.session_state.vd_player_filter = []
                clear_chart_encoding_widgets()

    decomposition = st.session_state.vd_decomposition
    results = st.session_state.vd_results

    if decomposition is not None:
        st.divider()
        if decomposition.get("limitations"):
            st.warning(decomposition["limitations"])

        # Player Key toggles re-run the SAME decomposition for a new
        # player set -- no second trip through the router.
        current_filter = sorted(p for p in roster if st.session_state.get(f"vd_pf_{p}", False))
        if current_filter != st.session_state.vd_player_filter:
            st.session_state.vd_player_filter = current_filter
            st.session_state.vd_results = execute_query_actions(
                decomposition, tree, source, selected_games=selected_games,
                player_filter=current_filter or None,
            )
            st.session_state.vd_encodings = {}
            clear_chart_encoding_widgets()
            results = st.session_state.vd_results

        valid = [r for r in results if "error" not in r and r.get("status") != "missing_metric"]
        missing = [r for r in results if r.get("status") == "missing_metric"]

        st.subheader("📊 Consolidated Analytical Matrix")
        st.caption("💡 Click any column header to sort by it.")
        consolidated = consolidate_action_results(valid)

        if not consolidated.empty:
            st.dataframe(format_table(consolidated, tree), use_container_width=True, hide_index=True)
            with st.expander("💡 Table provenance & metric contracts"):
                for column in consolidated.columns:
                    if column in ("Player", "Game"):
                        continue
                    leaf = find_leaf_by_exact_label(tree, str(column))
                    if leaf and leaf.spec:
                        st.markdown(f"- **{column}** — `{leaf.spec.description}`  *(by {leaf.authored_by})*")
        elif not missing:
            unrecognized = decomposition.get("unrecognized_terms") or []
            if unrecognized:
                terms = ", ".join(f"'{t}'" for t in unrecognized)
                st.error(f"I didn't recognize {terms} as a stat or metric — did you mean something else?")
            else:
                st.info("No computable values returned for this combination.")

        # ── in-situ authoring for a metric that doesn't exist yet ──
        for item in missing:
            st.divider()
            label = item["raw_metric_name"]
            if item["is_gibberish"]:
                st.error(f"I didn't recognize **'{label}'** as a stat or metric.")
                st.caption("This didn't look like a plausible volleyball stat, so no metric was drafted.")
                continue

            st.warning(f"⚠️ **Missing metric:** '{label}' is not in your knowledge base.")
            try:
                proposal = parse_phrase_to_event_spec(label, source.schema, tree)
            except LLMUnavailableError as error:
                st.error(f"LLM unreachable, so no draft could be made. ({error})")
                continue

            if not proposal.matched:
                st.error(f"Could not draft a metric: {proposal.message}")
                continue

            spec = build_spec(proposal, source.schema, tree)
            errors = spec.validate()
            st.markdown("**🤖 Proposed candidate contract**")
            st.code(spec.description)
            if errors:
                # The whole point of the trust boundary: a proposal that
                # names something not in the data is rejected here, not
                # committed and silently returning zero.
                st.error("Rejected by validation: " + "; ".join(errors))
                continue

            if st.button(f"🚀 Pre-fill '{proposal.suggested_label}' in the Knowledge Base tab",
                          key=f"vd_prefill_{label}", type="primary"):
                branches = get_branches(tree)
                branch = (proposal.suggested_branch_group
                          if proposal.suggested_branch_group in branches else next(iter(branches), None))
                st.session_state.vd_prefill = {
                    "label": proposal.suggested_label,
                    "description": proposal.human_description or "",
                    "branch": branch,
                    "aliases": proposal.suggested_aliases,
                    "payload": spec.payload,
                }
                st.session_state.vd_step = "review"
                st.session_state.vd_flash = ("info", f"Pre-filled '{proposal.suggested_label}' — see the Knowledge Base tab.")
                st.rerun()

        # ── charts ──
        if valid:
            with st.expander("📈 Interactive visualizations", expanded=True):
                colors = player_color_map(roster)
                chart_column, key_column = st.columns([5, 1])

                with key_column:
                    st.markdown("**Player Key**")
                    st.caption("Check a player to re-run this question for just them.")
                    if st.session_state.vd_player_filter:
                        st.caption("🔎 " + ", ".join(st.session_state.vd_player_filter))
                        if st.button("Clear filter", key="vd_pf_clear"):
                            for player in roster:
                                st.session_state.pop(f"vd_pf_{player}", None)
                            st.rerun()
                    for player in sorted(roster):
                        check, label_col = st.columns([1, 5])
                        with check:
                            st.checkbox(player, key=f"vd_pf_{player}", label_visibility="collapsed")
                        label_col.markdown(
                            f'<div style="display:flex;align-items:center;">'
                            f'<span style="display:inline-block;width:12px;height:12px;'
                            f'background:{colors[player]};border:1px solid {UMD_WHITE};margin-right:6px;"></span>'
                            f'<span style="font-size:0.8em;">{player}</span></div>',
                            unsafe_allow_html=True,
                        )

                with chart_column:
                    for index, item in enumerate(valid):
                        action = item["action"]
                        title = (action.get("title") or action.get("metric_of_interest")
                                 or action.get("skill_group") or f"Chart {index + 1}")
                        frame = item["result_df"]

                        if item.get("pipeline"):
                            st.caption(f"🔧 Pipeline: {describe_pipeline(item['pipeline'])}")
                        for note in item.get("pipeline_notes", []):
                            st.warning(note)
                        if item.get("game_note"):
                            st.info(item["game_note"])

                        if frame.empty or frame["Value"].notna().sum() == 0:
                            st.info(f"No computable values for **{title}**.")
                            continue

                        st.markdown(f"**{title}**")
                        encoding = st.session_state.vd_encodings.get(index) or default_encoding(frame)
                        options = slot_options(frame)

                        position_col, color_col, facet_col = st.columns(3)
                        chosen_position = position_col.selectbox(
                            "Group by", options, index=options.index(encoding.position),
                            format_func=lambda a: a or "(none)", key=f"vd_enc_pos_{index}")
                        chosen_color = color_col.selectbox(
                            "Color by", options, index=options.index(encoding.color),
                            format_func=lambda a: a or "(none)", key=f"vd_enc_col_{index}")
                        chosen_facet = facet_col.selectbox(
                            "Split into panels by", options, index=options.index(encoding.facet),
                            format_func=lambda a: a or "(none)", key=f"vd_enc_fac_{index}")

                        if chosen_position != encoding.position:
                            encoding = reconcile_encoding(encoding, "position", chosen_position)
                        if chosen_color != encoding.color:
                            encoding = reconcile_encoding(encoding, "color", chosen_color)
                        if chosen_facet != encoding.facet:
                            encoding = reconcile_encoding(encoding, "facet", chosen_facet)

                        if encoding.position is not None:
                            order = st.radio(
                                f"{encoding.position} order", ["value", "original"],
                                index=["value", "original"].index(encoding.game_order),
                                horizontal=True, key=f"vd_enc_ord_{index}",
                                format_func=lambda o: "By value" if o == "value" else "Original order")
                            if order != encoding.game_order:
                                encoding = set_game_order(encoding, order)

                        st.session_state.vd_encodings[index] = encoding
                        panels = render_encoded_panels(frame, encoding, value_col="Value", color_map=colors)
                        if not panels:
                            st.info("No computable values for this chart.")
                            continue

                        per_row = 2 if len(panels) > 1 else 1
                        for start in range(0, len(panels), per_row):
                            row = panels[start:start + per_row]
                            for column, (panel_title, figure) in zip(st.columns(per_row), row):
                                heading = title if len(panels) == 1 else f"{title} — {panel_title}"
                                figure.update_layout(**PLOTLY_BASE, title=heading, showlegend=False)
                                column.plotly_chart(figure, use_container_width=True,
                                                     key=f"vd_chart_{index}_{panel_title}")

        with st.expander("⚙️ LLM router decomposition"):
            st.write(f"**Intent:** {decomposition.get('intent_summary', '')}")
            st.write(f"**Reasoning:** {decomposition.get('reasoning', '')}")
            # Pipelines hold Operation dataclasses, which st.json cannot
            # serialize and renders as raw reprs -- show them as their
            # plain-language description instead.
            printable = dict(decomposition)
            printable["actions"] = [
                {**a, "pipeline": describe_pipeline(a.get("pipeline") or [])}
                for a in decomposition.get("actions", [])
            ]
            st.json(printable)


# ──────────────────────────────────────────────────────────────
# TAB 2 — KNOWLEDGE BASE
# ──────────────────────────────────────────────────────────────

with tab_kb:
    st.header("Current metrics tree (committed)")
    st.caption(f"Persisted to `{os.path.basename(DEFAULT_TREE_PATH)}` on merge.")

    root = tree.committed.get(tree.root_id)
    nodes, edges = [], []
    if root:
        nodes.append(AGNode(id=root.node_id, label="Volley Metrics", shape="hexagon",
                             color=UMD_RED, size=30, font={"color": UMD_WHITE}))
        for branch_id in root.children:
            branch = tree.committed.get(branch_id)
            if not branch:
                continue
            expanded = branch.node_id in st.session_state.vd_expanded
            nodes.append(AGNode(id=branch.node_id, label=branch.label, shape="box", color=UMD_GOLD,
                                 size=22, font={"color": UMD_WHITE},
                                 title=f"{len(branch.children)} metric(s)"))
            edges.append(AGEdge(source=root.node_id, target=branch.node_id, color=UMD_RED))
            if not expanded:
                continue
            for leaf_id in branch.children:
                leaf = tree.committed.get(leaf_id)
                if not leaf:
                    continue
                nodes.append(AGNode(id=leaf.node_id, label=leaf.label, shape="box", color=UMD_BLACK,
                                     size=16, font={"color": UMD_WHITE},
                                     title=leaf.spec.description if leaf.spec else leaf.label))
                edges.append(AGEdge(source=branch.node_id, target=leaf.node_id, color=UMD_RED))

    clicked = agraph(nodes=nodes, edges=edges, config=AGConfig(
        width=1200, height=600, directed=True, physics=False, hierarchical=True,
        direction="UD", nodeSpacing=160, levelSeparation=170,
        interaction={"zoomView": False, "navigationButtons": True, "dragView": True}))

    if clicked and clicked != st.session_state.vd_last_click:
        st.session_state.vd_last_click = clicked
        node = tree.committed.get(clicked)
        if node and node.kind == NodeKind.BRANCH:
            if clicked in st.session_state.vd_expanded:
                st.session_state.vd_expanded.discard(clicked)
            else:
                st.session_state.vd_expanded.add(clicked)
        st.session_state.vd_selected_node = clicked
        st.rerun()

    with st.expander("Browse as a plain list", expanded=not st.session_state.vd_expanded):
        if root:
            for branch_id in root.children:
                branch = tree.committed.get(branch_id)
                if not branch:
                    continue
                st.markdown(f"**{branch.label}** ({len(branch.children)} metrics)")
                for leaf_id in branch.children:
                    leaf = tree.committed.get(leaf_id)
                    if not leaf:
                        continue
                    label_col, delete_col = st.columns([6, 1])
                    label_col.markdown(
                        f"- **{leaf.label}** — {leaf.spec.description if leaf.spec else ''}")
                    if delete_col.button("Delete", key=f"vd_del_{leaf_id}"):
                        st.session_state.vd_pending_delete = (leaf_id, leaf.label)
                        st.rerun()

    if st.session_state.vd_pending_delete:
        node_id, label = st.session_state.vd_pending_delete
        st.warning(f"Delete **{label}**? This stages the removal — merge below to finalize.")
        yes, cancel = st.columns(2)
        if yes.button("Yes, stage this deletion", type="primary"):
            tree.delete_node(node_id)
            st.session_state.vd_pending_delete = None
            st.session_state.vd_flash = ("success", f"Staged deletion of {label}.")
            st.rerun()
        if cancel.button("Cancel"):
            st.session_state.vd_pending_delete = None
            st.rerun()

    st.divider()
    st.header("Add a new metric")

    if st.session_state.vd_step == "describe":
        phrase = st.text_input("Describe the metric in plain language", key="vd_describe",
                                placeholder='e.g. "attacks that got stuffed by the block"')
        if st.button("Interpret", type="primary"):
            try:
                proposal = parse_phrase_to_event_spec(phrase, source.schema, tree)
            except LLMUnavailableError as error:
                proposal = None
                st.error(f"LLM unreachable — metric authoring needs it. ({error})")

            if proposal is not None:
                if not proposal.matched:
                    st.error(proposal.message)
                else:
                    spec = build_spec(proposal, source.schema, tree)
                    errors = spec.validate()
                    if errors:
                        st.error("The proposal didn't validate: " + "; ".join(errors))
                    else:
                        branches = get_branches(tree)
                        branch = (proposal.suggested_branch_group
                                  if proposal.suggested_branch_group in branches
                                  else next(iter(branches), None))
                        st.session_state.vd_prefill = {
                            "label": proposal.suggested_label,
                            "description": proposal.human_description or "",
                            "branch": branch,
                            "aliases": proposal.suggested_aliases,
                            "payload": spec.payload,
                        }
                        st.session_state.vd_step = "review"
                        st.rerun()

    elif st.session_state.vd_step == "review":
        branches = get_branches(tree)
        st.text_input("Metric label", key="vd_review_label")
        st.text_input("Description", key="vd_review_description")
        st.text_input("Aliases (comma-separated)", key="vd_review_aliases")
        st.selectbox("Target branch", list(branches), key="vd_review_branch")
        st.markdown("**Contract**")
        st.json(st.session_state.get("vd_review_payload", {}))

        confirm, cancel = st.columns(2)
        if confirm.button("Confirm & stage", type="primary"):
            payload = st.session_state.get("vd_review_payload", {})
            from volley_event_spec import make_event_spec, make_measure_spec, make_metric_formula_spec
            if payload.get("kind") == "formula":
                spec = make_metric_formula_spec(payload["formula_expr"],
                                                 st.session_state.vd_review_description,
                                                 collect_leaf_labels(tree))
            elif payload.get("kind") == "measure":
                spec = make_measure_spec(payload["measure"], st.session_state.vd_review_description)
            else:
                spec = make_event_spec(payload.get("where", {}),
                                        st.session_state.vd_review_description, source.schema,
                                        aggregate=payload.get("aggregate", "count"))
            aliases = [a.strip() for a in st.session_state.vd_review_aliases.split(",") if a.strip()]
            try:
                tree.add_node(label=st.session_state.vd_review_label, kind=NodeKind.LEAF,
                               parent_id=branches[st.session_state.vd_review_branch], to="staging",
                               spec=spec, aliases=aliases, authored_by="coach")
                st.session_state.vd_step = "describe"
                st.session_state.vd_flash = ("success",
                    f"Staged {st.session_state.vd_review_label}. Review the diff below before merging.")
                st.rerun()
            except ValueError as error:
                st.error(str(error))

        if cancel.button("Cancel / start over"):
            st.session_state.vd_step = "describe"
            st.rerun()

    st.divider()
    st.header("Review staged changes (diff)")

    diff = tree.compute_diff()
    if not diff:
        st.info("No staged changes.")
    else:
        labels = {ChangeKind.ADD: "Added", ChangeKind.EDIT: "Edited",
                  ChangeKind.MOVE: "Moved", ChangeKind.DELETE: "Deleted"}
        for kind in (ChangeKind.ADD, ChangeKind.EDIT, ChangeKind.MOVE, ChangeKind.DELETE):
            group = [c for c in diff if c.kind == kind]
            if not group:
                continue
            st.subheader(f"{labels[kind]} ({len(group)})")
            for change in group:
                node = change.after or change.before
                description = node.spec.description if (node and node.spec) else "(branch)"
                st.markdown(f"- **{change.label}** — {description}")

        if st.button("Merge all staged changes", type="primary"):
            count = len(diff)
            tree.merge()
            try:
                save_tree(tree)
                st.session_state.vd_flash = ("success", f"Merged {count} change(s) and saved.")
            except Exception as error:
                st.session_state.vd_flash = ("warning",
                    f"Merged for this session, but saving failed: {error}")
            st.rerun()
