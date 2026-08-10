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
from typing import Any, FrozenSet, List, Optional, Set, Tuple

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
from recruiting_operations import Operation, Slice, run_pipeline, describe_pipeline
from recruiting_data_store import (
    GameInfo, get_games, load_game_df, save_committed_tree, load_committed_tree,
)

# Bridge Streamlit secrets into the environment before recruiting_llm.py
# reads them -- that module stays Streamlit-agnostic (so it's directly
# importable/testable, as test_integration.py already does) and reads
# GROQ_API_KEY from os.environ only. recruiting_data_store.py handles its
# own secrets (GITHUB_DATA_REPO/GITHUB_DATA_TOKEN) directly via st.secrets,
# so nothing to bridge for those. Wrapped in try/except since st.secrets
# raises (rather than behaving like an empty dict) when no secrets.toml
# exists at all -- e.g. local dev with credentials only in the shell env.
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

PLOTLY_BASE = dict(
    plot_bgcolor=UMD_BLACK, paper_bgcolor=UMD_BLACK,
    font=dict(family="sans-serif", color=UMD_WHITE),
    margin=dict(l=40, r=20, t=40, b=40),
    xaxis=dict(color=UMD_WHITE, gridcolor="#333333"),
    yaxis=dict(color=UMD_WHITE, gridcolor="#333333"),
)

QA_SAMPLE_QUESTIONS = [
    "What is Sloan's Kills Per Set in the Vegas Aces game?",
    "How was Sloan's serving throughout all games?",
    "Who has the highest Kills Per Set in the Mavs 816 game?",
]


# ──────────────────────────────────────────────────────────────
# GAME SUPPORT -- listing/fetching itself lives in recruiting_data_store.py
# (GameInfo, get_games, load_game_df); this is just the pure hint-matching
# logic, which never touches storage so it stays here.
# ──────────────────────────────────────────────────────────────

def _word_prefix_match(hint_words: List[str], name_words: List[str]) -> bool:
    return all(
        any(hw == nw or nw.startswith(hw) or hw.startswith(nw) for nw in name_words)
        for hw in hint_words
    )


def resolve_game_hint(hint: Optional[str], games: List[GameInfo]) -> List[GameInfo]:
    if not hint or not hint.strip() or not games:
        return []
    needle = hint.strip().lower()

    exact = [g for g in games if g.opponent.lower() == needle]
    if exact:
        return exact

    substring = [g for g in games if needle in g.opponent.lower() or g.opponent.lower() in needle]
    if substring:
        return substring

    needle_words = needle.split()
    prefix_hits = [g for g in games if _word_prefix_match(needle_words, g.opponent.lower().split())]
    if prefix_hits:
        return prefix_hits

    fuzzy = [g for g in games
             if difflib.SequenceMatcher(None, needle, g.opponent.lower()).ratio() >= 0.75]
    return fuzzy


# ──────────────────────────────────────────────────────────────
# SPEC EXECUTOR
# ──────────────────────────────────────────────────────────────

_COLUMN_REF_PATTERN = re.compile(r"\[([^\[\]]+)\]")


def coerce_cell_to_float(raw_value: Any) -> Optional[float]:
    if raw_value is None:
        return None
    if isinstance(raw_value, float) and math.isnan(raw_value):
        return None
    if isinstance(raw_value, str) and not raw_value.strip():
        return None
    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return None


def referenced_columns(spec: MetricSpec) -> List[str]:
    payload = spec.payload
    kind = payload.get("kind")
    if kind == "column":
        col = payload.get("column_ref")
        return [col] if col else []
    if kind == "formula":
        return _COLUMN_REF_PATTERN.findall(payload.get("formula_expr", ""))
    return []


@dataclass
class RowPick:
    row: Optional[pd.Series]
    row_label: Optional[str]
    used_fallback_zero: bool
    blank_columns: List[str] = field(default_factory=list)
    error: Optional[str] = None


def pick_example_row(df: pd.DataFrame, referenced_cols: List[str],
                      name_col: str = "Name") -> RowPick:
    if name_col not in df.columns:
        return RowPick(row=None, row_label=None, used_fallback_zero=False,
                        error=f"'{name_col}' column not found in the CSV.")

    player_rows = df[df[name_col].astype(str).str.strip().str.startswith("#")]
    if player_rows.empty:
        return RowPick(row=None, row_label=None, used_fallback_zero=False,
                        error="No player rows found (all rows looked like team aggregates).")

    missing_cols = [c for c in referenced_cols if c not in df.columns]
    if missing_cols:
        return RowPick(row=None, row_label=None, used_fallback_zero=False,
                        error=f"Column(s) not found in CSV: {', '.join(missing_cols)}")

    best_row = None
    best_blanks: List[str] = []
    for _, row in player_rows.iterrows():
        blanks = [c for c in referenced_cols if coerce_cell_to_float(row[c]) is None]
        if not blanks:
            return RowPick(row=row, row_label=str(row[name_col]),
                            used_fallback_zero=False, blank_columns=[])
        if best_row is None or len(blanks) < len(best_blanks):
            best_row, best_blanks = row, blanks

    return RowPick(row=best_row, row_label=str(best_row[name_col]),
                    used_fallback_zero=True, blank_columns=best_blanks)


_ALLOWED_AST_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.UAdd, ast.USub,
)


def safe_eval_arithmetic(expr: str) -> float:
    try:
        parsed = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Couldn't parse expression '{expr}': {e}")

    for node in ast.walk(parsed):
        if not isinstance(node, _ALLOWED_AST_NODES):
            raise ValueError(
                f"Expression '{expr}' contains a disallowed construct ({type(node).__name__})."
            )
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            raise ValueError(f"Expression '{expr}' contains a non-numeric literal ({node.value!r}).")

    return eval(compile(parsed, "<safe_eval_arithmetic>", "eval"))


def build_substituted_expr(formula_expr: str, row: pd.Series,
                            fallback_zero_cols: Optional[List[str]] = None) -> Tuple[str, List[str]]:
    fallback_zero_cols = set(fallback_zero_cols or [])
    missing: List[str] = []

    def _replace(match: "re.Match") -> str:
        col = match.group(1)
        if col not in row.index:
            missing.append(col)
            return "0"
        value = coerce_cell_to_float(row[col])
        if value is None:
            if col in fallback_zero_cols:
                return "0.0"
            missing.append(col)
            return "0"
        return repr(value)

    substituted = _COLUMN_REF_PATTERN.sub(_replace, formula_expr)
    return substituted, missing


@dataclass
class EvalResult:
    value: Optional[float]
    substituted_expr: Optional[str] = None
    row_label: Optional[str] = None
    missing_columns: List[str] = field(default_factory=list)
    used_fallback_zero: bool = False
    blank_columns: List[str] = field(default_factory=list)
    error: Optional[str] = None


def _find_leaf_by_label(tree: KnowledgeTree, label: str):
    for node in tree.committed.values():
        if node.kind == NodeKind.LEAF and node.label == label:
            return node
    return None


def _resolve_token_value(
    token: str, row: pd.Series, tree: Optional[KnowledgeTree],
    fallback_zero_cols: FrozenSet[str], _resolving: FrozenSet[str],
) -> Tuple[Optional[float], Optional[str]]:
    if token in row.index:
        value = coerce_cell_to_float(row[token])
        if value is not None:
            return value, None
        if token in fallback_zero_cols:
            return 0.0, None
        return None, f"'{token}' is blank for this row."

    if tree is not None:
        if token in _resolving:
            return None, f"circular metric reference through '{token}'"
        nested_node = _find_leaf_by_label(tree, token)
        if nested_node is not None and nested_node.spec is not None:
            nested = _evaluate_spec_for_row(
                nested_node.spec, row, tree=tree,
                fallback_zero_cols=fallback_zero_cols,
                _resolving=_resolving | {token},
            )
            if nested.error:
                return None, f"nested metric '{token}' failed: {nested.error}"
            return nested.value, None

    return None, f"'{token}' is not a real column or an existing metric"


def _evaluate_spec_for_row(
    spec: MetricSpec, row: pd.Series, tree: Optional[KnowledgeTree] = None,
    fallback_zero_cols: Optional[FrozenSet[str]] = None,
    _resolving: FrozenSet[str] = frozenset(),
) -> EvalResult:
    fallback_zero_cols = fallback_zero_cols or frozenset()
    kind = spec.payload.get("kind")

    if kind == "column":
        col = spec.payload["column_ref"]
        value, error = _resolve_token_value(col, row, tree, fallback_zero_cols, _resolving)
        return EvalResult(
            value=value, substituted_expr=f"[{col}] = {value if value is not None else 'N/A'}",
            used_fallback_zero=bool(fallback_zero_cols and col in fallback_zero_cols),
            blank_columns=list(fallback_zero_cols), error=error,
        )

    if kind == "formula":
        expr = spec.payload["formula_expr"]
        problems: List[str] = []

        def _replace(match: "re.Match") -> str:
            token = match.group(1)
            value, error = _resolve_token_value(token, row, tree, fallback_zero_cols, _resolving)
            if value is None:
                problems.append(error or f"'{token}' unavailable")
                return "0"
            return repr(value)

        substituted = _COLUMN_REF_PATTERN.sub(_replace, expr)
        if problems:
            return EvalResult(value=None, substituted_expr=substituted,
                               missing_columns=problems, error="; ".join(problems))
        try:
            value = safe_eval_arithmetic(substituted)
        except ZeroDivisionError:
            return EvalResult(value=None, substituted_expr=substituted,
                               error="Division by zero for this row.")
        except ValueError as e:
            return EvalResult(value=None, substituted_expr=substituted, error=str(e))
        return EvalResult(
            value=value, substituted_expr=substituted,
            used_fallback_zero=bool(fallback_zero_cols), blank_columns=list(fallback_zero_cols),
        )

    return EvalResult(value=None, error=f"Unknown spec kind '{kind}'.")


def evaluate_spec(spec: MetricSpec, df: pd.DataFrame, name_col: str = "Name",
                   tree: Optional[KnowledgeTree] = None) -> EvalResult:
    cols = referenced_columns(spec)
    if not cols:
        return EvalResult(value=None, error="Spec references no columns.")

    raw_cols = [c for c in cols if c in df.columns]
    pick = pick_example_row(df, raw_cols, name_col=name_col)
    if pick.error:
        return EvalResult(value=None, error=pick.error)

    result = _evaluate_spec_for_row(
        spec, pick.row, tree=tree,
        fallback_zero_cols=frozenset(pick.blank_columns) if pick.used_fallback_zero else frozenset(),
    )
    result.row_label = pick.row_label
    return result


def compute_for_all_players(spec: MetricSpec, df: pd.DataFrame,
                             tree: Optional[KnowledgeTree] = None,
                             name_col: str = "Name") -> List[Tuple[str, EvalResult]]:
    if name_col not in df.columns:
        return []
    player_rows = df[df[name_col].astype(str).str.strip().str.startswith("#")]
    results = []
    for _, row in player_rows.iterrows():
        result = _evaluate_spec_for_row(spec, row, tree=tree, fallback_zero_cols=frozenset())
        result.row_label = str(row[name_col])
        results.append((str(row[name_col]), result))
    return results


def resolve_player_name(hint: Optional[str], known_names: List[str]) -> Optional[str]:
    if not hint or not hint.strip() or not known_names:
        return None
    needle = hint.strip().lower()

    exact = [n for n in known_names if n.strip().lower() == needle]
    if exact:
        return exact[0]

    def _strip_jersey(name: str) -> str:
        return re.sub(r"^#\S*\s*", "", name).strip().lower()

    substring = [n for n in known_names
                 if needle in _strip_jersey(n) or _strip_jersey(n) in needle]
    if substring:
        return substring[0]

    fuzzy = [n for n in known_names
             if difflib.SequenceMatcher(None, needle, _strip_jersey(n)).ratio() >= 0.75]
    return fuzzy[0] if fuzzy else None


def run_metric_query(
    spec: MetricSpec, tree: KnowledgeTree,
    game_dfs: List[Tuple[GameInfo, pd.DataFrame]],
    player_name: Optional[str] = None, name_col: str = "Name",
) -> pd.DataFrame:
    records = []
    for game, df in game_dfs:
        for name, result in compute_for_all_players(spec, df, tree=tree, name_col=name_col):
            if player_name is not None and name != player_name:
                continue
            records.append({
                "Game": game.opponent, "Player": name,
                "Value": result.value, "Note": result.error or "",
            })
    return pd.DataFrame(records, columns=["Game", "Player", "Value", "Note"])


def run_category_query(
    tree: KnowledgeTree, branch_node_id: str,
    game_dfs: List[Tuple[GameInfo, pd.DataFrame]],
    player_name: Optional[str] = None, name_col: str = "Name",
) -> pd.DataFrame:
    branch = tree.committed.get(branch_node_id)
    if branch is None or branch.kind != NodeKind.BRANCH:
        return pd.DataFrame(columns=["Metric", "Game", "Player", "Value", "Note"])

    frames = []
    for leaf_id in branch.children:
        leaf = tree.committed.get(leaf_id)
        if not leaf or leaf.kind != NodeKind.LEAF or not leaf.spec:
            continue
        leaf_df = run_metric_query(leaf.spec, tree, game_dfs, player_name=player_name, name_col=name_col)
        leaf_df.insert(0, "Metric", leaf.label)
        frames.append(leaf_df)

    if not frames:
        return pd.DataFrame(columns=["Metric", "Game", "Player", "Value", "Note"])
    return pd.concat(frames, ignore_index=True)


def _pipeline_referenced_metrics(pipeline: List[Operation]) -> Set[str]:
    """Every metric NAME a pipeline's Slice predicates reference. Reduce/
    Rank/Compare only ever operate on an axis (Player/Game/Metric), never
    name a metric directly -- only a Slice's ValuePredicate can point at a
    metric OTHER than the one already being computed (e.g. slicing Passing
    by a Kills threshold, so Kills has to be fetched too even though the
    coach only wants to see Passing)."""
    referenced: Set[str] = set()
    for op in pipeline:
        if isinstance(op, Slice) and op.predicate is not None:
            referenced.add(op.predicate.metric)
    return referenced


def prepare_pipeline_frame(
    result_df: pd.DataFrame, pipeline: List[Operation], primary_metric_label: Optional[str],
    tree: KnowledgeTree, game_dfs: List[Tuple["GameInfo", pd.DataFrame]],
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Only called when an action's pipeline is non-empty (an action with no
    pipeline never reaches this, so a plain single-metric/category query's
    result_df is untouched -- byte-for-byte the same as before this
    feature existed).

    Ensures result_df carries a "Metric" column (tagging it with
    primary_metric_label when it doesn't already have one -- category
    results already do, from run_category_query), then fetches any
    ADDITIONAL metric the pipeline references via a cross-metric Slice
    predicate and concatenates it in, so run_pipeline always sees one
    consistent long frame no matter how many distinct metrics it actually
    touches.

    The extra fetch is always player-UNFILTERED (every player in
    game_dfs), matching how the caller already fetched result_df itself
    whenever a pipeline is present (see the action-execution loop in the
    Q&A tab): a Compare/Rank/Reduce across the Player axis needs the FULL
    player set to mean anything ("team average", "who's highest"), so
    player-narrowing the coach actually asked for is expressed as an
    explicit Slice(axis="Player", ...) step in the pipeline instead of a
    fetch-time filter -- see the caller for where that step comes from
    (either straight from the router, or auto-appended when the coach
    named a player and the pipeline didn't already narrow one).
    """
    notes: List[str] = []
    if "Metric" not in result_df.columns:
        result_df = result_df.copy()
        result_df.insert(0, "Metric", primary_metric_label)

    already_have = set(result_df["Metric"].dropna().unique())
    extra_needed = _pipeline_referenced_metrics(pipeline) - already_have

    frames = [result_df]
    for extra_label in sorted(extra_needed):
        extra_leaf = find_leaf_by_exact_label(tree, extra_label)
        if extra_leaf is None or extra_leaf.spec is None:
            notes.append(f"Pipeline referenced unknown metric '{extra_label}' -- skipped.")
            continue
        extra_df = run_metric_query(extra_leaf.spec, tree, game_dfs, player_name=None)
        extra_df.insert(0, "Metric", extra_label)
        frames.append(extra_df)

    combined = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    return combined, notes


# ──────────────────────────────────────────────────────────────
# SEMANTIC TIDY DATA & CONSOLIDATION ENGINE
# ──────────────────────────────────────────────────────────────

def tidy_data(df: pd.DataFrame, metric_label: Optional[str] = None) -> pd.DataFrame:
    """Reshapes long observations into a wide matrix (Player, Game, Metric columns)."""
    tidy = df.copy()
    if "Metric" not in tidy.columns:
        tidy.insert(0, "Metric", metric_label)
    tidy["Value"] = pd.to_numeric(tidy["Value"], errors="coerce")

    wide = tidy.pivot(index=["Player", "Game"], columns="Metric", values="Value")
    wide = wide.reset_index()
    wide.columns.name = None
    return wide.sort_values(["Player", "Game"], kind="stable").reset_index(drop=True)


def consolidate_action_results(action_results: List[dict]) -> pd.DataFrame:
    """Merges outputs across multiple query actions into a SINGLE consolidated wide table."""
    wide_frames = []

    for item in action_results:
        if "error" in item or item.get("status") == "missing_metric" or item.get("result_df") is None or item["result_df"].empty:
            continue
        
        res_df = item["result_df"]
        action = item["action"]
        metric_label = action.get("metric_of_interest")
        
        wide = tidy_data(res_df, metric_label=metric_label)
        if not wide.empty:
            wide_frames.append(wide)

    if not wide_frames:
        return pd.DataFrame()

    consolidated = wide_frames[0]
    for next_frame in wide_frames[1:]:
        metric_cols = [c for c in next_frame.columns if c not in ["Player", "Game"]]
        cols_to_use = ["Player", "Game"] + [c for c in metric_cols if c not in consolidated.columns]
        if len(cols_to_use) > 2:
            consolidated = pd.merge(consolidated, next_frame[cols_to_use], on=["Player", "Game"], how="outer")

    return consolidated.sort_values(["Player", "Game"], kind="stable").reset_index(drop=True)


def get_metric_format_pattern(tree: KnowledgeTree, metric_label: str) -> str:
    """Determines string formatting rules based on metric spec & metadata."""
    node = find_leaf_by_exact_label(tree, metric_label)
    lbl_lower = metric_label.lower()

    if "%" in lbl_lower or "pct" in lbl_lower or "percentage" in lbl_lower or "efficiency" in lbl_lower or ("rate" in lbl_lower and "success" in lbl_lower):
        return "{:.1%}"

    if node and node.spec:
        payload = node.spec.payload
        if payload.get("kind") == "column":
            col_ref = payload.get("column_ref")
            col_spec = RECRUITING_COLUMN_SCHEMA.get(col_ref)
            if col_spec:
                if col_spec.type == ColumnType.PERCENTAGE:
                    return "{:.1%}"
                elif col_spec.type in (ColumnType.RATE, ColumnType.RATING):
                    return "{:.2f}"
                elif col_spec.type == ColumnType.COUNT:
                    return "{:.0f}"

    if "per set" in lbl_lower or "/s" in lbl_lower or "rating" in lbl_lower or "rtg" in lbl_lower:
        return "{:.2f}"

    return "{:.2f}"


def format_tidy_table(df: pd.DataFrame, tree: KnowledgeTree, highlight: Optional[str] = None):
    """Applies semantic formatting, clean dash fills for missing values, and row highlights."""
    format_dict = {}
    for col in df.columns:
        if col in ["Player", "Game"]:
            continue
        format_dict[col] = get_metric_format_pattern(tree, str(col))

    def _row_style(row):
        hit = highlight and (row.get("Player") == highlight or row.get("Game") == highlight)
        return [f"background-color: {UMD_GOLD}; color: {UMD_BLACK}; font-weight: bold;" if hit else "" for _ in row]

    return df.style.format(format_dict, na_rep="—").apply(_row_style, axis=1)


# ──────────────────────────────────────────────────────────────
# HELPERS -- tree persistence and CSV listing/fetching now live entirely
# in recruiting_data_store.py (the one module that talks to VolleyData);
# imported at the top of this file.
# ──────────────────────────────────────────────────────────────

def _safe_get_games() -> List[GameInfo]:
    """get_games() talks to VolleyData over the network -- wrapped so a
    transient outage or misconfigured secret shows an empty game list
    instead of crashing the whole app (including at module-import time,
    which is what running this script or importing it for tests does)."""
    try:
        return get_games()
    except RuntimeError:
        return []


@st.cache_data
def get_known_player_pool() -> List[str]:
    pool: Set[str] = set()
    for game in _safe_get_games():
        df = load_game_df(game.path)
        if "Name" in df.columns:
            pool.update(n for n in df["Name"].astype(str) if n.strip().startswith("#"))
    return sorted(pool)


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
def render_tree_graph(tree: KnowledgeTree, known_games: List[GameInfo]) -> None:
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

    worked_example_df = load_game_df(known_games[0].path) if known_games else None
    render_node_panel(tree, st.session_state.selected_node_id, worked_example_df)


def render_metric_bar_chart(df: pd.DataFrame, title: str, highlight: Optional[str] = None) -> go.Figure:
    fig = go.Figure()
    valid = df[df["Value"].notna()]
    if valid.empty:
        return fig

    n_games = valid["Game"].nunique()
    n_players = valid["Player"].nunique()

    if n_players <= 1:
        colors = [UMD_GOLD if highlight and g == highlight else UMD_RED for g in valid["Game"]]
        fig.add_trace(go.Bar(x=valid["Game"], y=valid["Value"], marker_color=colors, customdata=valid["Player"]))
        fig.update_layout(xaxis_title="Game", yaxis_title=title, showlegend=False)
    elif n_games <= 1:
        colors = [UMD_GOLD if highlight and p == highlight else UMD_RED for p in valid["Player"]]
        fig.add_trace(go.Bar(x=valid["Player"], y=valid["Value"], marker_color=colors, customdata=valid["Player"]))
        fig.update_layout(xaxis_title="Player", yaxis_title=title, showlegend=False)
    else:
        for idx, player in enumerate(sorted(valid["Player"].unique())):
            sub = valid[valid["Player"] == player]
            color = UMD_GOLD if highlight and player == highlight else UMD_CYCLE[idx % len(UMD_CYCLE)]
            fig.add_trace(go.Bar(name=player, x=sub["Game"], y=sub["Value"], customdata=sub["Player"], marker_color=color))
        fig.update_layout(barmode="group", xaxis_title="Game", yaxis_title=title, showlegend=True)

    fig.update_layout(**PLOTLY_BASE, title=title)
    return fig


def render_category_bar_chart(df: pd.DataFrame, title: str, highlight: Optional[str] = None) -> go.Figure:
    fig = go.Figure()
    valid = df[df["Value"].notna()]
    if valid.empty:
        return fig

    n_games = valid["Game"].nunique()

    if n_games <= 1:
        colors = [UMD_GOLD if highlight and m == highlight else UMD_RED for m in valid["Metric"]]
        fig.add_trace(go.Bar(x=valid["Metric"], y=valid["Value"], marker_color=colors, customdata=valid["Metric"]))
        fig.update_layout(showlegend=False)
    else:
        for idx, game in enumerate(sorted(valid["Game"].unique())):
            sub = valid[valid["Game"] == game]
            fig.add_trace(go.Bar(name=game, x=sub["Metric"], y=sub["Value"], customdata=sub["Metric"],
                                  marker_color=UMD_CYCLE[idx % len(UMD_CYCLE)]))
        fig.update_layout(barmode="group", showlegend=True)

    fig.update_layout(**PLOTLY_BASE, title=title, xaxis_title="Metric", yaxis_title="Value")
    return fig


def extract_click_highlight(chart_click_key: str) -> Optional[str]:
    click_state = st.session_state.get(chart_click_key)
    if not click_state:
        return None
    points = (click_state.get("selection") or {}).get("points") or []
    if not points:
        return None
    pt = points[0]
    value = pt.get("customdata")
    if isinstance(value, list):
        value = value[0] if value else None
    return value if value is not None else pt.get("x")


# ──────────────────────────────────────────────────────────────
# SESSION INITIALIZATION
# ──────────────────────────────────────────────────────────────

if "tree" not in st.session_state:
    loaded_tree = load_committed_tree()
    _startup_save_warning = None
    if loaded_tree is not None:
        tree = loaded_tree
    else:
        tree, _ = seed_recruiting_tree()
        try:
            save_committed_tree(tree)
        except RuntimeError as e:
            # Run with the freshly-seeded tree in memory either way -- a
            # failed first-run save (VolleyData/secrets not set up yet)
            # shouldn't block the app from loading at all, just mean this
            # session's baseline won't be durably saved.
            _startup_save_warning = f"Couldn't save the seeded tree to VolleyData yet: {e}"
    st.session_state.tree = tree
    st.session_state.new_metric_step = "describe"
    st.session_state.new_metric_worked_example = None
    st.session_state.pending_review_prefill = None
    st.session_state.pending_delete_node_id = None
    st.session_state.pending_delete_label = None
    st.session_state.editing_node_id = None
    st.session_state.flash_message = ("warning", _startup_save_warning) if _startup_save_warning else None
    st.session_state.qa_pending_query = None
    st.session_state.qa_trigger = False
    st.session_state.qa_last_decomposition = None
    st.session_state.qa_action_results = []
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

tree = st.session_state.tree
known_games = _safe_get_games()
known_player_pool = get_known_player_pool()


# ──────────────────────────────────────────────────────────────
# HEADER + FLASH MESSAGE
# ──────────────────────────────────────────────────────────────

st.title("🏐 Recruiting Knowledge Base")
st.caption("Mixed-initiative knowledge base for volleyball analytics -- mechanically validated with EUD staging & diffs.")

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

    game_opponents = [g.opponent for g in known_games]
    st.multiselect(
        "Games (default scope for any part of your question that doesn't name one)",
        game_opponents, default=game_opponents, key="qa_games_multiselect",
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
                    decomposition = decompose_recruiting_query(query_text, tree, game_opponents)
            except LLMUnavailableError as e:
                st.session_state.qa_last_decomposition = None
                st.session_state.qa_action_results = []
                st.session_state.flash_message = ("error", f"LLM unreachable -- can't process your question. ({e})")
                st.rerun()

            selected_games = [g for g in known_games if g.opponent in st.session_state.qa_games_multiselect]
            # Lowercased for a case-insensitive membership check below -- the
            # router's own JSON isn't guaranteed to echo a term back with the
            # exact same casing it used for metric_of_interest (confirmed:
            # "Yendas" vs "yendas" in a real repair-step test).
            unrecognized_lower = {t.lower() for t in decomposition.get("unrecognized_terms", [])}
            action_results = []
            for action in decomposition.get("actions", []):
                is_category = bool(action.get("skill_group"))
                pipeline = action.get("pipeline") or []

                if is_category:
                    branch_id = get_branches(tree).get(action["skill_group"])
                    if branch_id is None:
                        action_results.append({"action": action, "error": "Category no longer exists in committed tree."})
                        continue
                else:
                    leaf = find_leaf_by_exact_label(tree, action.get("metric_of_interest"))
                    if leaf is None or leaf.spec is None:
                        raw_metric_name = action.get("metric_of_interest") or "Unknown Metric"
                        # Task 1.3's distinguishing heuristic: the router itself
                        # already tells us, structurally, whether it considered
                        # this term unrecognized (gibberish) vs a plausible new
                        # metric name it just made up in good faith -- trust
                        # THAT signal rather than re-guessing from the string
                        # ourselves (e.g. dictionary/edit-distance heuristics
                        # would just be a second, less-informed guess layered
                        # on top of the model's own stated confidence).
                        is_gibberish = raw_metric_name.strip().lower() in unrecognized_lower
                        action_results.append({
                            "action": action,
                            "status": "missing_metric",
                            "raw_metric_name": raw_metric_name,
                            "is_gibberish": is_gibberish,
                        })
                        continue

                resolved_player = resolve_player_name(action.get("player"), known_player_pool)

                game_note = None
                if action.get("game_hint"):
                    hint_games = resolve_game_hint(action["game_hint"], known_games)
                    if hint_games:
                        action_games = hint_games
                    else:
                        action_games = selected_games
                        game_note = f"Couldn't identify game '{action['game_hint']}' -- showing all selected games instead."
                else:
                    action_games = selected_games

                game_dfs = [(g, load_game_df(g.path)) for g in action_games]
                # Whenever a pipeline is present, fetch EVERY player -- a
                # Reduce/Rank/Compare across the Player axis (team average,
                # "who's highest") needs the full player set to be
                # meaningful; pre-filtering to resolved_player here would
                # make a "team average" equal that one player's own value.
                # Player-narrowing the coach actually asked for is applied
                # as an explicit pipeline step instead (below), which runs
                # AFTER any cross-player aggregation.
                fetch_player = None if pipeline else resolved_player
                if is_category:
                    result_df = run_category_query(tree, branch_id, game_dfs, player_name=fetch_player)
                else:
                    result_df = run_metric_query(leaf.spec, tree, game_dfs, player_name=fetch_player)

                # Pipeline post-processing -- ONLY touches the frame when a
                # pipeline was actually emitted; an action with none takes
                # this exact same path it always has (no Metric-column
                # insertion, no run_pipeline call at all, no player-fetch
                # change either, per fetch_player above), so a plain
                # single-metric/category query is byte-for-byte unchanged.
                pipeline_notes: List[str] = []
                if pipeline:
                    pipeline = list(pipeline)
                    has_player_slice = any(isinstance(op, Slice) and op.axis == "Player" for op in pipeline)
                    if resolved_player is not None and not has_player_slice:
                        # The coach named a specific player but the pipeline
                        # itself doesn't narrow to one -- narrow LAST, so
                        # whatever Compare/Rank/Reduce runs first still sees
                        # every player it needs to.
                        pipeline.append(Slice(axis="Player", keep=[resolved_player]))

                    # Same reasoning, Metric axis: a cross-metric Slice
                    # predicate (e.g. "her passing in games she had 10+
                    # kills") pulls in an EXTRA metric's rows alongside the
                    # one actually being asked about (see
                    # prepare_pipeline_frame) -- without narrowing back down,
                    # the final table would show both. Only for single-
                    # metric actions: a category action's whole point is
                    # showing every metric in it, so it's never auto-narrowed.
                    has_metric_slice = any(isinstance(op, Slice) and op.axis == "Metric" for op in pipeline)
                    primary_metric = action.get("metric_of_interest")
                    if not is_category and primary_metric and not has_metric_slice:
                        pipeline.append(Slice(axis="Metric", keep=[primary_metric]))

                    combined_df, fetch_notes = prepare_pipeline_frame(
                        result_df, pipeline, action.get("metric_of_interest"), tree, game_dfs,
                    )
                    result_df, run_notes = run_pipeline(combined_df, pipeline)
                    pipeline_notes = fetch_notes + run_notes

                action_results.append({
                    "action": action, "is_category": is_category, "resolved_player": resolved_player,
                    "action_games": action_games, "game_note": game_note, "result_df": result_df,
                    "pipeline": pipeline, "pipeline_notes": pipeline_notes,
                })

            st.session_state.qa_last_decomposition = decomposition
            st.session_state.qa_action_results = action_results

    decomposition = st.session_state.qa_last_decomposition
    action_results = st.session_state.qa_action_results

    if decomposition is not None:
        st.divider()
        if decomposition.get("limitations"):
            st.warning(decomposition["limitations"])

        valid_action_results = [item for item in action_results if "error" not in item and item.get("status") != "missing_metric"]
        missing_metric_actions = [item for item in action_results if item.get("status") == "missing_metric"]

        # ── 1. CONSOLIDATED TIDY TABLE (For valid metrics) ──
        st.subheader("📊 Consolidated Analytical Matrix")
        consolidated_df = consolidate_action_results(valid_action_results)

        if not consolidated_df.empty:
            chart_click_key = "qa_global_chart_click"
            highlight = extract_click_highlight(chart_click_key)
            formatted_table = format_tidy_table(consolidated_df, tree, highlight=highlight)
            st.dataframe(formatted_table, use_container_width=True, hide_index=True)

            if highlight:
                st.caption(f"🔗 Linked selection: **{highlight}**")

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
                # Entirely-unrecognized case ("sloan's yendas per game"): no
                # valid result, no plausible-but-uncommitted metric either --
                # an honest rejection per term, never an AI-drafted formula,
                # since there's nothing here that even LOOKED like a real
                # metric name to draft one for.
                terms = ", ".join(f"'{t}'" for t in unrecognized)
                st.error(f"I didn't recognize {terms} as a stat or metric -- did you mean something else?")
            else:
                st.info("No computable values returned for this combination.")

        # ── 2. IN-SITU KNOWLEDGE BASE AUTHORING (For missing metrics) ──
        for item in missing_metric_actions:
            action = item["action"]
            missing_label = item["raw_metric_name"]

            st.divider()

            if item["is_gibberish"]:
                # Task 1.3: the router itself flagged this exact term as
                # unrecognized (see is_gibberish's computation above) --
                # show an honest rejection, and do NOT call the metric
                # author here at all. Drafting a formula for a term the
                # router already said it didn't understand would be
                # exactly the hallucinated typo-correction the system
                # prompt's grounding rules exist to forbid.
                st.error(f"I didn't recognize **'{missing_label}'** as a stat or metric -- did you mean something else?")
                st.caption("This didn't look like a plausible volleyball stat, so no formula was drafted for it.")
                continue

            st.warning(f"⚠️ **Missing Metric Detected:** The metric **'{missing_label}'** is not in your Knowledge Base.")

            # Use LLM to draft a candidate proposal for this missing metric
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
                for i, item in enumerate(valid_action_results):
                    action = item["action"]
                    title = action.get("title") or action.get("metric_of_interest") or action.get("skill_group") or f"Chart {i + 1}"
                    result_df = item["result_df"]

                    # Routing-inspection moment for the pipeline itself (same
                    # spirit as the "LLM Router Decomposition" expander below):
                    # show what was actually computed before the coach trusts
                    # the number, and surface any note from fetching/running
                    # it (e.g. an unresolvable cross-metric predicate, or a
                    # pipeline step that emptied the result) even when that
                    # means the chart itself gets skipped just after this.
                    if item.get("pipeline"):
                        st.caption(f"🔧 Pipeline: {describe_pipeline(item['pipeline'])}")
                    for note in item.get("pipeline_notes", []):
                        st.warning(note)

                    if result_df.empty or result_df["Value"].notna().sum() == 0:
                        continue

                    is_category = item["is_category"]
                    sub_key = f"qa_chart_click_{i}"
                    sub_highlight = extract_click_highlight(sub_key)

                    if is_category and result_df["Player"].nunique() > 1:
                        st.caption(f"Chart for '{title}' omitted (multiple players in category query -- see table above).")
                    elif is_category:
                        fig = render_category_bar_chart(result_df, title=title, highlight=sub_highlight)
                        st.plotly_chart(fig, use_container_width=True, on_select="rerun", selection_mode=["points"], key=sub_key)
                    else:
                        fig = render_metric_bar_chart(result_df, title=title, highlight=sub_highlight)
                        st.plotly_chart(fig, use_container_width=True, on_select="rerun", selection_mode=["points"], key=sub_key)

        # ── 4. ROUTING DETAILS ──
        with st.expander("⚙️ LLM Router Decomposition"):
            st.write(f"**Intent Summary:** {decomposition.get('intent_summary', '')}")
            st.write(f"**Reasoning:** {decomposition.get('reasoning', '')}")
            st.json(decomposition)


# ──────────────────────────────────────────────────────────────
# TAB 2: KNOWLEDGE BASE
# ──────────────────────────────────────────────────────────────

with tab_kb:
    st.header("Current Metrics Tree (committed)")
    st.caption("Persisted to the private VolleyData repo — updates on merge.")

    render_tree_graph(tree, known_games)

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
                worked_example_df = load_game_df(known_games[0].path) if known_games else None
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
            worked_example_df = load_game_df(known_games[0].path) if known_games else None
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
            # save_committed_tree is the ONLY durable persistence step now --
            # it pushes straight to the private VolleyData repo, no local
            # file involved. It's already merged into the in-memory tree
            # for this session even if the push below fails, so a failure
            # here is "not durably saved yet," not "lost."
            flash_level, flash_text = "success", f"Merged {n_changes} change(s) into committed tree."
            try:
                save_committed_tree(tree)
            except RuntimeError as e:
                flash_level, flash_text = "warning", (
                    f"Merged for this session, but couldn't save to VolleyData "
                    f"(edit won't survive a redeploy): {e}"
                )
            st.session_state.flash_message = (flash_level, flash_text)
            st.rerun()