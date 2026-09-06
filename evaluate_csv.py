"""
evaluate_csv.py
================
The MEASURE-grain evaluator: a MetricSpec against one Huddle CSV.

Moved VERBATIM out of app.py during unification -- not rewritten, not
merged with the event-grain evaluator -- because the two disagree on
questions that only have answers at one grain:

  * A blank cell is a real thing in a CSV export and means "not
    recorded", so it is an ERROR here rather than a zero. There is no
    such thing as a blank action row.
  * Division by zero is an error here and NaN there, and the CSV app's
    error text is what the UI shows a coach.
  * A [Token] resolves COLUMN-first and metric-label-second, because a
    Huddle formula is authored against export columns. At event grain
    there are no columns to reference, only metrics.

Folding these into one evaluator would have meant picking one set of
answers and silently changing recruiting numbers in exactly the cases
nobody writes a test for. The shared boundary is the CUBE both
evaluators produce (evaluate.py dispatches on grain), not the arithmetic
that gets there.
"""

import ast
import math
import re
from typing import TYPE_CHECKING
from dataclasses import dataclass, field
from typing import Any, FrozenSet, List, Optional, Tuple

if TYPE_CHECKING:  # pragma: no cover
    from recruiting_data_store import GameInfo

import pandas as pd

from recruiting_tree import KnowledgeTree, MetricSpec, NodeKind

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


# ──────────────────────────────────────────────────────────────
# CUBE CONSTRUCTION
# ──────────────────────────────────────────────────────────────
# The one measure-grain implementation. evaluate._evaluate_metric_measure
# calls straight into this rather than restating it, so "the unified path
# agrees with the old path" is structural instead of a coincidence two
# copies have to keep re-earning.

def game_label(game: object) -> str:
    """
    The Game axis value for one match.

    GameInfo.opponent is what the CSV path has always used; a plain
    string is accepted so a caller (and a test) need not build a
    GameInfo. Defined once and imported by source_csv, because two
    copies of "what is this game called" that drift produce a cube whose
    rows silently fail to line up.
    """
    return str(getattr(game, "opponent", game))


def run_metric_query(
    spec: MetricSpec, tree: KnowledgeTree,
    game_dfs: List[Tuple["GameInfo", pd.DataFrame]],
    player_name: Optional[str] = None, name_col: str = "Name",
) -> pd.DataFrame:
    records = []
    for game, df in game_dfs:
        for name, result in compute_for_all_players(spec, df, tree=tree, name_col=name_col):
            if player_name is not None and name != player_name:
                continue
            records.append({
                "Game": game_label(game), "Player": name,
                "Value": result.value, "Note": result.error or "",
            })
    return pd.DataFrame(records, columns=["Game", "Player", "Value", "Note"])


def run_category_query(
    tree: KnowledgeTree, branch_node_id: str,
    game_dfs: List[Tuple["GameInfo", pd.DataFrame]],
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
