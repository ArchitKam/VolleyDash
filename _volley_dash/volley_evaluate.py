"""
volley_evaluate.py
===================
Evaluates a MetricSpec against a Source and returns the SAME tidy shape
the existing CSV path produces -- columns Game / Player / Value / Note,
one row per (Player, Game) -- so everything downstream
(recruiting_operations' Slice/Reduce/Rank/Compare, the tidy/consolidate
step, recruiting_encoding's charts) consumes DVW results without
knowing they came from a different file format.

VECTORIZED, NOT ROW-BY-ROW. The CSV evaluator in app.py loops
`for _, row in player_rows.iterrows()` and re-substitutes the formula
string per row, which is fine at ~15 rows per match but not at ~1,500
action rows per match times 22 matches. Here an event metric is ONE
boolean mask plus ONE groupby per metric per source, and a formula is
arithmetic over aligned Series.

THE (Player, Game) UNIVERSE, and why counts are 0 rather than missing.
A player who played a match but recorded no kill has ZERO kills, and a
player who did not dress for that match has NO kills -- a different
statement, which must not be plotted as a zero or averaged in as one.
The universe is therefore every (Player, Game) pair where the roster
metadata says the player played at least one set, unioned with pairs
that actually have actions; event counts are reindexed onto it and
filled with 0, and anyone outside it simply has no row.

Division by zero yields NaN, never inf -- an undefined ratio reads as
"no value", consistent with how blanks are treated in the CSV path.

Imports no Streamlit and no app.py, so it is usable from a plain script
and directly testable.
"""

import ast
import re
from typing import Dict, FrozenSet, List, Optional, Tuple

import pandas as pd

import _parent_path  # noqa: F401
from recruiting_tree import KnowledgeTree, MetricSpec, NodeKind

from volley_event_spec import COUNT, COUNT_DISTINCT
from volley_source import Source

TOKEN_PATTERN = re.compile(r"\[([^\[\]]+)\]")

PLAYER_AXIS = "Player"
GAME_AXIS = "Game"
OUTPUT_COLUMNS = [GAME_AXIS, PLAYER_AXIS, "Value", "Note"]

# Only arithmetic. Same closed node set as the CSV evaluator's
# safe_eval_arithmetic, for the same reason: a formula is authored by a
# user, so the expression language must not be able to reach anything
# but numbers.
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Name, ast.Load, ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.UAdd, ast.USub,
)


class MetricEvaluationError(Exception):
    """Raised for a spec that cannot be evaluated at all (unknown kind,
    circular reference, missing metric). Distinct from a metric that
    legitimately evaluates to no rows."""


def referenced_metrics(spec: MetricSpec) -> List[str]:
    """Metric labels a formula spec depends on; [] for event/measure
    specs, which are leaves of the dependency graph."""
    if spec.payload.get("kind") != "formula":
        return []
    return TOKEN_PATTERN.findall(spec.payload.get("formula_expr", ""))


def _identity_columns(source: Source) -> Tuple[str, str]:
    identity = source.identity_fields()
    return identity[PLAYER_AXIS], identity[GAME_AXIS]


def universe_index(source: Source) -> pd.MultiIndex:
    """
    Every (Player, Game) pair that should have a value at all.

    Roster participation first (so a player who was on court but never
    touched the ball still appears, with 0 rather than nothing), plus
    any pair that has actions, so a source with no measures() still
    works.
    """
    player_column, game_column = _identity_columns(source)
    pairs = set()

    measures = source.measures()
    if not measures.empty and "sets_played" in measures.columns:
        played = measures[measures["sets_played"] > 0]
        pairs.update(zip(played[player_column], played[game_column]))

    facts = source.facts()
    if not facts.empty:
        pairs.update(zip(facts[player_column], facts[game_column]))

    if not pairs:
        return pd.MultiIndex.from_tuples([], names=[PLAYER_AXIS, GAME_AXIS])
    return pd.MultiIndex.from_tuples(sorted(pairs), names=[PLAYER_AXIS, GAME_AXIS])


def _event_mask(facts: pd.DataFrame, where: Dict[str, object]) -> pd.Series:
    """AND across fields, OR within a field's value list. Vectorized:
    one boolean Series per field, combined with &."""
    mask = pd.Series(True, index=facts.index)
    for field, raw_value in where.items():
        if field not in facts.columns:
            # Validation already rejects this; treated as matching
            # nothing rather than silently ignoring the constraint,
            # which would over-count.
            return pd.Series(False, index=facts.index)
        values = raw_value if isinstance(raw_value, (list, tuple, set, frozenset)) else [raw_value]
        values = [str(v) for v in values]
        mask &= facts[field].astype(str).isin(values)
    return mask


def _evaluate_event(spec: MetricSpec, source: Source, index: pd.MultiIndex) -> pd.Series:
    facts = source.facts()
    player_column, game_column = _identity_columns(source)
    if facts.empty:
        return pd.Series(0.0, index=index)

    matching = facts[_event_mask(facts, spec.payload.get("where", {}))]
    aggregate = spec.payload.get("aggregate", COUNT)

    if matching.empty:
        counts = pd.Series(dtype="float64")
    elif aggregate == COUNT_DISTINCT:
        counts = matching.groupby([player_column, game_column])[spec.payload["field"]].nunique()
    else:
        counts = matching.groupby([player_column, game_column]).size()

    counts.index = counts.index.set_names([PLAYER_AXIS, GAME_AXIS])
    # 0, not NaN: everyone in the universe played, so "no matching
    # action" is a real zero.
    return counts.reindex(index).fillna(0).astype("float64")


def _evaluate_measure(spec: MetricSpec, source: Source, index: pd.MultiIndex) -> pd.Series:
    measures = source.measures()
    name = spec.payload.get("measure")
    if measures.empty or name not in measures.columns:
        return pd.Series(float("nan"), index=index)

    player_column, game_column = _identity_columns(source)
    series = measures.set_index([player_column, game_column])[name]
    series = series[~series.index.duplicated(keep="first")]
    series.index = series.index.set_names([PLAYER_AXIS, GAME_AXIS])
    return series.reindex(index).astype("float64")


def _evaluate_formula(spec: MetricSpec, source: Source, tree: KnowledgeTree,
                       index: pd.MultiIndex, resolving: FrozenSet[str]) -> pd.Series:
    expression = spec.payload.get("formula_expr", "")
    tokens = TOKEN_PATTERN.findall(expression)
    if not tokens:
        raise MetricEvaluationError(f"Formula '{expression}' references no bracketed metrics.")

    # Map each [Token] to a safe identifier so the expression can be
    # parsed as ordinary Python; metric labels contain spaces and
    # punctuation and are not valid identifiers themselves.
    substitutions: Dict[str, pd.Series] = {}
    rendered = expression
    for position, token in enumerate(dict.fromkeys(tokens)):
        name = f"_m{position}"
        substitutions[name] = _resolve_token(token, source, tree, index, resolving)
        rendered = rendered.replace(f"[{token}]", name)

    try:
        parsed = ast.parse(rendered, mode="eval")
    except SyntaxError as error:
        raise MetricEvaluationError(f"Couldn't parse formula '{expression}': {error}")

    for node in ast.walk(parsed):
        if not isinstance(node, _ALLOWED_NODES):
            raise MetricEvaluationError(
                f"Formula '{expression}' contains a disallowed construct "
                f"({type(node).__name__})."
            )
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            raise MetricEvaluationError(
                f"Formula '{expression}' contains a non-numeric literal ({node.value!r})."
            )

    return _eval_node(parsed.body, substitutions, index)


def _eval_node(node: ast.AST, values: Dict[str, pd.Series], index: pd.MultiIndex) -> pd.Series:
    """Walks the arithmetic tree with pandas Series as operands, so the
    whole formula is computed for every (Player, Game) at once."""
    if isinstance(node, ast.Constant):
        return pd.Series(float(node.value), index=index)

    if isinstance(node, ast.Name):
        return values[node.id]

    if isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand, values, index)
        return operand if isinstance(node.op, ast.UAdd) else -operand

    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left, values, index)
        right = _eval_node(node.right, values, index)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            # NaN rather than inf for a zero denominator: an undefined
            # ratio is "no value", not an enormous one. Matches the CSV
            # evaluator, which reports division by zero as an error
            # rather than a number.
            #
            # float("nan") rather than pd.NA specifically: pd.NA in a
            # float column forces object dtype and then refuses to cast
            # back ("float() argument must be ... not 'NAType'"). Real
            # trigger -- a libero has 0 attack attempts, so Hitting
            # Efficiency divides by zero for her in every match.
            denominator = right.astype("float64").replace(0.0, float("nan"))
            return left.astype("float64") / denominator
        raise MetricEvaluationError(f"Unsupported operator {type(node.op).__name__}.")

    raise MetricEvaluationError(f"Unsupported expression node {type(node).__name__}.")


def _find_leaf(tree: KnowledgeTree, label: str):
    return next(
        (n for n in tree.committed.values() if n.kind == NodeKind.LEAF and n.label == label),
        None,
    )


def _resolve_token(token: str, source: Source, tree: KnowledgeTree,
                    index: pd.MultiIndex, resolving: FrozenSet[str]) -> pd.Series:
    """
    One bracketed token -> a per-(Player, Game) Series.

    The cycle guard is carried forward from _resolve_token in app.py: a
    token already being resolved further up the stack means the metric
    graph has a loop, and without this the recursion would simply run
    until Python's stack limit.
    """
    if token in resolving:
        raise MetricEvaluationError(f"Circular metric reference through '{token}'.")

    leaf = _find_leaf(tree, token)
    if leaf is None or leaf.spec is None:
        raise MetricEvaluationError(f"'{token}' is not an existing metric.")

    return _evaluate_series(leaf.spec, source, tree, index, resolving | {token})


def _evaluate_series(spec: MetricSpec, source: Source, tree: KnowledgeTree,
                      index: pd.MultiIndex, resolving: FrozenSet[str]) -> pd.Series:
    kind = spec.payload.get("kind")
    if kind == "event":
        return _evaluate_event(spec, source, index)
    if kind == "measure":
        return _evaluate_measure(spec, source, index)
    if kind == "formula":
        return _evaluate_formula(spec, source, tree, index, resolving)
    raise MetricEvaluationError(f"Unknown spec kind '{kind}'.")


def evaluate_metric(spec: MetricSpec, source: Source, tree: KnowledgeTree,
                     player: Optional[str] = None) -> pd.DataFrame:
    """
    Evaluate one metric into the cube's tidy long form.

    Returns columns Game / Player / Value / Note -- byte-compatible with
    what run_metric_query produces for the CSV path, so a caller can
    hand the result straight to run_pipeline / tidy_data / the chart
    encoder.

    A spec that cannot be evaluated at all comes back as an EMPTY frame
    with the failure in Note on every row it would have had, rather than
    raising: one broken metric in a category should not take down the
    other eleven.
    """
    index = universe_index(source)
    if len(index) == 0:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    try:
        series = _evaluate_series(spec, source, tree, index, frozenset())
        note = ""
    except MetricEvaluationError as error:
        series = pd.Series(float("nan"), index=index)
        note = str(error)

    frame = series.rename("Value").reset_index()
    frame["Note"] = note
    frame = frame[OUTPUT_COLUMNS]
    if player is not None:
        frame = frame[frame[PLAYER_AXIS] == player]
    return frame.reset_index(drop=True)


def evaluate_category(tree: KnowledgeTree, branch_node_id: str, source: Source,
                       player: Optional[str] = None) -> pd.DataFrame:
    """
    Every metric under one branch, stacked with a Metric column -- the
    event-grain twin of run_category_query, and the reason a skill_group
    question returns one block per metric rather than one number.
    """
    columns = ["Metric"] + OUTPUT_COLUMNS
    branch = tree.committed.get(branch_node_id)
    if branch is None or branch.kind != NodeKind.BRANCH:
        return pd.DataFrame(columns=columns)

    frames = []
    for leaf_id in branch.children:
        leaf = tree.committed.get(leaf_id)
        if not leaf or leaf.kind != NodeKind.LEAF or not leaf.spec:
            continue
        leaf_frame = evaluate_metric(leaf.spec, source, tree, player=player)
        leaf_frame.insert(0, "Metric", leaf.label)
        frames.append(leaf_frame)

    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)
