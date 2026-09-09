"""
evaluate.py
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
from typing import Dict, FrozenSet, List, Optional, Sequence

import pandas as pd

from recruiting_tree import KnowledgeTree, MetricSpec, NodeKind

from event_spec import COUNT, COUNT_DISTINCT
from evaluate_csv import run_category_query, run_metric_query
from source import GAME_AXIS, PLAYER_AXIS, SET_AXIS, Grain, Source

TOKEN_PATTERN = re.compile(r"\[([^\[\]]+)\]")

#: The grain every evaluation uses unless a caller asks for more. Set is
#: opt-in, so an unfiltered question stays a per-match summary exactly as
#: it was before the Set axis existed.
DEFAULT_AXES = (PLAYER_AXIS, GAME_AXIS)

OUTPUT_COLUMNS = [GAME_AXIS, PLAYER_AXIS, "Value", "Note"]


def output_columns(axes: List[str]) -> List[str]:
    """Tidy column order for a given grain: Game, Player, [Set], Value,
    Note. Set sits after the identities it subdivides so a table reads
    coarse-to-fine left to right."""
    ordered = [axis for axis in (GAME_AXIS, PLAYER_AXIS, SET_AXIS) if axis in axes]
    return ordered + ["Value", "Note"]

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


def resolve_axes(source: Source, axes: Optional[Sequence[str]] = None) -> List[str]:
    """
    The identity axes an evaluation will actually group by.

    Filtered against what the source offers, so asking for Set on a
    source that has no Set column is a no-op rather than a KeyError --
    the CSV source genuinely cannot answer per-set questions, and a
    caller that forgets to check should get a match-level answer, not a
    crash.
    """
    available = source.identity_fields()
    requested = list(axes) if axes else list(DEFAULT_AXES)
    return [axis for axis in requested if axis in available]


def _identity_columns(source: Source, axes: Sequence[str]) -> List[str]:
    identity = source.identity_fields()
    return [identity[axis] for axis in axes]


def universe_index(source: Source, axes: Optional[Sequence[str]] = None) -> pd.MultiIndex:
    """
    Every identity tuple that should have a value at all, at the
    requested grain.

    Roster participation first (so a player who was on court but never
    touched the ball still appears, with 0 rather than nothing), plus
    any tuple that has actions, so a source with no measures() still
    works.
    """
    axes = resolve_axes(source, axes)
    columns = _identity_columns(source, axes)
    tuples = set()

    measures = source.measures()
    if not measures.empty and "sets_played" in measures.columns:
        played = measures[measures["sets_played"] > 0]
        if all(column in played.columns for column in columns):
            tuples.update(zip(*(played[column] for column in columns)))

    facts = source.facts()
    if not facts.empty and all(column in facts.columns for column in columns):
        tuples.update(zip(*(facts[column] for column in columns)))

    if not tuples:
        return pd.MultiIndex.from_tuples([], names=list(axes))
    return pd.MultiIndex.from_tuples(sorted(tuples), names=list(axes))


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
    columns = _identity_columns(source, list(index.names))
    if facts.empty:
        return pd.Series(0.0, index=index)

    matching = facts[_event_mask(facts, spec.payload.get("where", {}))]
    aggregate = spec.payload.get("aggregate", COUNT)

    if matching.empty:
        counts = pd.Series(dtype="float64")
    elif aggregate == COUNT_DISTINCT:
        counts = matching.groupby(columns)[spec.payload["field"]].nunique()
    else:
        counts = matching.groupby(columns).size()

    counts.index = counts.index.set_names(list(index.names))
    # 0, not NaN: everyone in the universe played, so "no matching
    # action" is a real zero.
    return counts.reindex(index).fillna(0).astype("float64")


def _evaluate_measure(spec: MetricSpec, source: Source, index: pd.MultiIndex) -> pd.Series:
    """
    A measure is reported by the source at ITS finest grain, which may
    be finer than the cube being built -- sets played is per set, but a
    question with no set filter wants it per match. Combining is the
    source's call (measure_aggregation), not an assumption here: summing
    a percentage would be silently wrong.

    Grouping rather than set_index also removes the old need to drop
    duplicate index entries, which at a coarser grain were not
    duplicates at all but the very rows that need adding up.
    """
    measures = source.measures()
    name = spec.payload.get("measure")
    if measures.empty or name not in measures.columns:
        return pd.Series(float("nan"), index=index)

    columns = _identity_columns(source, list(index.names))
    if not all(column in measures.columns for column in columns):
        return pd.Series(float("nan"), index=index)

    how = source.measure_aggregation(name)
    series = measures.groupby(columns)[name].agg(how)
    series.index = series.index.set_names(list(index.names))
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
                     player: Optional[str] = None,
                     axes: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """
    One metric against any source, as the cube's tidy long form.

    Dispatches on GRAIN, because the two grains disagree about what a
    blank value and a division by zero mean and the CSV answers are what
    a coach already reads in the recruiting app (evaluate_csv.py's
    docstring). The shared boundary is this function's OUTPUT -- columns
    Game / Player / [Set] / Value / Note -- not the arithmetic behind it.
    """
    if source.grain is Grain.MEASURE:
        return _evaluate_metric_measure(spec, source, tree, player=player)
    return _evaluate_metric_event(spec, source, tree, player=player, axes=axes)


def _evaluate_metric_measure(spec: MetricSpec, source: Source, tree: KnowledgeTree,
                              player: Optional[str] = None) -> pd.DataFrame:
    """
    MEASURE grain: one row per player per match already, so the spec is
    evaluated against each row rather than aggregated onto an index.

    This is run_metric_query from app.py, moved rather than rewritten --
    same functions, same order, same Note text -- so the recruiting
    numbers and the recruiting error messages are the ones that were
    there before unification.

    `axes` is not accepted: there is no finer grain to ask for. A caller
    that requests Set gets match totals via resolve_axes, which is the
    same answer this would give and one fewer way to be surprised.
    """
    frames = getattr(source, "frames", None)
    if frames is None:
        raise MetricEvaluationError("A measure-grain source must expose frames().")

    frame = run_metric_query(spec, tree, frames(), player_name=player)
    return frame[OUTPUT_COLUMNS].reset_index(drop=True)


def _evaluate_metric_event(spec: MetricSpec, source: Source, tree: KnowledgeTree,
                            player: Optional[str] = None,
                            axes: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """
    Evaluate one metric into the cube's tidy long form.

    Returns columns Game / Player / [Set] / Value / Note -- the Set
    column only when `axes` asks for it, so the default output stays
    byte-compatible with what the CSV path produces and a caller can
    hand the result straight to run_pipeline / tidy_data / the chart
    encoder.

    To scope an evaluation to particular sets WITHOUT splitting by set,
    narrow the source (source.scope_source) and leave `axes` alone: the
    denominators narrow with it, which is what makes a per-set rate
    correct inside a single set.

    A spec that cannot be evaluated at all comes back as a frame with
    the failure in Note on every row it would have had, rather than
    raising: one broken metric in a category should not take down the
    other eleven.
    """
    resolved = resolve_axes(source, axes)
    columns = output_columns(resolved)

    index = universe_index(source, resolved)
    if len(index) == 0:
        return pd.DataFrame(columns=columns)

    try:
        series = _evaluate_series(spec, source, tree, index, frozenset())
        note = ""
    except MetricEvaluationError as error:
        series = pd.Series(float("nan"), index=index)
        note = str(error)

    frame = series.rename("Value").reset_index()
    frame["Note"] = note
    frame = frame[columns]
    if player is not None:
        frame = frame[frame[PLAYER_AXIS] == player]
    return frame.reset_index(drop=True)


def evaluate_category(tree: KnowledgeTree, branch_node_id: str, source: Source,
                       player: Optional[str] = None,
                       axes: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """
    Every metric under one branch, stacked with a Metric column -- the
    event-grain twin of run_category_query, and the reason a skill_group
    question returns one block per metric rather than one number.
    """
    columns = ["Metric"] + (
        OUTPUT_COLUMNS if source.grain is Grain.MEASURE
        else output_columns(resolve_axes(source, axes))
    )
    branch = tree.committed.get(branch_node_id)
    if branch is None or branch.kind != NodeKind.BRANCH:
        return pd.DataFrame(columns=columns)

    if source.grain is Grain.MEASURE:
        measure_frames = getattr(source, "frames", None)
        if measure_frames is None:
            raise MetricEvaluationError("A measure-grain source must expose frames().")
        return run_category_query(tree, branch_node_id, measure_frames(), player_name=player)

    frames = []
    for leaf_id in branch.children:
        leaf = tree.committed.get(leaf_id)
        if not leaf or leaf.kind != NodeKind.LEAF or not leaf.spec:
            continue
        leaf_frame = evaluate_metric(leaf.spec, source, tree, player=player, axes=axes)
        if leaf_frame.empty and not list(leaf_frame.columns):
            continue
        leaf_frame.insert(0, "Metric", leaf.label)
        frames.append(leaf_frame)

    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)
