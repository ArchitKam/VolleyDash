#omsairam omsairam omsairam 
"""
recruiting_operations.py
=========================
A CLOSED algebra of post-processing operations, replacing the ad hoc
GamePredicate/apply_aggregation pair.

WHY THIS IS CLOSED (and why that matters):
  Metrics are unbounded -- domain semantics have no ceiling, which is
  exactly why the knowledge tree must be user-extensible.
  Operations are bounded -- because the result of any query is always the
  same shape: a three-axis cube

      Player x Game x Metric -> Value

  Over a fixed-shape cube, the meaningful transformations are a small
  closed set (the same argument relational algebra makes for tables):

      Slice   -- restrict an axis (by identity, or by a value predicate)
      Reduce  -- collapse an axis with an aggregation
      Rank    -- order along an axis, optionally limited
      Compare -- express each cell relative to a reduction over an axis

  Consequence: the LLM emits a PIPELINE (an ordered list of these), not a
  choice among named functions. A question shape nobody anticipated
  becomes a new composition, not new Python. That is the property the
  ad hoc version lacked.

WHAT THIS DOES NOT CLAIM:
  Closure is a claim about the operation space over THIS cube shape, not
  a claim that every conceivable coach question is expressible. Two known
  gaps, listed rather than hidden:
    - Chronological/trend questions ("is she improving?") need Game to
      carry an order. The cube treats Game as unordered/categorical, so
      Rank(axis="Game") sorts by VALUE, not by date. Adding trend support
      means giving Game an ordinal, which is a schema change, not a new
      operation.
    - Set-level or rally-level questions are outside this data entirely
      (the recruiting CSV export is already aggregated per player per
      match), so no operation here can recover them.
  Both belong in the paper as scope limits, and the second is a property
  of the export, not of this design.

VALIDATION IS EMPIRICAL, NOT ASSERTED:
  The right way to defend closure is to take real coach questions, map
  each to a pipeline, and report coverage -- questions expressible / total.
  An unmappable question is then evidence about the algebra, not a bug to
  paper over. That coverage number is a reportable result.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Union

import pandas as pd

# Cube axes, ordered coarse to fine. Set sits between Game and Metric
# because it SUBDIVIDES a game: _remaining_axes preserves this order when
# it decides what a Rank groups within, so ranking inside a set groups by
# (Player, Game) rather than some arbitrary column order.
#
# Not every source offers every axis -- Set exists only where the data
# can see individual sets (source.axes()). An operation naming an axis
# the current result has no column for is refused by name at execution
# time rather than silently ignored, so a Set operation against CSV data
# reports that it cannot be done instead of quietly returning match
# totals.
AXES = ("Player", "Game", "Set", "Metric")

_COMPARISONS = {
    ">":  lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "<":  lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}

# Aggregations available to Reduce and Compare. Deliberately a whitelist:
# adding "median" or a percentile is a ONE-LINE addition here and requires
# no change to any operation, pipeline, or caller -- which is the point.
# Justin never authors an operation; the algebra's parameter space covers
# the variation instead.
_AGGS: Dict[str, str] = {
    "mean": "mean", "sum": "sum", "min": "min",
    "max": "max", "count": "count", "median": "median",
}

VALID_COMPARISONS = sorted(_COMPARISONS)
VALID_AGGS = sorted(_AGGS)


@dataclass(frozen=True)
class ValuePredicate:
    """
    "...where <metric> <op> <threshold>", optionally decided by ONE named
    player rather than each player's own value.

    `metric` must be present in the frame being operated on -- the caller
    fetches every metric the pipeline references into one long frame
    first. That's why no separate predicate DataFrame is needed here:
    Metric is an axis of the cube, so a predicate on another metric is
    just a lookup along that axis.
    """
    metric: str
    op: str
    threshold: float
    decided_by_player: Optional[str] = None

    def describe(self) -> str:
        who = f"{self.decided_by_player}'s " if self.decided_by_player else "each player's own "
        return f"{who}{self.metric} {self.op} {self.threshold:g}"


@dataclass(frozen=True)
class Slice:
    """Restrict `axis` -- either to an explicit set of values (`keep`), or
    to those whose rows satisfy `predicate`."""
    axis: Literal["Player", "Game", "Set", "Metric"]
    keep: Optional[List[str]] = None
    predicate: Optional[ValuePredicate] = None

    def describe(self) -> str:
        if self.keep is not None:
            return f"{self.axis} in {self.keep}"
        if self.predicate is not None:
            return f"{self.axis} where {self.predicate.describe()}"
        return f"{self.axis} (no-op)"


@dataclass(frozen=True)
class Reduce:
    """Collapse `axis`, aggregating Value with `how`. Always emits
    'N Used' so a partial aggregation can't masquerade as a complete
    one (a 2-of-5-game mean must never look like a 5-of-5 mean)."""
    axis: Literal["Player", "Game", "Set", "Metric"]
    how: str = "mean"

    def describe(self) -> str:
        return f"{self.how} over {self.axis}"


@dataclass(frozen=True)
class Rank:
    """Order along `axis` by Value, optionally keeping the top/bottom N.
    NOTE: ranks by VALUE, not chronology -- see the trend caveat in the
    module docstring."""
    axis: Literal["Player", "Game", "Set", "Metric"]
    descending: bool = True
    limit: Optional[int] = None

    def describe(self) -> str:
        direction = "highest" if self.descending else "lowest"
        shown = max(3, self.limit) if self.limit is not None else 3
        return f"{direction} {shown} by {self.axis}"


@dataclass(frozen=True)
class Compare:
    """
    Express each Value relative to a baseline computed by reducing over
    `axis`. mode="difference" -> value - baseline; "ratio" -> value /
    baseline. This is what answers "is she above team average" (axis=
    Player) and "is this her best game" (axis=Game).
    """
    axis: Literal["Player", "Game", "Set", "Metric"]
    how: str = "mean"
    mode: Literal["difference", "ratio"] = "difference"

    def describe(self) -> str:
        return f"vs {self.how} across {self.axis} ({self.mode})"


Operation = Union[Slice, Reduce, Rank, Compare]


# ──────────────────────────────────────────────────────────────
# EXECUTION
# ──────────────────────────────────────────────────────────────

def _remaining_axes(df: pd.DataFrame, exclude: str) -> List[str]:
    return [a for a in AXES if a in df.columns and a != exclude]


def _apply_slice(df: pd.DataFrame, op: Slice) -> tuple:
    if op.axis not in df.columns:
        return df, f"Can't slice on '{op.axis}' -- not a column in this result."

    if op.keep is not None:
        return df[df[op.axis].isin(op.keep)].reset_index(drop=True), None

    if op.predicate is None:
        return df, None

    pred = op.predicate
    compare = _COMPARISONS.get(pred.op)
    if compare is None:
        return df.iloc[0:0], f"Unknown comparison '{pred.op}' (expected {', '.join(VALID_COMPARISONS)})."
    if "Metric" not in df.columns:
        return df, "Predicate needs a Metric column to look up the deciding metric."

    basis = df[df["Metric"] == pred.metric].copy()
    if basis.empty:
        return df.iloc[0:0], (
            f"Predicate metric '{pred.metric}' isn't in this result -- "
            "fetch it alongside the displayed metric before applying the pipeline."
        )
    basis["Value"] = pd.to_numeric(basis["Value"], errors="coerce")

    if pred.decided_by_player is not None:
        basis = basis[basis["Player"] == pred.decided_by_player]

    uncomputable = sorted(set(basis[basis["Value"].isna()][op.axis]))
    basis = basis[basis["Value"].notna()]
    passing = basis[basis["Value"].apply(lambda v: compare(v, pred.threshold))]

    if pred.decided_by_player is not None:
        # One player gates the axis for everyone.
        kept = set(passing[op.axis])
        out = df[df[op.axis].isin(kept)]
    else:
        # Per-player gating: qualification is a (Player, axis) pair, so an
        # axis value can count for one player and not another.
        pairs = set(zip(passing["Player"], passing[op.axis]))
        out = df[df.apply(lambda r: (r["Player"], r[op.axis]) in pairs, axis=1)]

    note = None
    if uncomputable:
        note = (f"Excluded {len(uncomputable)} {op.axis.lower()}(s) with no computable "
                f"'{pred.metric}' value: {', '.join(map(str, uncomputable))}.")
    return out.reset_index(drop=True), note


def _apply_reduce(df: pd.DataFrame, op: Reduce) -> tuple:
    how = _AGGS.get(op.how)
    if how is None:
        return df, f"Unknown aggregation '{op.how}' (expected {', '.join(VALID_AGGS)})."
    if op.axis not in df.columns:
        return df, f"Can't reduce over '{op.axis}' -- not a column in this result."

    keep = _remaining_axes(df, op.axis)
    if not keep:
        return df, f"Refusing to reduce over '{op.axis}' -- nothing would remain to group by."

    work = df.copy()
    work["Value"] = pd.to_numeric(work["Value"], errors="coerce")
    out = (work.groupby(keep, dropna=False)
                .agg(Value=("Value", how), **{"N Used": ("Value", "count")})
                .reset_index())
    out[op.axis] = f"{op.how} of {op.axis.lower()}s"
    return out, None


def _apply_rank(df: pd.DataFrame, op: Rank) -> tuple:
    """
    Ranks WITHIN each remaining-axis group, not across the whole frame --
    e.g. Rank(axis="Player") on a category result (every metric in a
    skill group at once, or several metrics a pipeline pulled in
    together) ranks each Metric's (and Game's) players independently.
    Without this, a "top server" question would sort Serve SA rows
    against Serve Rtg. rows in one global pile and keep only a handful
    of rows total across every metric -- exactly the "everything
    condenses to one number" symptom. Grouping by the same remaining
    axes a multi-metric result already carries makes ranking a category
    behave identically to ranking N separate single-metric actions.

    Always keeps at least 3 rows per group (more if op.limit asks for
    more) -- "who's the highest" reads better with a little context than
    as one bare number, and a short list costs nothing extra to show. A
    group with fewer than 3 candidates just shows all of it.
    """
    if op.axis not in df.columns:
        return df, f"Can't rank on '{op.axis}' -- not a column in this result."
    work = df.copy()
    work["Value"] = pd.to_numeric(work["Value"], errors="coerce")
    effective_limit = max(3, op.limit) if op.limit is not None else 3

    group_cols = _remaining_axes(work, op.axis)
    # na_position="last" so missing data never wins a "highest" ranking.
    # A global sort by Value first means every per-group subset pulled out
    # afterward is already correctly ordered by Value too, regardless of
    # sort stability -- filtering rows out of a Value-sorted sequence
    # can't un-sort them.
    work = work.sort_values("Value", ascending=not op.descending, na_position="last")

    if not group_cols:
        return work.head(effective_limit).reset_index(drop=True), None

    ranked = work.groupby(group_cols, dropna=False, sort=False, group_keys=False).head(effective_limit)
    # Re-sort so same-group rows sit together in the displayed/plotted
    # order (grouped-then-ranked reads far better than the interleaved
    # order a pure global sort would leave them in).
    ranked = ranked.sort_values(
        group_cols + ["Value"], ascending=[True] * len(group_cols) + [not op.descending], na_position="last",
    )
    return ranked.reset_index(drop=True), None


def _apply_compare(df: pd.DataFrame, op: Compare) -> tuple:
    how = _AGGS.get(op.how)
    if how is None:
        return df, f"Unknown aggregation '{op.how}' (expected {', '.join(VALID_AGGS)})."
    if op.axis not in df.columns:
        return df, f"Can't compare across '{op.axis}' -- not a column in this result."

    keep = _remaining_axes(df, op.axis)
    if not keep:
        return df, f"Refusing to compare across '{op.axis}' -- no axis left to hold fixed."

    work = df.copy()
    work["Value"] = pd.to_numeric(work["Value"], errors="coerce")
    baseline = work.groupby(keep, dropna=False)["Value"].transform(how)

    if op.mode == "difference":
        work["Value"] = work["Value"] - baseline
    elif op.mode == "ratio":
        # Guard: a zero baseline yields NaN, not inf -- an undefined
        # comparison should read as "no value", consistent with how blanks
        # are treated everywhere else.
        work["Value"] = work["Value"] / baseline.replace(0, pd.NA)
    else:
        return df, f"Unknown compare mode '{op.mode}' (expected 'difference' or 'ratio')."

    work["Baseline"] = baseline
    return work.reset_index(drop=True), None


_DISPATCH = {Slice: _apply_slice, Reduce: _apply_reduce, Rank: _apply_rank, Compare: _apply_compare}


def run_pipeline(df: pd.DataFrame, pipeline: List[Operation]) -> tuple:
    """
    Applies operations in the given order and returns (df, notes).

    ORDER IS THE CALLER'S (i.e. the LLM's) RESPONSIBILITY and it matters:
    slice-then-reduce averages only the games that qualified; reduce-then-
    slice averages everything and then filters the averages. Both are
    legitimate questions, so this does not silently reorder -- it executes
    what it was given, which keeps the emitted pipeline an honest record
    of what was actually computed.

    An empty pipeline returns the frame unchanged, so this layer is
    strictly additive over existing behavior.
    """
    notes: List[str] = []
    out = df
    for op in pipeline:
        handler = _DISPATCH.get(type(op))
        if handler is None:
            notes.append(f"Unknown operation {type(op).__name__} -- skipped.")
            continue
        out, note = handler(out, op)
        if note:
            notes.append(note)
        if out.empty:
            notes.append(f"No rows remained after: {op.describe()}.")
            break
    return out, notes


def describe_pipeline(pipeline: List[Operation]) -> str:
    """Plain-language rendering of a pipeline, for the routing-inspection
    panel -- so a coach can read what the system decided to do, in the
    same spirit as showing a formula alongside a worked example."""
    return " -> ".join(op.describe() for op in pipeline) if pipeline else "(no operations)"


# ──────────────────────────────────────────────────────────────
# JSON SERIALIZATION -- for round-tripping a pipeline over the LLM API.
# A flat, "op"-discriminated dict per operation, with fields matching each
# dataclass's own field names 1:1 (see recruiting_llm.py's router prompt
# for the exact shape shown to the model). Kept here rather than
# hand-rolled in app.py so the ONLY place that knows how an Operation
# maps to/from JSON is this module -- the same "one fixed
# executor/serializer" principle the rest of this codebase already
# follows for MetricSpec.
#
# from_dict() never silently coerces an invalid axis/aggregation/
# comparison/op-kind -- it raises ValueError. A wrong axis name here
# would silently change what got computed rather than surface as
# something the caller (recruiting_llm._validate_and_repair) can drop
# and log, consistent with this codebase's "mechanically validate,
# never guess" rule for anything the LLM proposes.
# ──────────────────────────────────────────────────────────────

def operation_to_dict(op: Operation) -> dict:
    if isinstance(op, Slice):
        return {
            "op": "slice",
            "axis": op.axis,
            "keep": op.keep,
            "predicate": (
                {
                    "metric": op.predicate.metric,
                    "op": op.predicate.op,
                    "threshold": op.predicate.threshold,
                    "decided_by_player": op.predicate.decided_by_player,
                }
                if op.predicate is not None else None
            ),
        }
    if isinstance(op, Reduce):
        return {"op": "reduce", "axis": op.axis, "how": op.how}
    if isinstance(op, Rank):
        return {"op": "rank", "axis": op.axis, "descending": op.descending, "limit": op.limit}
    if isinstance(op, Compare):
        return {"op": "compare", "axis": op.axis, "how": op.how, "mode": op.mode}
    raise ValueError(f"Unknown operation type: {type(op).__name__}")


def operation_from_dict(d: Dict) -> Operation:
    if not isinstance(d, dict):
        raise ValueError(f"Expected an operation object, got {type(d).__name__}.")

    kind = d.get("op")
    axis = d.get("axis")
    if axis not in AXES:
        raise ValueError(f"Invalid axis {axis!r} (expected one of {AXES}).")

    if kind == "slice":
        keep = d.get("keep")
        if keep is not None and not isinstance(keep, list):
            raise ValueError(f"Slice.keep must be a list or null, got {type(keep).__name__}.")
        predicate = None
        raw_pred = d.get("predicate")
        if raw_pred:
            if not isinstance(raw_pred, dict):
                raise ValueError(f"Slice.predicate must be an object or null, got {type(raw_pred).__name__}.")
            pred_op = raw_pred.get("op")
            if pred_op not in _COMPARISONS:
                raise ValueError(f"Invalid comparison {pred_op!r} (expected one of {VALID_COMPARISONS}).")
            metric = raw_pred.get("metric")
            if not isinstance(metric, str) or not metric.strip():
                raise ValueError("Slice.predicate.metric must be a non-empty string.")
            try:
                threshold = float(raw_pred.get("threshold"))
            except (TypeError, ValueError):
                raise ValueError(f"Slice.predicate.threshold must be numeric, got {raw_pred.get('threshold')!r}.")
            predicate = ValuePredicate(
                metric=metric, op=pred_op, threshold=threshold,
                decided_by_player=raw_pred.get("decided_by_player") or None,
            )
        return Slice(axis=axis, keep=keep, predicate=predicate)

    if kind == "reduce":
        how = d.get("how", "mean")
        if how not in _AGGS:
            raise ValueError(f"Invalid aggregation {how!r} (expected one of {VALID_AGGS}).")
        return Reduce(axis=axis, how=how)

    if kind == "rank":
        limit = d.get("limit")
        if limit is not None:
            try:
                limit = int(limit)
            except (TypeError, ValueError):
                raise ValueError(f"Rank.limit must be an integer or null, got {limit!r}.")
        return Rank(axis=axis, descending=bool(d.get("descending", True)), limit=limit)

    if kind == "compare":
        how = d.get("how", "mean")
        if how not in _AGGS:
            raise ValueError(f"Invalid aggregation {how!r} (expected one of {VALID_AGGS}).")
        mode = d.get("mode", "difference")
        if mode not in ("difference", "ratio"):
            raise ValueError(f"Invalid compare mode {mode!r} (expected 'difference' or 'ratio').")
        return Compare(axis=axis, how=how, mode=mode)

    raise ValueError(f"Unknown operation kind {kind!r} (expected one of slice/reduce/rank/compare).")


def pipeline_to_dicts(pipeline: List[Operation]) -> List[dict]:
    return [operation_to_dict(op) for op in pipeline]


def pipeline_from_dicts(dicts: List[Dict]) -> List[Operation]:
    """Raises ValueError (naming which entry and why) on the first invalid
    operation -- callers (recruiting_llm._validate_and_repair) catch this
    and drop the whole pipeline for that action rather than run a partially
    trusted one, same all-or-nothing mechanical validation used for a
    MetricSpec's own formula."""
    return [operation_from_dict(entry) for entry in dicts]
