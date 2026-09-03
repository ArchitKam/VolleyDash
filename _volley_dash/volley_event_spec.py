"""
volley_event_spec.py
=====================
Two more MetricSpec payload kinds -- "event" and "measure" -- plus a
formula kind whose tokens resolve against METRICS rather than CSV
columns. Built entirely from outside recruiting_tree.py: MetricSpec
already takes an arbitrary payload dict and a validator callable, so a
new kind needs no change to the tree engine at all, exactly as
make_column_spec / make_formula_spec do for the CSV world.

    {"kind": "event",   "where": {...}, "aggregate": "count"}
    {"kind": "measure", "measure": "sets_played"}
    {"kind": "formula", "formula_expr": "[Kills] / [Sets Played]", ...}

WHY "event" IS A FILTER PLUS AN AGGREGATE
At MEASURE grain a metric can name its value directly, because the
export already computed it ("Attack K" IS the kill count). At EVENT
grain nothing is precomputed: a kill is one row that happens to have
skill=Attack and evaluation_code=#, so the metric has to SAY that, and
counting is what turns rows into a number.

WHY THE VALIDATOR KEYS ON (skill, evaluation_code) AND NOT ON THE CODE
An evaluation code has no meaning on its own -- "#" is an ace on a
Serve, a perfect pass on a Reception and a kill on an Attack -- and
several skills never take some codes at all (in this corpus Set only
ever takes #, - and =). Checking the code against a flat six-value list
would happily accept "a Set evaluated /", which cannot occur, and would
silently return 0 forever. So the check asks the source's schema for
the codes legal FOR THAT SKILL and names them in the error when it
fails.

Validation returns a list of error strings, the same contract as the
existing validators in recruiting_tree.py, so a bad event spec is
rejected by tree.add_node() through exactly the same path a bad formula
already is.
"""

from typing import Callable, Dict, List, Optional, Sequence, Set

import _parent_path  # noqa: F401  -- puts the parent app on sys.path
from recruiting_tree import MetricSpec

from volley_source import SourceSchema

# Aggregates an event spec may ask for. A whitelist, so an unknown one
# is a validation error rather than an AttributeError at evaluation.
COUNT = "count"
COUNT_DISTINCT = "count_distinct"
VALID_AGGREGATES = (COUNT, COUNT_DISTINCT)

# Per-(Player, Game) quantities a "measure" spec may name. These come
# from Source.measures(), not from counting event rows -- see
# volley_source_dvw.py for why sets played cannot be counted from
# actions without being wrong for liberos.
SETS_PLAYED = "sets_played"
VALID_MEASURES = (SETS_PLAYED,)

_SKILL_FIELD = "skill"
_EVALUATION_FIELD = "evaluation_code"


def _as_value_list(value) -> List[str]:
    """A `where` value is either one value or a list of them (OR within
    the field), so both collapse to a list here and the rest of the code
    never branches on which was written."""
    if isinstance(value, (list, tuple, set, frozenset)):
        return [str(v) for v in value]
    return [str(value)]


def describe_where(where: Dict[str, object]) -> str:
    parts = []
    for field in sorted(where):
        values = _as_value_list(where[field])
        rendered = values[0] if len(values) == 1 else "{" + ", ".join(sorted(values)) + "}"
        parts.append(f"{field}={rendered}")
    return " and ".join(parts) if parts else "every action"


def _validate_where(where, schema: SourceSchema, errors: List[str]) -> None:
    if not isinstance(where, dict) or not where:
        errors.append(
            "An event metric needs a non-empty 'where' -- without one it counts "
            "every action by every player, which is not a metric."
        )
        return

    for field, raw_value in where.items():
        if not schema.has_field(field):
            errors.append(
                f"'{field}' is not a field in this source. Available: "
                f"{', '.join(schema.field_names())}"
            )
            continue

        allowed = schema.field_values(field)
        for value in _as_value_list(raw_value):
            if allowed is not None and value not in allowed:
                errors.append(
                    f"'{value}' is not a value {field} takes in the loaded matches. "
                    f"Seen: {', '.join(sorted(allowed))}"
                )

    # The conditional rule: a code is only legal for the skill it sits
    # with, so this can only be checked when both are pinned.
    if _SKILL_FIELD in where and _EVALUATION_FIELD in where:
        skills = _as_value_list(where[_SKILL_FIELD])
        codes = _as_value_list(where[_EVALUATION_FIELD])
        for skill in skills:
            legal = schema.dependent_field_values(_SKILL_FIELD, _EVALUATION_FIELD, skill)
            if legal is None:
                continue
            for code in codes:
                if code not in legal:
                    errors.append(
                        f"evaluation code '{code}' never occurs on a {skill} in the loaded "
                        f"matches -- codes seen for {skill}: {', '.join(sorted(legal))}. "
                        "(A code means a different thing per skill, so this pairing would "
                        "silently count zero forever.)"
                    )


def _make_event_validator(schema: SourceSchema) -> Callable[[Dict], List[str]]:
    def _validate(payload: Dict) -> List[str]:
        errors: List[str] = []
        _validate_where(payload.get("where"), schema, errors)

        aggregate = payload.get("aggregate", COUNT)
        if aggregate not in VALID_AGGREGATES:
            errors.append(
                f"Unknown aggregate '{aggregate}' -- expected one of {', '.join(VALID_AGGREGATES)}."
            )
        elif aggregate == COUNT_DISTINCT:
            field = payload.get("field")
            if not field:
                errors.append("aggregate 'count_distinct' needs a 'field' to count distinct values of.")
            elif not schema.has_field(field):
                errors.append(
                    f"count_distinct field '{field}' is not a field in this source. "
                    f"Available: {', '.join(schema.field_names())}"
                )
        return errors

    return _validate


def _make_measure_validator() -> Callable[[Dict], List[str]]:
    def _validate(payload: Dict) -> List[str]:
        measure = payload.get("measure")
        if measure not in VALID_MEASURES:
            return [
                f"Unknown measure '{measure}' -- expected one of {', '.join(VALID_MEASURES)}."
            ]
        return []

    return _validate


def _make_metric_formula_validator(valid_metric_labels: Set[str]) -> Callable[[Dict], List[str]]:
    """
    A formula at EVENT grain composes over other METRICS only -- never
    over raw event fields, which are categorical attributes of a single
    action and have no per-player value to do arithmetic on. So the
    token vocabulary here is the set of committed metric labels, not
    RECRUITING_COLUMN_SCHEMA (whose CSV column names mean nothing to a
    .dvw file).
    """
    from recruiting_tree import _COLUMN_REF_PATTERN  # same bracket syntax, one definition

    def _validate(payload: Dict) -> List[str]:
        errors: List[str] = []
        expression = payload.get("formula_expr", "")
        tokens = _COLUMN_REF_PATTERN.findall(expression)
        if not tokens:
            errors.append(
                f"Formula '{expression}' references no bracketed metrics -- "
                "expected at least one [Metric Name]."
            )
        for token in tokens:
            if token not in valid_metric_labels:
                known = ", ".join(sorted(valid_metric_labels)) or "(none yet)"
                errors.append(
                    f"'{token}' referenced in the formula is not an existing metric. Known: {known}"
                )
        return errors

    return _validate


def make_event_spec(where: Dict[str, object], description: str,
                     schema: SourceSchema, aggregate: str = COUNT,
                     field: Optional[str] = None) -> MetricSpec:
    """
    A metric defined as "count the actions matching `where`".

    `schema` is captured by the validator closure, mirroring how
    recruiting_tree._make_validator closes over extra_valid_refs -- the
    spec stays a plain dict, and what counts as valid travels with it.
    """
    payload: Dict[str, object] = {"kind": "event", "where": dict(where), "aggregate": aggregate}
    if field is not None:
        payload["field"] = field
    rendered = describe_where(where)
    label = "actions" if aggregate == COUNT else f"distinct {field}"
    return MetricSpec(
        description=f"Event: {label} where {rendered} -- {description}",
        payload=payload,
        validator=_make_event_validator(schema),
    )


def make_measure_spec(measure: str, description: str) -> MetricSpec:
    """A per-(Player, Game) quantity read from Source.measures() rather
    than counted from action rows."""
    return MetricSpec(
        description=f"Measure: {measure} -- {description}",
        payload={"kind": "measure", "measure": measure},
        validator=_make_measure_validator(),
    )


def make_metric_formula_spec(formula_expr: str, human_description: str,
                              valid_metric_labels: Sequence[str]) -> MetricSpec:
    """Arithmetic over other metrics, e.g. "[Kills] / [Sets Played]".
    Identical in shape to the CSV world's formula payload, so
    recruiting_operations and the persistence layer treat both the
    same -- only token resolution differs."""
    return MetricSpec(
        description=f"Formula: {formula_expr} -- {human_description}",
        payload={
            "kind": "formula",
            "formula_expr": formula_expr,
            "human_description": human_description,
        },
        validator=_make_metric_formula_validator(set(valid_metric_labels)),
    )
