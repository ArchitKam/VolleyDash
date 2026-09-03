"""
test_event_spec.py
===================
Validation of the event-grain metric kinds.

The load-bearing case is the conditional one: an evaluation code that is
perfectly legal for one skill must be REJECTED for another, because the
code's meaning depends on the skill it attaches to.
"""

import pytest

from fake_source import sample_source
from volley_event_spec import (
    COUNT_DISTINCT, make_event_spec, make_measure_spec, make_metric_formula_spec,
)

SCHEMA = sample_source().schema


# ── event specs ────────────────────────────────────────────────

def test_accepts_a_legal_skill_and_code_pair():
    spec = make_event_spec({"skill": "Attack", "evaluation_code": "#"}, "Kills", SCHEMA)
    assert spec.validate() == []


def test_accepts_a_skill_only_filter():
    """"every attack regardless of outcome" is a legitimate primitive."""
    assert make_event_spec({"skill": "Attack"}, "Attack attempts", SCHEMA).validate() == []


def test_accepts_a_list_of_codes_meaning_or():
    spec = make_event_spec(
        {"skill": "Attack", "evaluation_code": ["#", "+"]}, "Kills or positive", SCHEMA)
    assert spec.validate() == []


def test_rejects_a_code_that_is_valid_for_a_different_skill():
    """
    "+" occurs on Attack in this data but never on Serve. Keyed on the
    code alone this would pass; keyed on the (skill, code) pair it must
    fail, and the message must say which codes the skill really takes.
    """
    errors = make_event_spec(
        {"skill": "Serve", "evaluation_code": "+"}, "Bogus", SCHEMA).validate()
    assert errors
    assert "never occurs on a Serve" in errors[0]
    assert "#" in errors[0]  # names what Serve actually does take


def test_rejects_an_unknown_field():
    errors = make_event_spec({"not_a_field": "x"}, "Bogus", SCHEMA).validate()
    assert errors and "not a field in this source" in errors[0]


def test_rejects_a_value_the_field_never_takes():
    errors = make_event_spec({"skill": "Juggling"}, "Bogus", SCHEMA).validate()
    assert errors and "not a value skill takes" in errors[0]


def test_rejects_an_unknown_aggregate():
    errors = make_event_spec(
        {"skill": "Attack"}, "Bogus", SCHEMA, aggregate="frobnicate").validate()
    assert errors and "Unknown aggregate" in errors[0]


def test_rejects_an_empty_where():
    errors = make_event_spec({}, "Everything", SCHEMA).validate()
    assert errors and "non-empty 'where'" in errors[0]


def test_count_distinct_requires_a_known_field():
    assert make_event_spec({"skill": "Attack"}, "d", SCHEMA,
                            aggregate=COUNT_DISTINCT).validate()
    assert make_event_spec({"skill": "Attack"}, "d", SCHEMA,
                            aggregate=COUNT_DISTINCT, field="nope").validate()
    assert make_event_spec({"skill": "Attack"}, "d", SCHEMA,
                            aggregate=COUNT_DISTINCT, field="evaluation_code").validate() == []


# ── measure specs ──────────────────────────────────────────────

def test_measure_spec_accepts_sets_played_and_rejects_anything_else():
    assert make_measure_spec("sets_played", "d").validate() == []
    errors = make_measure_spec("sets_lost", "d").validate()
    assert errors and "Unknown measure" in errors[0]


# ── formula specs ──────────────────────────────────────────────

def test_formula_accepts_known_metric_tokens():
    spec = make_metric_formula_spec("[Kills] / [Sets Played]", "d", {"Kills", "Sets Played"})
    assert spec.validate() == []


def test_formula_rejects_an_unknown_metric_token():
    errors = make_metric_formula_spec("[Kills] / [Nope]", "d", {"Kills"}).validate()
    assert errors and "'Nope'" in errors[0]


def test_formula_rejects_an_expression_with_no_tokens():
    errors = make_metric_formula_spec("1 + 1", "d", {"Kills"}).validate()
    assert errors and "references no bracketed metrics" in errors[0]


def test_formula_does_not_accept_raw_csv_columns():
    """A .dvw has no "Attack K" column; only METRICS are valid tokens
    here, unlike the CSV world's formula validator."""
    errors = make_metric_formula_spec("[Attack K]", "d", {"Kills"}).validate()
    assert errors
