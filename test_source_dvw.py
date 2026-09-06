"""
test_source_dvw.py
===================
Unit tests for the DVW adapter's pure logic (labels, identity, schema
derivation), plus integration tests over the real corpus that skip
gracefully when it is absent.

Each unit test below pins a decision that a real file forced -- the
comments name the specific case rather than describing the code.
"""

import glob
import os

import pandas as pd
import pytest

from source import FieldRole, Grain
from source_dvw import (
    DvwSource, MatchInfo, discover_matches, format_match_day, player_label, shorten_team_name,
)

from store_dvw import DEFAULT_DVW_DIR as DVW_DIR


# ── labels ─────────────────────────────────────────────────────

def test_player_label_includes_the_jersey_number():
    """A roster in this corpus carries the SAME name at two different
    numbers, so the name alone is not an identity. Names here are
    stand-ins -- the real roster is data and stays out of this repo."""
    assert player_label("44", "Sample Player") == "#44 Sample Player"
    assert player_label("17", "Sample Player") != player_label("44", "Sample Player")


def test_player_label_is_empty_for_a_row_with_no_player():
    """Every skill="Point" row (a rally-outcome marker belonging to a
    TEAM) has a null number and name. Formatting those naively produced
    the label "#nan nan", which behaved as a phantom roster member with
    3-5 "sets played" in all 22 files."""
    assert player_label(None, None) == ""
    assert player_label(float("nan"), float("nan")) == ""
    assert player_label("nan", "nan") == ""


def test_player_label_keeps_real_names_containing_nan():
    """Guards the fix above from over-reaching. Two surnames in this
    corpus contain the substring "nan", so a substring test for "nan"
    would erase real players; only the exact token may count."""
    assert player_label("7", "Robin Brennan") == "#7 Robin Brennan"
    assert player_label("8", "Sam Fernandez") == "#8 Sam Fernandez"


def test_shorten_team_name_handles_the_real_forms_in_this_corpus():
    assert shorten_team_name("University of Maryland") == "Maryland"
    assert shorten_team_name("Ohio State University") == "Ohio State"
    assert shorten_team_name("Indiana University, Bloomington") == "Indiana"
    assert shorten_team_name("Rutgers University") == "Rutgers"


def test_format_match_day_and_bad_input():
    assert format_match_day("11/08/2025") == "Nov 8"
    assert format_match_day("") == ""
    assert format_match_day("not-a-date") == ""
    assert format_match_day("13/40/2025") == ""  # month out of range, not guessed at


def test_match_label_includes_the_date_because_opponents_repeat():
    """Maryland plays Penn State, Indiana and Rutgers twice each here,
    so the opponent alone is not a key."""
    first = MatchInfo("a.dvw", "University of Maryland", "Pennsylvania State University",
                       "10/05/2025", "University of Maryland")
    second = MatchInfo("b.dvw", "Pennsylvania State University", "University of Maryland",
                        "11/23/2025", "University of Maryland")
    assert first.label != second.label
    assert first.label == "Pennsylvania State (Oct 5)"
    assert second.opponent == "Pennsylvania State University"


# ── the real corpus ────────────────────────────────────────────

real_files = sorted(glob.glob(os.path.join(DVW_DIR, "*.dvw")))
requires_corpus = pytest.mark.skipif(not real_files, reason="no .dvw corpus available here")


@pytest.fixture(scope="module")
def maryland_source():
    return DvwSource(real_files, team_of_interest="University of Maryland")


@requires_corpus
def test_discover_matches_is_not_recursive():
    """The corpus keeps duplicate copies of the same match in sibling
    folders; loading one twice would double every count."""
    found = discover_matches(DVW_DIR)
    assert found and all(f.endswith(".dvw") for f in found)
    assert len(found) == len(set(os.path.basename(f) for f in found))


@requires_corpus
def test_grain_and_identity(maryland_source):
    assert maryland_source.grain is Grain.EVENT
    assert maryland_source.identity_fields() == {
        "Player": "player_label", "Game": "match_label", "Set": "set_label",
    }


@requires_corpus
def test_dvw_offers_the_set_axis(maryland_source):
    """Event grain can see individual sets, so it advertises the axis;
    the CSV source cannot and does not (test_evaluate.py)."""
    assert maryland_source.axes() == ["Player", "Game", "Set", "Metric"]


@requires_corpus
def test_every_fact_carries_a_real_set_label(maryland_source):
    """get_plays() emits a trailing pseudo-set beyond the sets actually
    played; action rows must never carry it, or the UI would offer a
    set 4 in a 3-0 sweep."""
    labels = set(maryland_source.facts()["set_label"])
    assert labels and "" not in labels
    assert labels <= {"Set 1", "Set 2", "Set 3", "Set 4", "Set 5"}


@requires_corpus
def test_sets_played_is_reported_per_set_and_sums_to_the_match(maryland_source):
    """The per-set frame must be an exact refinement of the per-match
    answer that preceded it -- same totals, more detail -- or every
    existing per-set rate silently changes."""
    measures = maryland_source.measures()
    assert set(measures["sets_played"]) == {1}, "one row per set actually played"

    per_match = measures.groupby(["player_label", "match_label"])["sets_played"].sum()
    assert per_match.max() <= 5
    assert (per_match > 0).all()

    # A player's set labels within a match must be distinct: counting the
    # same set twice is the failure mode this replaces.
    counted = measures.groupby(["player_label", "match_label"])["set_label"].nunique()
    assert (counted == per_match).all()


@requires_corpus
def test_facts_contain_only_real_player_actions(maryland_source):
    facts = maryland_source.facts()
    assert not facts.empty
    assert facts["skill"].notna().all(), "non-action rows must be dropped"
    assert (facts["player_label"].str.strip() != "").all()
    assert "Point" not in set(facts["skill"]), "rally-outcome rows are not player actions"


@requires_corpus
def test_team_scoping_keeps_one_bench(maryland_source):
    assert set(maryland_source.facts()["team"]) == {"University of Maryland"}


@requires_corpus
def test_schema_vocabulary_is_derived_per_skill(maryland_source):
    vocabulary = maryland_source.schema.dependent_values[("skill", "evaluation_code")]
    # Measured across the corpus: Set only ever takes #, - and =.
    assert vocabulary["Set"] == frozenset({"#", "-", "="})
    # Attack never takes "!".
    assert "!" not in vocabulary["Attack"]
    assert "#" in vocabulary["Attack"]


@requires_corpus
def test_schema_marks_identities_and_dimensions(maryland_source):
    schema = maryland_source.schema
    assert schema.fields["player_label"].role is FieldRole.IDENTITY
    assert schema.fields["skill"].role is FieldRole.DIMENSION
    assert schema.field_values("skill") is not None


@requires_corpus
def test_sets_played_never_exceeds_the_sets_actually_contested(maryland_source):
    """A lineup entered for a set that was never played sits in the
    file; counting it reported 4 sets played in a 3-0 match."""
    measures = maryland_source.measures()
    assert not measures.empty
    assert measures["sets_played"].max() <= 5
    assert (measures["sets_played"] >= 0).all()


@requires_corpus
def test_sets_played_covers_every_player_who_recorded_an_action(maryland_source):
    """The libero records digs and receptions but never appears in the
    six-player rotation columns; a rotation-based count would score her
    0 and make every per-set metric a division by zero."""
    facts, measures = maryland_source.facts(), maryland_source.measures()
    played = set(measures[measures["sets_played"] > 0]["player_label"])
    acting = set(facts["player_label"])
    assert acting - played == set(), "a player with actions must have sets played"


@requires_corpus
def test_match_labels_are_unique(maryland_source):
    labels = [m.label for m in maryland_source.matches()]
    assert len(labels) == len(set(labels)), "opponent+date must identify a match"


# ── the Set axis, end to end ───────────────────────────────────

@requires_corpus
def test_per_set_split_sums_back_to_the_match_total(maryland_source):
    """Splitting by Set must be a REFINEMENT: the same kills, shown in
    more detail. If these disagree the Set axis is inventing or losing
    actions."""
    from evaluate import evaluate_metric
    from seed_dvw import seed_dvw_tree
    from recruiting_tree import NodeKind

    tree, _ = seed_dvw_tree(maryland_source.schema)
    kills = next(n for n in tree.committed.values()
                 if n.kind == NodeKind.LEAF and n.label == "Kills")

    per_match = evaluate_metric(kills.spec, maryland_source, tree)
    per_set = evaluate_metric(kills.spec, maryland_source, tree,
                              axes=["Player", "Game", "Set"])
    assert list(per_set.columns) == ["Game", "Player", "Set", "Value", "Note"]

    rolled = per_set.groupby(["Game", "Player"])["Value"].sum().sort_index()
    expected = per_match.set_index(["Game", "Player"])["Value"].sort_index()
    pd.testing.assert_series_equal(rolled, expected, check_names=False)


@requires_corpus
def test_scoping_to_one_set_rebases_the_per_set_denominator(maryland_source):
    """The decision recorded for a selected set: Sets Played becomes the
    sets played WITHIN the slice, so "Kills Per Set" in set 1 is her
    kills in set 1 rather than her kills in set 1 spread over the whole
    match."""
    from evaluate import evaluate_metric
    from seed_dvw import seed_dvw_tree
    from source import scope_source
    from recruiting_tree import NodeKind

    tree, _ = seed_dvw_tree(maryland_source.schema)
    leaves = {n.label: n for n in tree.committed.values() if n.kind == NodeKind.LEAF}
    scoped = scope_source(maryland_source, {"Set": ["Set 1"]})

    sets_played = evaluate_metric(leaves["Sets Played"].spec, scoped, tree)
    assert set(sets_played["Value"].dropna()) <= {0.0, 1.0}, "one set in scope is at most one set played"

    kills = evaluate_metric(leaves["Kills"].spec, scoped, tree)
    rate = evaluate_metric(leaves["Kills Per Set"].spec, scoped, tree)
    merged = kills.merge(rate, on=["Game", "Player"], suffixes=("_kills", "_rate"))
    merged = merged.merge(sets_played.rename(columns={"Value": "sets"}), on=["Game", "Player"])

    played = merged[merged["sets"] == 1]
    assert not played.empty
    pd.testing.assert_series_equal(
        played["Value_rate"], played["Value_kills"], check_names=False,
    )


@requires_corpus
def test_scoping_to_a_set_does_not_change_the_unscoped_answer(maryland_source):
    """Guards against the wrapper mutating the source it wraps -- the
    filtered view and the original are used side by side in one
    session."""
    from evaluate import evaluate_metric
    from seed_dvw import seed_dvw_tree
    from source import scope_source
    from recruiting_tree import NodeKind

    tree, _ = seed_dvw_tree(maryland_source.schema)
    kills = next(n for n in tree.committed.values()
                 if n.kind == NodeKind.LEAF and n.label == "Kills")

    before = evaluate_metric(kills.spec, maryland_source, tree)
    evaluate_metric(kills.spec, scope_source(maryland_source, {"Set": ["Set 2"]}), tree)
    after = evaluate_metric(kills.spec, maryland_source, tree)
    pd.testing.assert_frame_equal(before, after)
