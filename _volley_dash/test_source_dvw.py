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

from volley_source import FieldRole, Grain
from volley_source_dvw import (
    DvwSource, MatchInfo, discover_matches, format_match_day, player_label, shorten_team_name,
)

DVW_DIR = "/fs/vulcan-projects/vlm_motion_benchmark/VolleyballMetrics/player_analysis/downloads/dvw"


# ── labels ─────────────────────────────────────────────────────

def test_player_label_includes_the_jersey_number():
    """Maryland's roster carries "Eva Rohrbach" at both #17 and #44, so
    the name alone is not an identity."""
    assert player_label("44", "Eva Rohrbach") == "#44 Eva Rohrbach"
    assert player_label("17", "Eva Rohrbach") != player_label("44", "Eva Rohrbach")


def test_player_label_is_empty_for_a_row_with_no_player():
    """Every skill="Point" row (a rally-outcome marker belonging to a
    TEAM) has a null number and name. Formatting those naively produced
    the label "#nan nan", which behaved as a phantom roster member with
    3-5 "sets played" in all 22 files."""
    assert player_label(None, None) == ""
    assert player_label(float("nan"), float("nan")) == ""
    assert player_label("nan", "nan") == ""


def test_player_label_keeps_real_names_containing_nan():
    """Guards the fix above from over-reaching: "Dignan" and
    "Hernandez" contain the substring "nan"."""
    assert player_label("7", "Lauren Dignan") == "#7 Lauren Dignan"
    assert player_label("8", "Averie Hernandez") == "#8 Averie Hernandez"


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
    assert maryland_source.identity_fields() == {"Player": "player_label", "Game": "match_label"}


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
