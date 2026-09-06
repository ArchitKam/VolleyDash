"""
test_evaluate.py
=================
The evaluator, against a source small enough that every expected number
is worked out by hand in the test (see fake_source.sample_source).
"""

import pandas as pd
import pytest

from recruiting_tree import KnowledgeTree, NodeKind

from fake_source import sample_source
from evaluate import MetricEvaluationError, evaluate_category, evaluate_metric, universe_index
from event_spec import make_event_spec, make_measure_spec, make_metric_formula_spec

SOURCE = sample_source()
SCHEMA = SOURCE.schema


def build_tree():
    """Kills / Attack Errors / Attack Attempts / Sets Played, plus two
    formulas over them."""
    tree = KnowledgeTree()
    tree.add_root()
    attack = tree.add_node(label="Attack", kind=NodeKind.BRANCH,
                            parent_id=tree.root_id, to="committed")
    sets = tree.add_node(label="Sets", kind=NodeKind.BRANCH,
                          parent_id=tree.root_id, to="committed")

    for label, where in [("Kills", {"skill": "Attack", "evaluation_code": "#"}),
                         ("Attack Errors", {"skill": "Attack", "evaluation_code": "="}),
                         ("Attack Attempts", {"skill": "Attack"})]:
        tree.add_node(label=label, kind=NodeKind.LEAF, parent_id=attack, to="committed",
                       spec=make_event_spec(where, label, SCHEMA))
    tree.add_node(label="Sets Played", kind=NodeKind.LEAF, parent_id=sets, to="committed",
                   spec=make_measure_spec("sets_played", "Sets played"))

    labels = {"Kills", "Attack Errors", "Attack Attempts", "Sets Played"}
    tree.add_node(label="Kills Per Set", kind=NodeKind.LEAF, parent_id=attack, to="committed",
                   spec=make_metric_formula_spec("[Kills] / [Sets Played]", "kps", labels))
    tree.add_node(label="Hitting Efficiency", kind=NodeKind.LEAF, parent_id=attack, to="committed",
                   spec=make_metric_formula_spec(
                       "([Kills] - [Attack Errors]) / [Attack Attempts]", "eff", labels))
    tree.seed_from_committed()
    return tree, attack


TREE, ATTACK_BRANCH = build_tree()


def spec_for(label):
    return next(n for n in TREE.committed.values()
                if n.kind == NodeKind.LEAF and n.label == label).spec


def value_of(label, player, game):
    frame = evaluate_metric(spec_for(label), SOURCE, TREE)
    row = frame[(frame["Player"] == player) & (frame["Game"] == game)]
    assert len(row) == 1, f"expected exactly one row for {player}/{game}"
    return row["Value"].iloc[0]


# ── shape ──────────────────────────────────────────────────────

def test_output_matches_the_csv_paths_tidy_shape():
    """Columns must match run_metric_query's so the pipeline, tidy step
    and chart encoder all consume this unchanged."""
    assert list(evaluate_metric(spec_for("Kills"), SOURCE, TREE).columns) == \
        ["Game", "Player", "Value", "Note"]


def test_universe_is_every_player_match_pair_that_played():
    index = universe_index(SOURCE)
    assert set(index) == {("Sloan", "Game A"), ("Sloan", "Game B"),
                          ("Kendal", "Game A"), ("Kendal", "Game B")}


# ── event counting ─────────────────────────────────────────────

def test_counts_events_matching_the_filter():
    assert value_of("Kills", "Sloan", "Game A") == 3
    assert value_of("Kills", "Sloan", "Game B") == 1
    assert value_of("Kills", "Kendal", "Game A") == 2


def test_a_player_who_played_but_recorded_none_is_zero_not_missing():
    """Kendal played Game B and took no attacks: 0 kills, not NaN --
    "did not do it" and "was not there" must stay distinguishable."""
    assert value_of("Kills", "Kendal", "Game B") == 0
    assert value_of("Attack Attempts", "Kendal", "Game B") == 0


def test_skill_only_filter_counts_every_outcome():
    assert value_of("Attack Attempts", "Sloan", "Game A") == 5  # 3 kills + 1 error + 1 positive


def test_a_code_never_paired_with_that_skill_counts_zero_everywhere():
    spec = make_event_spec({"skill": "Reception", "evaluation_code": "#"}, "d", SCHEMA)
    frame = evaluate_metric(spec, SOURCE, TREE)
    assert frame[frame["Player"] == "Kendal"].set_index("Game").loc["Game B", "Value"] == 1


# ── measures ───────────────────────────────────────────────────

def test_measure_reads_from_the_sources_measure_frame():
    assert value_of("Sets Played", "Sloan", "Game A") == 3
    assert value_of("Sets Played", "Kendal", "Game B") == 4


# ── formulas ───────────────────────────────────────────────────

def test_formula_composes_over_other_metrics():
    assert value_of("Kills Per Set", "Sloan", "Game A") == pytest.approx(3 / 3)
    assert value_of("Kills Per Set", "Sloan", "Game B") == pytest.approx(1 / 2)


def test_formula_with_several_operands_and_precedence():
    # (3 kills - 1 error) / 5 attacks
    assert value_of("Hitting Efficiency", "Sloan", "Game A") == pytest.approx((3 - 1) / 5)


def test_division_by_zero_is_nan_not_inf():
    """Kendal took no attacks in Game B, so his hitting efficiency is
    undefined -- NaN, never inf and never 0."""
    value = value_of("Hitting Efficiency", "Kendal", "Game B")
    assert pd.isna(value)


# ── failure handling ───────────────────────────────────────────

def test_circular_reference_is_reported_not_infinitely_recursed():
    tree = KnowledgeTree()
    tree.add_root()
    branch = tree.add_node(label="B", kind=NodeKind.BRANCH, parent_id=tree.root_id, to="committed")
    # Two metrics that reference each other. Validation cannot catch
    # this (each token exists); the evaluator's cycle guard must.
    tree.add_node(label="A", kind=NodeKind.LEAF, parent_id=branch, to="committed",
                   spec=make_metric_formula_spec("[B] + 1", "a", {"B"}))
    tree.add_node(label="B", kind=NodeKind.LEAF, parent_id=branch, to="committed",
                   spec=make_metric_formula_spec("[A] + 1", "b", {"A"}))
    tree.seed_from_committed()

    spec = next(n for n in tree.committed.values()
                if n.kind == NodeKind.LEAF and n.label == "A").spec
    frame = evaluate_metric(spec, SOURCE, tree)
    assert frame["Value"].isna().all()
    assert "Circular metric reference" in frame["Note"].iloc[0]


def test_a_formula_naming_a_missing_metric_reports_rather_than_raises():
    spec = make_metric_formula_spec("[Ghost] * 2", "g", {"Ghost"})
    frame = evaluate_metric(spec, SOURCE, TREE)
    assert frame["Value"].isna().all()
    assert "not an existing metric" in frame["Note"].iloc[0]


def test_player_filter_narrows_the_result():
    frame = evaluate_metric(spec_for("Kills"), SOURCE, TREE, player="Sloan")
    assert set(frame["Player"]) == {"Sloan"}


# ── categories ─────────────────────────────────────────────────

def test_category_returns_every_metric_in_the_branch():
    frame = evaluate_category(TREE, ATTACK_BRANCH, SOURCE)
    assert set(frame["Metric"]) == {"Kills", "Attack Errors", "Attack Attempts",
                                    "Kills Per Set", "Hitting Efficiency"}
    assert list(frame.columns) == ["Metric", "Game", "Player", "Value", "Note"]
