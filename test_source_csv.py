"""
test_source_csv.py
===================
Proves the unified path is the SAME path for recruiting.

The golden lock (test_golden_parity.py) freezes the app's output end to
end. This file checks the narrower, sharper claim that makes the lock
hold: for every committed metric in the real knowledge base,
evaluate_metric(spec, CsvSource(...)) equals run_metric_query(spec,
...) exactly -- same rows, same order, same values, same Note text.

Comparing against the ORIGINAL function rather than a recorded
expectation is deliberate: if run_metric_query changes, this must fail.
"""

import json
import os
from typing import List, Tuple

import pandas as pd
import pytest

from evaluate_csv import run_category_query, run_metric_query
from query import (
    consolidate_action_results, execute_query_actions, find_leaf_by_exact_label,
    get_branches,
)
import recruiting_data_store
from recruiting_data_store import GameInfo
from evaluate import evaluate_metric
from recruiting_tree import KnowledgeTree, NodeKind
from source import Grain
from source_csv import CsvSource

HERE = os.path.dirname(os.path.abspath(__file__))
_STAGED_TREE_JSON = os.path.join(HERE, "VolleyData_upload", "recruiting_kb_data.json")


def _row(name: str, attack_k: float, attack_e: float, sets_played: float, pass_pct: float) -> dict:
    return {"Name": name, "Attack K": attack_k, "Attack E": attack_e,
            "Sets Sets Played": sets_played, "Receive Pass%": pass_pct}


@pytest.fixture
def real_tree() -> KnowledgeTree:
    if not os.path.exists(_STAGED_TREE_JSON):
        pytest.skip("staged recruiting_kb_data.json not available")
    with open(_STAGED_TREE_JSON) as handle:
        return recruiting_data_store._tree_from_json_dict(json.load(handle))


@pytest.fixture
def games() -> List[Tuple[GameInfo, pd.DataFrame]]:
    """Includes a BLANK cell and a ZERO denominator on purpose: those are
    the two cases where the measure and event grains legitimately
    disagree, so they are exactly where a careless merge of the two
    evaluators would have shown up."""
    return [
        (GameInfo(path="a", filename="a.csv", opponent="Game A"),
         pd.DataFrame([_row("#7 Sloan T.", 12, 2, 3, 0.8),
                       _row("#22 Azana S.", 8, 1, 3, 0.7),
                       {"Name": "Team Totals", "Attack K": 20, "Attack E": 3,
                        "Sets Sets Played": 3, "Receive Pass%": 0.75}])),
        (GameInfo(path="b", filename="b.csv", opponent="Game B"),
         pd.DataFrame([_row("#7 Sloan T.", 9, 3, 0, 0.6),          # zero denominator
                       _row("#22 Azana S.", 15, 2, 3, None)])),    # blank cell
    ]


def _all_committed_specs(tree: KnowledgeTree):
    return [(node.label, node.spec) for node in tree.committed.values()
            if node.kind == NodeKind.LEAF and node.spec is not None]


# ── the adapter itself ─────────────────────────────────────────

def test_grain_and_axes(games):
    source = CsvSource(games)
    assert source.grain is Grain.MEASURE
    assert source.axes() == ["Player", "Game", "Metric"], "a CSV row has no set detail"
    assert "Set" not in source.identity_fields()


def test_team_aggregate_rows_are_not_players(games):
    """Huddle exports carry team totals in the same table; counting them
    as a player would add a phantom to every ranking."""
    labels = CsvSource(games).player_labels()
    assert labels == ["#22 Azana S.", "#7 Sloan T."]
    assert not any("Team" in label for label in labels)


def test_game_labels_keep_selection_order(games):
    assert CsvSource(games).game_labels() == ["Game A", "Game B"]
    assert CsvSource(list(reversed(games))).game_labels() == ["Game B", "Game A"]


def test_schema_marks_numeric_columns_as_measures(games):
    schema = CsvSource(games).schema
    assert schema.fields["Attack K"].role.value == "measure"
    assert schema.fields["Player"].role.value == "identity"
    assert not any(spec.role.value == "dimension" for spec in schema.fields.values()), (
        "a measure-grain row has no attributes to filter on"
    )


# ── parity with the pre-unification path ───────────────────────

def test_every_committed_metric_matches_run_metric_query(real_tree, games):
    source = CsvSource(games)
    checked = 0
    for label, spec in _all_committed_specs(real_tree):
        expected = run_metric_query(spec, real_tree, games)
        produced = evaluate_metric(spec, source, real_tree)
        pd.testing.assert_frame_equal(
            produced.reset_index(drop=True), expected.reset_index(drop=True),
            check_dtype=False, obj=f"metric {label!r}",
        )
        checked += 1
    assert checked >= 5, f"expected the real tree to have metrics to check, got {checked}"


def test_parity_holds_when_filtered_to_one_player(real_tree, games):
    source = CsvSource(games)
    for label, spec in _all_committed_specs(real_tree):
        expected = run_metric_query(spec, real_tree, games, player_name="#7 Sloan T.")
        produced = evaluate_metric(spec, source, real_tree, player="#7 Sloan T.")
        pd.testing.assert_frame_equal(
            produced.reset_index(drop=True), expected.reset_index(drop=True),
            check_dtype=False, obj=f"metric {label!r}",
        )


def test_blank_and_zero_denominator_still_report_errors_not_numbers(real_tree, games):
    """The two grain-specific behaviours the CSV evaluator was kept for.
    If this ever returns 0.0 or NaN with an empty Note, the measure path
    has been quietly replaced by the event one."""
    rate = find_leaf_by_exact_label(real_tree, "Kills Per Set")
    assert rate is not None
    frame = evaluate_metric(rate.spec, CsvSource(games), real_tree)

    zero_denominator = frame[(frame["Game"] == "Game B") & (frame["Player"] == "#7 Sloan T.")]
    assert len(zero_denominator) == 1
    assert pd.isna(zero_denominator["Value"].iloc[0])
    assert zero_denominator["Note"].iloc[0], "a zero denominator must say why, not just be empty"


def test_a_category_matches_run_category_query(real_tree, games):
    branch_id = get_branches(real_tree).get("Attack")
    if branch_id is None:
        pytest.skip("no Attack branch committed")
    from evaluate import evaluate_category

    expected = run_category_query(real_tree, branch_id, games)
    produced = evaluate_category(real_tree, branch_id, CsvSource(games))
    pd.testing.assert_frame_equal(
        produced.reset_index(drop=True), expected.reset_index(drop=True), check_dtype=False,
    )
