"""
test_query.py
==============
The decision path from "the router produced a decomposition" to "here is
a frame to draw", exercised against the real code (not a re-implementation
of it) with the hand-countable fake source.

Also covers the seed and the round-trip through volley_store, since a
metric that cannot be saved and reloaded is not really committed.
"""

import json
import os
import tempfile

import pandas as pd
import pytest

import _parent_path  # noqa: F401
from recruiting_operations import Rank, Reduce, Slice
from recruiting_tree import NodeKind

from fake_source import sample_source
from test_evaluate import TREE
from volley_query import (
    consolidate_action_results, execute_query_actions, known_games, known_players,
    metric_format_pattern, resolve_game_hint, resolve_player_name, tidy_data,
)
from volley_seed import seed_volley_tree
from volley_store import load_tree, save_tree, tree_from_json_dict, tree_to_json_dict

SOURCE = sample_source()
ROSTER = ["#3 Ajack Malual", "#24 Ally Williams", "#44 Eva Rohrbach", "#17 Eva Rohrbach"]


def action(**overrides):
    base = {"metric_of_interest": "Kills", "skill_group": None, "player": None,
            "game_hint": None, "title": "t", "pipeline": []}
    base.update(overrides)
    return {"actions": [base], "unrecognized_terms": []}


# ── name resolution ────────────────────────────────────────────

def test_resolves_a_bare_surname():
    assert resolve_player_name("Malual", ROSTER) == "#3 Ajack Malual"


def test_resolves_a_jersey_number():
    assert resolve_player_name("#24", ROSTER) == "#24 Ally Williams"
    assert resolve_player_name("24", ROSTER) == "#24 Ally Williams"


def test_jersey_disambiguates_duplicate_names():
    """Two Eva Rohrbachs; the number is the only thing separating them."""
    assert resolve_player_name("#44", ROSTER) == "#44 Eva Rohrbach"
    assert resolve_player_name("#17", ROSTER) == "#17 Eva Rohrbach"


def test_unknown_player_resolves_to_none_rather_than_guessing():
    assert resolve_player_name("Zzzzzz", ROSTER) is None


def test_game_hint_returns_every_match_it_could_mean():
    """Three opponents are played twice, so a hint is legitimately
    ambiguous and both matches must be shown."""
    games = ["Penn State (Oct 5)", "Penn State (Nov 23)", "Minnesota (Nov 21)"]
    assert set(resolve_game_hint("Penn State", games)) == {games[0], games[1]}
    assert resolve_game_hint("Minnesota", games) == ["Minnesota (Nov 21)"]
    assert resolve_game_hint("Nobody", games) == []


# ── execution ──────────────────────────────────────────────────

def test_plain_metric_lookup():
    result = execute_query_actions(action(), TREE, SOURCE)[0]
    assert list(result["result_df"].columns) == ["Game", "Player", "Value", "Note"]
    assert set(result["result_df"]["Player"]) == {"Sloan", "Kendal"}


def test_player_narrowing():
    result = execute_query_actions(action(player="Sloan"), TREE, SOURCE)[0]
    assert set(result["result_df"]["Player"]) == {"Sloan"}


def test_game_scoping():
    result = execute_query_actions(action(game_hint="Game A"), TREE, SOURCE)[0]
    assert set(result["result_df"]["Game"]) == {"Game A"}


def test_unknown_metric_becomes_a_missing_metric_result():
    """The UI offers to author it -- so it must NOT be silently dropped."""
    result = execute_query_actions(action(metric_of_interest="Yendas"), TREE, SOURCE)[0]
    assert result["status"] == "missing_metric"
    assert result["raw_metric_name"] == "Yendas"
    assert result["is_gibberish"] is False


def test_router_flagged_gibberish_is_marked_as_such():
    decomposition = action(metric_of_interest="Yendas")
    decomposition["unrecognized_terms"] = ["yendas"]
    result = execute_query_actions(decomposition, TREE, SOURCE)[0]
    assert result["is_gibberish"] is True


def test_category_action_returns_every_metric_in_the_branch():
    result = execute_query_actions(
        action(metric_of_interest=None, skill_group="Attack"), TREE, SOURCE)[0]
    assert set(result["result_df"]["Metric"]) == {
        "Kills", "Attack Errors", "Attack Attempts", "Kills Per Set", "Hitting Efficiency"}


def test_rank_over_a_category_keeps_every_metric_independently():
    """The condensing bug: ranking a multi-metric frame globally kept
    only `limit` rows across ALL metrics. Each metric must get its own
    ranking."""
    decomposition = action(metric_of_interest=None, skill_group="Attack",
                            pipeline=[Reduce(axis="Game", how="mean"),
                                      Rank(axis="Player", descending=True, limit=1)])
    frame = execute_query_actions(decomposition, TREE, SOURCE)[0]["result_df"]
    assert set(frame["Metric"]) == {
        "Kills", "Attack Errors", "Attack Attempts", "Kills Per Set", "Hitting Efficiency"}


def test_player_filter_overrides_the_actions_own_player():
    """A Player Key click re-runs the same question for someone else,
    without a second trip through the router."""
    result = execute_query_actions(
        action(player="Sloan"), TREE, SOURCE, player_filter=["Kendal"])[0]
    assert set(result["result_df"]["Player"]) == {"Kendal"}


def test_multiple_actions_are_all_executed():
    decomposition = {"actions": [
        {"metric_of_interest": "Kills", "skill_group": None, "player": "Sloan",
         "game_hint": None, "title": "a", "pipeline": []},
        {"metric_of_interest": "Sets Played", "skill_group": None, "player": "Sloan",
         "game_hint": None, "title": "b", "pipeline": []},
    ], "unrecognized_terms": []}
    assert len(execute_query_actions(decomposition, TREE, SOURCE)) == 2


# ── tidy / consolidate ─────────────────────────────────────────

def test_consolidation_merges_actions_into_one_matrix():
    decomposition = {"actions": [
        {"metric_of_interest": "Kills", "skill_group": None, "player": None,
         "game_hint": None, "title": "a", "pipeline": []},
        {"metric_of_interest": "Sets Played", "skill_group": None, "player": None,
         "game_hint": None, "title": "b", "pipeline": []},
    ], "unrecognized_terms": []}
    consolidated = consolidate_action_results(execute_query_actions(decomposition, TREE, SOURCE))
    assert set(consolidated.columns) == {"Player", "Game", "Kills", "Sets Played"}
    row = consolidated[(consolidated["Player"] == "Sloan") & (consolidated["Game"] == "Game A")]
    assert row["Kills"].iloc[0] == 3 and row["Sets Played"].iloc[0] == 3


def test_counts_format_as_whole_numbers_and_rates_do_not():
    """Formatting follows the SPEC KIND, which at event grain is exact."""
    assert metric_format_pattern(TREE, "Kills") == "{:.0f}"
    assert metric_format_pattern(TREE, "Sets Played") == "{:.0f}"
    assert metric_format_pattern(TREE, "Kills Per Set") == "{:.2f}"


# ── seed + persistence ─────────────────────────────────────────

def test_seed_builds_primitives_and_derived_metrics():
    tree, branches = seed_volley_tree(SOURCE.schema)
    labels = {n.label for n in tree.committed.values() if n.kind == NodeKind.LEAF}
    assert "Kills" in labels and "Sets Played" in labels
    assert "Kills Per Set" in labels, "derived metrics must seed after their primitives"
    assert "Attack" in branches and "Sets" in branches


def test_seed_skips_primitives_the_data_cannot_express():
    """The fake source has no Block/Dig rows at all, so those primitives
    must not be committed -- a metric that can only ever return 0 is
    worse than an absent one."""
    tree, _ = seed_volley_tree(SOURCE.schema)
    labels = {n.label for n in tree.committed.values() if n.kind == NodeKind.LEAF}
    assert "Kill Blocks" not in labels
    assert "Digs" not in labels


def test_seeded_tree_round_trips_through_json():
    tree, _ = seed_volley_tree(SOURCE.schema)
    before = {n.label for n in tree.committed.values() if n.kind == NodeKind.LEAF}

    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "kb.json")
        save_tree(tree, path)
        reloaded = load_tree(SOURCE.schema, path)

    assert reloaded is not None
    after = {n.label for n in reloaded.committed.values() if n.kind == NodeKind.LEAF}
    assert before == after


def test_round_trip_preserves_an_event_metrics_where_clause():
    """recruiting_data_store's serializer only knows column/formula and
    would drop the where-clause, i.e. the entire definition."""
    tree, _ = seed_volley_tree(SOURCE.schema)
    reloaded = tree_from_json_dict(tree_to_json_dict(tree), SOURCE.schema)
    kills = next(n for n in reloaded.committed.values()
                 if n.kind == NodeKind.LEAF and n.label == "Kills")
    assert kills.spec.payload["where"] == {"skill": "Attack", "evaluation_code": "#"}
    assert kills.spec.validate() == []


def test_reloaded_metrics_still_compute():
    from volley_evaluate import evaluate_metric
    tree, _ = seed_volley_tree(SOURCE.schema)
    reloaded = tree_from_json_dict(tree_to_json_dict(tree), SOURCE.schema)
    spec = next(n for n in reloaded.committed.values()
                if n.kind == NodeKind.LEAF and n.label == "Kills Per Set").spec
    frame = evaluate_metric(spec, SOURCE, reloaded)
    row = frame[(frame["Player"] == "Sloan") & (frame["Game"] == "Game A")]
    assert row["Value"].iloc[0] == pytest.approx(1.0)


def test_load_tree_returns_none_for_a_missing_or_corrupt_file():
    """A corrupt file must not make the app unstartable -- the caller
    seeds fresh instead."""
    assert load_tree(SOURCE.schema, "/nonexistent/nope.json") is None
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "bad.json")
        with open(path, "w") as handle:
            handle.write("{not json")
        assert load_tree(SOURCE.schema, path) is None


# ── staging / diff / merge, inherited from the tree engine ─────

def test_staging_a_metric_does_not_affect_committed_until_merge():
    from volley_event_spec import make_event_spec
    tree, branches = seed_volley_tree(SOURCE.schema)
    spec = make_event_spec({"skill": "Attack", "evaluation_code": "+"}, "positive", SOURCE.schema)
    tree.add_node(label="Positive Swings", kind=NodeKind.LEAF, parent_id=branches["Attack"],
                   to="staging", spec=spec, authored_by="coach")

    committed = {n.label for n in tree.committed.values() if n.kind == NodeKind.LEAF}
    assert "Positive Swings" not in committed
    assert len(tree.compute_diff()) == 1

    tree.merge()
    committed = {n.label for n in tree.committed.values() if n.kind == NodeKind.LEAF}
    assert "Positive Swings" in committed


def test_add_node_rejects_an_invalid_event_spec():
    """The trust boundary: a spec naming a (skill, code) pair the data
    never contains is refused on the same path a bad formula is."""
    from volley_event_spec import make_event_spec
    tree, branches = seed_volley_tree(SOURCE.schema)
    bad = make_event_spec({"skill": "Serve", "evaluation_code": "+"}, "bogus", SOURCE.schema)
    with pytest.raises(ValueError):
        tree.add_node(label="Bogus", kind=NodeKind.LEAF, parent_id=branches["Attack"],
                       to="staging", spec=bad)
