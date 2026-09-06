"""
test_query.py
==============
The decision path from "the router produced a decomposition" to "here is
a frame to draw", exercised against the real code (not a re-implementation
of it) with the hand-countable fake source.

Also covers the seed and the round-trip through store_dvw, since a
metric that cannot be saved and reloaded is not really committed.
"""

import json
import os
import tempfile

import pandas as pd
import pytest

from recruiting_operations import Rank, Reduce, Slice
from recruiting_tree import NodeKind

from fake_source import sample_source
from test_evaluate import TREE
from query import (
    consolidate_action_results, execute_query_actions, known_games, known_players,
    metric_format_pattern, resolve_game_hint, resolve_player_name, tidy_data,
)
from seed_dvw import seed_dvw_tree
from store_dvw import load_tree, save_tree, tree_from_json_dict, tree_to_json_dict

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
    tree, branches = seed_dvw_tree(SOURCE.schema)
    labels = {n.label for n in tree.committed.values() if n.kind == NodeKind.LEAF}
    assert "Kills" in labels and "Sets Played" in labels
    assert "Kills Per Set" in labels, "derived metrics must seed after their primitives"
    assert "Attack" in branches and "Sets" in branches


def test_seed_skips_primitives_the_data_cannot_express():
    """The fake source has no Block/Dig rows at all, so those primitives
    must not be committed -- a metric that can only ever return 0 is
    worse than an absent one."""
    tree, _ = seed_dvw_tree(SOURCE.schema)
    labels = {n.label for n in tree.committed.values() if n.kind == NodeKind.LEAF}
    assert "Kill Blocks" not in labels
    assert "Digs" not in labels


def test_seeded_tree_round_trips_through_json():
    tree, _ = seed_dvw_tree(SOURCE.schema)
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
    tree, _ = seed_dvw_tree(SOURCE.schema)
    reloaded = tree_from_json_dict(tree_to_json_dict(tree), SOURCE.schema)
    kills = next(n for n in reloaded.committed.values()
                 if n.kind == NodeKind.LEAF and n.label == "Kills")
    assert kills.spec.payload["where"] == {"skill": "Attack", "evaluation_code": "#"}
    assert kills.spec.validate() == []


def test_reloaded_metrics_still_compute():
    from evaluate import evaluate_metric
    tree, _ = seed_dvw_tree(SOURCE.schema)
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
    from event_spec import make_event_spec
    tree, branches = seed_dvw_tree(SOURCE.schema)
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
    from event_spec import make_event_spec
    tree, branches = seed_dvw_tree(SOURCE.schema)
    bad = make_event_spec({"skill": "Serve", "evaluation_code": "+"}, "bogus", SOURCE.schema)
    with pytest.raises(ValueError):
        tree.add_node(label="Bogus", kind=NodeKind.LEAF, parent_id=branches["Attack"],
                       to="staging", spec=bad)


# ── the Set drill-down through the query layer ─────────────────

class TestSetScoping:
    """set_hint NARROWS the data; a pipeline naming the Set axis SPLITS
    the cube by set. They are different things and they compose."""

    @staticmethod
    def _action(**overrides):
        action = {"metric_of_interest": "Kills", "skill_group": None, "player": None,
                  "game_hint": None, "set_hint": None, "title": "t", "pipeline": []}
        action.update(overrides)
        return {"actions": [action]}

    def _run(self, tree, source, **overrides):
        return execute_query_actions(self._action(**overrides), tree, source)[0]

    def test_no_set_hint_summarises_every_set(self):
        tree, _ = seed_dvw_tree(SOURCE.schema)
        result = self._run(tree, SOURCE)
        assert "Set" not in result["result_df"].columns, "the default stays a per-match summary"
        assert result["set_hint"] is None and result["set_note"] is None

    def test_set_hint_narrows_without_adding_a_set_column(self):
        """Asking about one set is a filter, not a split: the answer is
        still one row per player per match, just computed from less."""
        tree, _ = seed_dvw_tree(SOURCE.schema)
        everything = self._run(tree, SOURCE)["result_df"]
        one_set = self._run(tree, SOURCE, set_hint="Set 1")["result_df"]

        assert "Set" not in one_set.columns
        assert one_set["Value"].sum() <= everything["Value"].sum(), (
            "one set cannot contain more kills than the whole match"
        )

    def test_ranking_on_the_set_axis_splits_the_cube(self):
        tree, _ = seed_dvw_tree(SOURCE.schema)
        from fake_source import sample_set_source
        source = sample_set_source()
        tree, _ = seed_dvw_tree(source.schema)
        result = self._run(tree, source, pipeline=[Rank(axis="Set", descending=True, limit=3)])
        assert "Set" in result["result_df"].columns
        assert not result["result_df"].empty


class TestSetOnASourceThatHasNoSets:
    """The refusal that matters: silently returning match totals for "in
    set 3" is a wrong answer that looks like a right one."""

    @staticmethod
    def _measure_source():
        import pandas as pd
        from source_csv import CsvSource

        frame = pd.DataFrame([
            {"Name": "#7 Sloan T.", "Attack K": 12, "Sets Sets Played": 3},
            {"Name": "#22 Azana S.", "Attack K": 8, "Sets Sets Played": 3},
        ])
        return CsvSource([("Game A", frame)])

    @staticmethod
    def _tree():
        from recruiting_tree import KnowledgeTree, NodeKind, make_column_spec
        tree = KnowledgeTree()
        tree.add_root()
        branch = tree.add_node(label="Attack", kind=NodeKind.BRANCH,
                               parent_id=tree.root_id, to="committed")
        tree.add_node(label="Kills", kind=NodeKind.LEAF, parent_id=branch, to="committed",
                      spec=make_column_spec("Attack K"))
        tree.seed_from_committed()
        return tree

    def test_a_set_hint_is_refused_by_name_and_the_answer_still_comes_back(self):
        source, tree = self._measure_source(), self._tree()
        result = execute_query_actions(
            {"actions": [{"metric_of_interest": "Kills", "skill_group": None, "player": None,
                          "game_hint": None, "set_hint": "Set 1", "title": "t", "pipeline": []}]},
            tree, source,
        )[0]
        assert result["set_hint"] is None, "the filter must be dropped, not applied to nothing"
        assert "can't be broken down by set" in result["set_note"]
        assert not result["result_df"].empty, "the coach still gets the match totals"
        assert "Set" not in result["result_df"].columns

    def test_a_set_axis_operation_is_refused_the_same_way(self):
        source, tree = self._measure_source(), self._tree()
        result = execute_query_actions(
            {"actions": [{"metric_of_interest": "Kills", "skill_group": None, "player": None,
                          "game_hint": None, "set_hint": None, "title": "t",
                          "pipeline": [Rank(axis="Set", descending=True, limit=3)]}]},
            tree, source,
        )[0]
        assert result["set_note"] and "can't be broken down by set" in result["set_note"]
