"""
test_integration.py
=====================
Integration tests for Tasks 1-2 (router bugfix + recruiting_operations.py
wiring). Exercises REAL production code end to end:

    recruiting_llm._validate_and_repair (real repair/validation)
    -> app.py's action-execution helpers (run_metric_query,
       run_category_query, prepare_pipeline_frame, run_pipeline, tidy_data,
       consolidate_action_results) -- all real, unmodified-by-tests code.

The router's raw JSON is supplied directly (as if a real LLM had already
produced it) for most tests, rather than requiring a live vLLM server --
CI/offline environments can't guarantee one is running. The router's raw
JSON shape is exactly what recruiting_llm._build_router_system_prompt
asks the model for, so this is testing "does the deterministic repair +
execution pipeline behave correctly given a well-formed decomposition",
which is the part that doesn't depend on network availability. The small
number of tests that DO call the real LLM (test_live_*) skip gracefully
via LLMUnavailableError when no server is reachable, rather than failing
the whole suite in an environment without one.

Run with: pytest test_integration.py -v
"""
import json
import math
import os
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pytest

import app
import recruiting_data_store
from recruiting_llm import (
    LLMUnavailableError, _validate_and_repair, decompose_recruiting_query,
    merge_same_shape_actions, parse_phrase_to_formula, parse_phrase_to_formula_llm,
)
from recruiting_operations import Reduce, Rank, Compare, Slice, ValuePredicate
from recruiting_tree import KnowledgeTree

# The real recruiting_kb_data.json now lives in the private VolleyData repo,
# not on local disk -- app.load_committed_tree() needs live GITHUB_DATA_REPO/
# GITHUB_DATA_TOKEN secrets and network access to reach it, neither of which
# CI/offline test runs can guarantee. This points at the local staging copy
# (see VolleyData_upload/ alongside this file, the same file that gets
# uploaded to VolleyData by hand) so real_tree below still exercises the
# genuine committed data, just read straight from disk instead of over the
# network.
_STAGED_TREE_JSON = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "VolleyData_upload", "recruiting_kb_data.json",
)


# ──────────────────────────────────────────────────────────────
# FIXTURES
# ──────────────────────────────────────────────────────────────

@pytest.fixture
def real_tree() -> KnowledgeTree:
    """A fresh copy of the REAL committed tree, loaded straight from this
    project's actual persisted recruiting_kb_data.json -- not a synthetic
    stand-in, so 'Kills Per Set' / 'Passing quality percentage' / 'Kills'
    below are the genuine already-committed metrics, exercising the real
    knowledge base state rather than a test-only fixture tree."""
    assert os.path.exists(_STAGED_TREE_JSON), (
        f"expected the staged copy of recruiting_kb_data.json at {_STAGED_TREE_JSON} "
        "(see VolleyData_upload/) -- this is the same file that gets uploaded to VolleyData"
    )
    with open(_STAGED_TREE_JSON) as f:
        tree = recruiting_data_store._tree_from_json_dict(json.load(f))
    for label in ("Kills Per Set", "Passing quality percentage", "Kills"):
        assert app.find_leaf_by_exact_label(tree, label) is not None, f"expected '{label}' to be committed"
    return tree


def _synthetic_row(name: str, attack_k: float, attack_e: float, sets_played: float, pass_pct: float) -> dict:
    return {"Name": name, "Attack K": attack_k, "Attack E": attack_e,
            "Sets Sets Played": sets_played, "Receive Pass%": pass_pct}


@pytest.fixture
def synthetic_games() -> List[Tuple["app.GameInfo", pd.DataFrame]]:
    """3 controlled games x 2 players, hand-picked values so pipeline
    results (mean/rank/compare) can be asserted EXACTLY rather than just
    'didn't crash'.

    Kills Per Set = Attack K / Sets Sets Played:
        Sloan: Game A = 12/3 = 4.000   Game B = 9/3 = 3.000   Game C = 5/2 = 2.500  (mean 3.1667)
        Azana: Game A =  8/3 = 2.667   Game B = 15/3 = 5.000  Game C = 6/2 = 3.000  (mean 3.5556)
    Sloan's own Kills (Attack K, raw): Game A=12 (>=10), Game B=9 (<10), Game C=5 (<10)
    Azana's own Kills (Attack K, raw): Game A=8 (<10), Game B=15 (>=10), Game C=6 (<10)
    """
    game_a = app.GameInfo(path="synthetic-A", filename="A.csv", opponent="Game A")
    game_b = app.GameInfo(path="synthetic-B", filename="B.csv", opponent="Game B")
    game_c = app.GameInfo(path="synthetic-C", filename="C.csv", opponent="Game C")

    df_a = pd.DataFrame([_synthetic_row("#7 Sloan T.", 12, 2, 3, 0.8), _synthetic_row("#22 Azana S.", 8, 1, 3, 0.7)])
    df_b = pd.DataFrame([_synthetic_row("#7 Sloan T.", 9, 3, 3, 0.6), _synthetic_row("#22 Azana S.", 15, 2, 3, 0.9)])
    df_c = pd.DataFrame([_synthetic_row("#7 Sloan T.", 5, 1, 2, 0.75), _synthetic_row("#22 Azana S.", 6, 0, 2, 0.65)])

    return [(game_a, df_a), (game_b, df_b), (game_c, df_c)]


def _run_repaired_actions(raw_decomposition: dict, tree: KnowledgeTree,
                           game_dfs: List[Tuple["app.GameInfo", pd.DataFrame]],
                           known_player_pool: List[str]) -> List[dict]:
    """
    Mirrors the REAL action-execution loop inside app.py's `with tab_qa:`
    block line for line (same functions, same fetch_player/pipeline/
    auto-append-Slice logic) but driven directly in a test, without a
    Streamlit runtime. Kept as a small local helper (not extracted into
    app.py itself) since app.py's loop is UI code interleaved with widget
    calls -- this reproduces its DECISION logic exactly, calling only the
    real, non-Streamlit functions it calls.
    """
    decomposition = _validate_and_repair(raw_decomposition, tree)
    action_results = []

    for action in decomposition.get("actions", []):
        is_category = bool(action.get("skill_group"))
        pipeline = action.get("pipeline") or []

        if is_category:
            branch_id = app.get_branches(tree).get(action["skill_group"])
            if branch_id is None:
                action_results.append({"action": action, "error": "Category no longer exists."})
                continue
        else:
            leaf = app.find_leaf_by_exact_label(tree, action.get("metric_of_interest"))
            if leaf is None or leaf.spec is None:
                raw_metric_name = action.get("metric_of_interest") or "Unknown Metric"
                unrecognized_lower = {t.lower() for t in decomposition.get("unrecognized_terms", [])}
                action_results.append({
                    "action": action, "status": "missing_metric",
                    "raw_metric_name": raw_metric_name,
                    "is_gibberish": raw_metric_name.strip().lower() in unrecognized_lower,
                })
                continue

        # Mirrors app.py's raw_player/multi_players handling exactly: a
        # merged action's "player" field can be a LIST of raw hints
        # (merge_same_shape_actions, recruiting_llm.py), each of which
        # still needs resolving individually against the real roster.
        raw_player = action.get("player")
        if isinstance(raw_player, list):
            resolved_players = []
            seen_players = set()
            for hint in raw_player:
                candidate = app.resolve_player_name(hint, known_player_pool)
                if candidate is not None and candidate not in seen_players:
                    seen_players.add(candidate)
                    resolved_players.append(candidate)
            multi_players = resolved_players if len(resolved_players) >= 2 else None
            resolved_player = resolved_players[0] if len(resolved_players) == 1 else None
        else:
            resolved_player = app.resolve_player_name(raw_player, known_player_pool)
            multi_players = None
        action_games = [g for g, _ in game_dfs]  # tests pre-resolve games directly

        fetch_player = None if (pipeline or multi_players) else resolved_player
        if is_category:
            result_df = app.run_category_query(tree, branch_id, game_dfs, player_name=fetch_player)
        else:
            result_df = app.run_metric_query(leaf.spec, tree, game_dfs, player_name=fetch_player)

        pipeline_notes: List[str] = []
        if pipeline or multi_players:
            pipeline = list(pipeline)
            has_player_slice = any(isinstance(op, Slice) and op.axis == "Player" for op in pipeline)
            if not has_player_slice:
                if multi_players:
                    pipeline.append(Slice(axis="Player", keep=multi_players))
                elif resolved_player is not None:
                    pipeline.append(Slice(axis="Player", keep=[resolved_player]))
            has_metric_slice = any(isinstance(op, Slice) and op.axis == "Metric" for op in pipeline)
            primary_metric = action.get("metric_of_interest")
            if not is_category and primary_metric and not has_metric_slice:
                pipeline.append(Slice(axis="Metric", keep=[primary_metric]))
            combined_df, fetch_notes = app.prepare_pipeline_frame(
                result_df, pipeline, action.get("metric_of_interest"), tree, game_dfs,
            )
            result_df, run_notes = app.run_pipeline(combined_df, pipeline)
            pipeline_notes = fetch_notes + run_notes

        action_results.append({
            "action": action, "is_category": is_category, "resolved_player": resolved_player,
            "action_games": action_games, "result_df": result_df,
            "pipeline": pipeline, "pipeline_notes": pipeline_notes,
        })

    return action_results


KNOWN_PLAYERS = ["#7 Sloan T.", "#22 Azana S."]


# ──────────────────────────────────────────────────────────────
# 1. REGRESSION: single-metric query, no pipeline
# ──────────────────────────────────────────────────────────────

def test_regression_single_metric_no_pipeline(real_tree, synthetic_games):
    """Must produce IDENTICAL output to the pre-pipeline code path: no
    Metric column inserted, no run_pipeline call, no player-fetch change."""
    raw = {
        "intent_summary": "test", "reasoning": "", "limitations": "", "unrecognized_terms": [],
        "actions": [{
            "metric_of_interest": "Kills Per Set", "skill_group": None,
            "player": "Sloan", "game_hint": None, "title": "Sloan's Kills Per Set",
        }],
    }
    results = _run_repaired_actions(raw, real_tree, synthetic_games, KNOWN_PLAYERS)
    assert len(results) == 1
    item = results[0]
    assert item["pipeline"] == []
    assert item["resolved_player"] == "#7 Sloan T."

    df = item["result_df"]
    assert "Metric" not in df.columns, "no-pipeline result must stay exactly as run_metric_query produces it"
    assert list(df.columns) == ["Game", "Player", "Value", "Note"]
    assert set(df["Player"]) == {"#7 Sloan T."}
    values = dict(zip(df["Game"], df["Value"]))
    assert math.isclose(values["Game A"], 4.0)
    assert math.isclose(values["Game B"], 3.0)
    assert math.isclose(values["Game C"], 2.5)

    # Byte-identical to calling run_metric_query directly ourselves.
    leaf = app.find_leaf_by_exact_label(real_tree, "Kills Per Set")
    direct = app.run_metric_query(leaf.spec, real_tree, synthetic_games, player_name="#7 Sloan T.")
    pd.testing.assert_frame_equal(df.reset_index(drop=True), direct.reset_index(drop=True))


# ──────────────────────────────────────────────────────────────
# 2. Rank -- "who has the highest Kills Per Set"
# ──────────────────────────────────────────────────────────────

def test_rank_highest_kills_per_set(real_tree, synthetic_games):
    raw = {
        "intent_summary": "test", "reasoning": "", "limitations": "", "unrecognized_terms": [],
        "actions": [{
            "metric_of_interest": "Kills Per Set", "skill_group": None,
            "player": None, "game_hint": None, "title": "Highest Kills Per Set",
            "pipeline": [
                {"op": "reduce", "axis": "Game", "how": "mean"},
                {"op": "rank", "axis": "Player", "descending": True, "limit": 1},
            ],
        }],
    }
    results = _run_repaired_actions(raw, real_tree, synthetic_games, KNOWN_PLAYERS)
    df = results[0]["result_df"]
    assert len(df) == 1
    top = df.iloc[0]
    assert top["Player"] == "#22 Azana S."  # mean 3.5556 > Sloan's 3.1667
    assert math.isclose(top["Value"], (8/3 + 15/3 + 6/2) / 3, rel_tol=1e-9)
    assert top["N Used"] == 3


# ──────────────────────────────────────────────────────────────
# 3. Slice with a cross-metric predicate -- "Sloan's passing in games
#    she had 10+ kills"
# ──────────────────────────────────────────────────────────────

def test_slice_cross_metric_predicate(real_tree, synthetic_games):
    raw = {
        "intent_summary": "test", "reasoning": "", "limitations": "", "unrecognized_terms": [],
        "actions": [{
            "metric_of_interest": "Passing quality percentage", "skill_group": None,
            "player": "Sloan", "game_hint": None, "title": "Sloan's Passing, 10+ Kill games",
            "pipeline": [
                {"op": "slice", "axis": "Game", "predicate": {
                    "metric": "Kills", "op": ">=", "threshold": 10, "decided_by_player": None}},
            ],
        }],
    }
    results = _run_repaired_actions(raw, real_tree, synthetic_games, KNOWN_PLAYERS)
    item = results[0]
    # The auto-appended trailing Player-slice (Sloan named, no Player-slice
    # in the router's own pipeline) must show up in what actually ran.
    assert any(isinstance(op, Slice) and op.axis == "Player" for op in item["pipeline"])

    df = item["result_df"]
    assert len(df) == 1, df
    row = df.iloc[0]
    assert row["Player"] == "#7 Sloan T."
    assert row["Game"] == "Game A"  # only game where Sloan's OWN kills (12) >= 10
    assert math.isclose(row["Value"], 0.8)  # Receive Pass% in Game A


def test_cross_metric_predicate_decided_by_named_player(real_tree, synthetic_games):
    """'Everyone's passing in games where SLOAN had 10+ kills' -- one
    player's value gates the axis for everyone, per ValuePredicate's own
    decided_by_player field (already covered at the unit level by
    test_operations.py; this confirms the SAME behavior survives the
    real fetch/repair/execution wiring, not just the algebra in isolation)."""
    raw = {
        "intent_summary": "test", "reasoning": "", "limitations": "", "unrecognized_terms": [],
        "actions": [{
            "metric_of_interest": "Passing quality percentage", "skill_group": None,
            "player": None, "game_hint": None, "title": "Everyone's Passing, Sloan 10+ Kill games",
            "pipeline": [
                {"op": "slice", "axis": "Game", "predicate": {
                    "metric": "Kills", "op": ">=", "threshold": 10, "decided_by_player": "#7 Sloan T."}},
            ],
        }],
    }
    results = _run_repaired_actions(raw, real_tree, synthetic_games, KNOWN_PLAYERS)
    df = results[0]["result_df"]
    # Only Game A qualifies (Sloan's kills=12 there); both players' passing
    # for Game A should come through since no player-slice was applied.
    assert set(df["Game"]) == {"Game A"}
    assert set(df["Player"]) == {"#7 Sloan T.", "#22 Azana S."}


# ──────────────────────────────────────────────────────────────
# 4. Reduce -- "average Kills Per Set across all games"
# ──────────────────────────────────────────────────────────────

def test_reduce_average_across_games(real_tree, synthetic_games):
    raw = {
        "intent_summary": "test", "reasoning": "", "limitations": "", "unrecognized_terms": [],
        "actions": [{
            "metric_of_interest": "Kills Per Set", "skill_group": None,
            "player": "Sloan", "game_hint": None, "title": "Sloan's Average Kills Per Set",
            "pipeline": [{"op": "reduce", "axis": "Game", "how": "mean"}],
        }],
    }
    results = _run_repaired_actions(raw, real_tree, synthetic_games, KNOWN_PLAYERS)
    df = results[0]["result_df"]
    assert len(df) == 1
    row = df.iloc[0]
    assert row["Player"] == "#7 Sloan T."
    assert math.isclose(row["Value"], (4.0 + 3.0 + 2.5) / 3, rel_tol=1e-9)
    assert row["N Used"] == 3


# ──────────────────────────────────────────────────────────────
# 5. Compare -- "is Sloan above team average on Kills Per Set"
# ──────────────────────────────────────────────────────────────

def test_compare_above_team_average(real_tree, synthetic_games):
    raw = {
        "intent_summary": "test", "reasoning": "", "limitations": "", "unrecognized_terms": [],
        "actions": [{
            "metric_of_interest": "Kills Per Set", "skill_group": None,
            "player": "Sloan", "game_hint": None, "title": "Sloan vs Team Average",
            "pipeline": [{"op": "compare", "axis": "Player", "how": "mean", "mode": "difference"}],
        }],
    }
    results = _run_repaired_actions(raw, real_tree, synthetic_games, KNOWN_PLAYERS)
    item = results[0]
    df = item["result_df"]
    assert set(df["Player"]) == {"#7 Sloan T."}, "must be narrowed to Sloan by the auto-appended trailing slice"
    assert len(df) == 3  # one row per game

    by_game = dict(zip(df["Game"], df["Value"]))
    # Game A: Sloan 4.0 vs team mean (4.0+2.667)/2 = 3.3333 -> +0.6667
    assert math.isclose(by_game["Game A"], 4.0 - (4.0 + 8/3) / 2, rel_tol=1e-9)
    # Game B: Sloan 3.0 vs team mean (3.0+5.0)/2 = 4.0 -> -1.0
    assert math.isclose(by_game["Game B"], 3.0 - (3.0 + 5.0) / 2, rel_tol=1e-9)
    # Game C: Sloan 2.5 vs team mean (2.5+3.0)/2 = 2.75 -> -0.25
    assert math.isclose(by_game["Game C"], 2.5 - (2.5 + 3.0) / 2, rel_tol=1e-9)


# ──────────────────────────────────────────────────────────────
# 6. "yendas"-style gibberish -- honest rejection, no AI-drafted formula
# ──────────────────────────────────────────────────────────────

def test_gibberish_term_no_ai_draft_offered(real_tree, synthetic_games):
    """Router flags the term as BOTH metric_of_interest (its best-effort
    echo) AND unrecognized_terms (its own low-confidence signal) -- the
    is_gibberish flag must come out True, and app.py's rendering (traced
    by hand here since it's UI code) must not call the metric author."""
    raw = {
        "intent_summary": "test", "reasoning": "", "limitations": "",
        "unrecognized_terms": ["yendas"],
        "actions": [{
            "metric_of_interest": "Yendas", "skill_group": None,
            "player": "Sloan", "game_hint": None, "title": "",
        }],
    }
    results = _run_repaired_actions(raw, real_tree, synthetic_games, KNOWN_PLAYERS)
    assert len(results) == 1
    item = results[0]
    assert item["status"] == "missing_metric"
    assert item["is_gibberish"] is True, "case-insensitive match against unrecognized_terms must catch 'Yendas' vs 'yendas'"
    # app.py's rendering branches on is_gibberish BEFORE ever calling
    # parse_phrase_to_formula_llm/parse_phrase_to_formula for this item --
    # confirmed by reading the source directly rather than re-deriving it,
    # since that's UI code this test can't execute without a Streamlit
    # runtime.
    with open("app.py") as f:
        src = f.read()
    assert 'if item["is_gibberish"]:' in src
    idx = src.index('if item["is_gibberish"]:')
    next_branch_idx = src.index('st.warning(f"⚠️ **Missing Metric Detected', idx)
    branch_body = src[idx:next_branch_idx]
    assert "continue" in branch_body, "gibberish branch must skip the AI-drafting code entirely"
    assert "parse_phrase_to_formula_llm" not in branch_body


def test_gibberish_entirely_unrecognized_no_actions(real_tree, synthetic_games):
    """The router's OWN grounding rules (already in the prompt before this
    change) say to produce actions: [] for pure gibberish -- confirm the
    no-actions-at-all case round-trips through repair correctly and still
    carries unrecognized_terms for the UI's top-level honest-rejection
    message."""
    raw = {
        "intent_summary": "test", "reasoning": "", "limitations": "",
        "unrecognized_terms": ["yendas"], "actions": [],
    }
    decomposition = _validate_and_repair(raw, real_tree)
    assert decomposition["actions"] == []
    assert decomposition["unrecognized_terms"] == ["yendas"]


# ──────────────────────────────────────────────────────────────
# 7. Genuinely novel-but-plausible metric -- in-situ authoring still fires
# ──────────────────────────────────────────────────────────────

def test_plausible_new_metric_offers_in_situ_authoring(real_tree, synthetic_games):
    raw = {
        "intent_summary": "test", "reasoning": "", "limitations": "",
        "unrecognized_terms": [],  # router did NOT flag this term -- it proposed it in good faith
        "actions": [{
            "metric_of_interest": "Serve Aces Per Set", "skill_group": None,
            "player": "Sloan", "game_hint": None, "title": "",
        }],
    }
    results = _run_repaired_actions(raw, real_tree, synthetic_games, KNOWN_PLAYERS)
    item = results[0]
    assert item["status"] == "missing_metric"
    assert item["is_gibberish"] is False, "not in unrecognized_terms -> plausible new metric, in-situ authoring should still fire"

    # The actual in-situ authoring call app.py makes for a non-gibberish
    # missing metric -- real function, rule-based fallback path (no live
    # LLM needed) since "Serve Aces Per Set" ~ the rule-based aces_per_set
    # rule's phrasing.
    draft = parse_phrase_to_formula(item["raw_metric_name"])
    assert draft.matched, "expected the rule-based fallback to recognize an aces-per-set phrasing"
    assert draft.formula_expr == "[Serve SA] / [Sets Sets Played]"


# ──────────────────────────────────────────────────────────────
# 8. Pipeline repair robustness -- invalid pipeline dropped, action kept
# ──────────────────────────────────────────────────────────────

def test_invalid_pipeline_dropped_not_whole_action(real_tree):
    raw = {
        "intent_summary": "test", "reasoning": "", "limitations": "", "unrecognized_terms": [],
        "actions": [{
            "metric_of_interest": "Kills Per Set", "skill_group": None,
            "player": None, "game_hint": None, "title": "Bad pipeline",
            "pipeline": [{"op": "rank", "axis": "NotAnAxis", "descending": True, "limit": 1}],
        }],
    }
    repaired = _validate_and_repair(raw, real_tree)
    assert len(repaired["actions"]) == 1
    assert repaired["actions"][0]["pipeline"] == []
    assert "Dropped invalid pipeline" in repaired["limitations"]


def test_empty_pipeline_is_a_true_no_op_through_real_repair(real_tree, synthetic_games):
    """Confirms recruiting_operations.run_pipeline's own no-op guarantee
    (already covered by test_operations.py) survives being routed through
    the REAL repair step too, not just called directly."""
    raw = {
        "intent_summary": "test", "reasoning": "", "limitations": "", "unrecognized_terms": [],
        "actions": [{
            "metric_of_interest": "Kills Per Set", "skill_group": None,
            "player": None, "game_hint": None, "title": "", "pipeline": [],
        }],
    }
    results = _run_repaired_actions(raw, real_tree, synthetic_games, KNOWN_PLAYERS)
    df = results[0]["result_df"]
    assert "Metric" not in df.columns
    assert set(df["Player"]) == {"#7 Sloan T.", "#22 Azana S."}


# ──────────────────────────────────────────────────────────────
# 9. merge_same_shape_actions -- undoing the router's own "one action per
#    player" split after the fact, without touching that instruction.
# ──────────────────────────────────────────────────────────────

def _action(metric=None, skill_group=None, player=None, game_hint=None,
            title="", pipeline=None, is_committed=True):
    return {
        "metric_of_interest": metric, "skill_group": skill_group, "player": player,
        "game_hint": game_hint, "title": title, "pipeline": pipeline or [],
        "is_committed": is_committed,
    }


def test_merge_two_same_shaped_single_player_actions():
    actions = [
        _action(metric="Kills Per Set", player="Sloan", title="Sloan's Kills Per Set"),
        _action(metric="Kills Per Set", player="Azana", title="Azana's Kills Per Set"),
    ]
    merged = merge_same_shape_actions(actions)
    assert len(merged) == 1
    assert merged[0]["player"] == ["Sloan", "Azana"]
    assert merged[0]["title"] == "Sloan vs Azana -- Kills Per Set"
    # Everything else carried over from the group, untouched.
    assert merged[0]["metric_of_interest"] == "Kills Per Set"
    assert merged[0]["is_committed"] is True


def test_merge_three_or_more_same_shaped_actions_combine_into_one():
    actions = [
        _action(metric="Kills Per Set", player=name)
        for name in ("Sloan", "Azana", "Yuki", "Priya")
    ]
    merged = merge_same_shape_actions(actions)
    assert len(merged) == 1
    assert merged[0]["player"] == ["Sloan", "Azana", "Yuki", "Priya"]


def test_merge_does_not_combine_different_metric():
    actions = [
        _action(metric="Kills Per Set", player="Sloan"),
        _action(metric="Aces Per Set", player="Azana"),
    ]
    merged = merge_same_shape_actions(actions)
    assert len(merged) == 2
    assert {a["player"] for a in merged} == {"Sloan", "Azana"}


def test_merge_does_not_combine_different_skill_group():
    actions = [
        _action(skill_group="Serve", player="Sloan"),
        _action(skill_group="Attack", player="Azana"),
    ]
    merged = merge_same_shape_actions(actions)
    assert len(merged) == 2


def test_merge_does_not_combine_different_game_hint():
    actions = [
        _action(metric="Kills Per Set", player="Sloan", game_hint="Vegas Aces"),
        _action(metric="Kills Per Set", player="Azana", game_hint="Mavs 816"),
    ]
    merged = merge_same_shape_actions(actions)
    assert len(merged) == 2


def test_merge_does_not_combine_different_pipeline():
    actions = [
        _action(metric="Kills Per Set", player="Sloan", pipeline=[Rank(axis="Player", limit=1)]),
        _action(metric="Kills Per Set", player="Azana", pipeline=[]),
    ]
    merged = merge_same_shape_actions(actions)
    assert len(merged) == 2


def test_lone_action_passes_through_unchanged():
    actions = [_action(metric="Kills Per Set", player="Sloan", title="Sloan's Kills Per Set")]
    merged = merge_same_shape_actions(actions)
    assert merged == actions


def test_merge_skips_actions_with_null_or_duplicate_player():
    # A null player and an already-multi player never merge with anything,
    # even same-shaped -- only distinct, non-null, single-string players
    # are mergeable.
    actions = [
        _action(metric="Kills Per Set", player=None),
        _action(metric="Kills Per Set", player="Sloan"),
        _action(metric="Kills Per Set", player=["Already", "Merged"]),
    ]
    merged = merge_same_shape_actions(actions)
    assert merged == actions  # nothing here is a mergeable pair, all pass through


def test_merge_end_to_end_through_real_repair_and_execution(real_tree, synthetic_games):
    """The full path: router-shaped JSON (as if the model split "Sloan's
    and Azana's Kills Per Set" into two actions, per its own system-prompt
    instruction) -> _validate_and_repair (merges them) -> the real
    execution loop (resolves each raw hint, Slice-narrows) -> one result
    containing BOTH players, not two separate results."""
    raw = {
        "intent_summary": "test", "reasoning": "", "limitations": "", "unrecognized_terms": [],
        "actions": [
            {"metric_of_interest": "Kills Per Set", "skill_group": None,
             "player": "Sloan", "game_hint": None, "title": "Sloan's Kills Per Set"},
            {"metric_of_interest": "Kills Per Set", "skill_group": None,
             "player": "Azana", "game_hint": None, "title": "Azana's Kills Per Set"},
        ],
    }
    repaired = _validate_and_repair(raw, real_tree)
    assert len(repaired["actions"]) == 1, "the two same-shaped single-player actions should have merged"
    assert repaired["actions"][0]["player"] == ["Sloan", "Azana"]

    results = _run_repaired_actions(raw, real_tree, synthetic_games, KNOWN_PLAYERS)
    assert len(results) == 1
    df = results[0]["result_df"]
    assert set(df["Player"]) == {"#7 Sloan T.", "#22 Azana S."}


# ──────────────────────────────────────────────────────────────
# 10. JSON persistence untouched -- regression guard per the task's own
#     instruction to treat any breakage here as cross-layer leakage.
# ──────────────────────────────────────────────────────────────

def test_json_persistence_round_trip_untouched(real_tree):
    """Persistence itself now goes over the network to the private
    VolleyData repo (see test_data_store.py's mocked save/load round-trip
    tests for that transport) -- this checks the serialization format
    itself is lossless, independent of transport."""
    serialized = json.dumps(recruiting_data_store.tree_to_json_dict(real_tree))
    reloaded = recruiting_data_store._tree_from_json_dict(json.loads(serialized))
    assert set(reloaded.committed.keys()) == set(real_tree.committed.keys())
    for node_id, node in real_tree.committed.items():
        reloaded_node = reloaded.committed[node_id]
        assert reloaded_node.label == node.label
        assert reloaded_node.kind == node.kind
        if node.spec is not None:
            assert reloaded_node.spec.payload == node.spec.payload


# ──────────────────────────────────────────────────────────────
# 11. LIVE tests -- real vLLM call, skip gracefully if unreachable
# ──────────────────────────────────────────────────────────────

def test_live_router_yendas_query(real_tree):
    known_games = ["Game A", "Game B", "Game C"]
    try:
        decomposition = decompose_recruiting_query("sloan's yendas per game", real_tree, known_games)
    except LLMUnavailableError:
        pytest.skip("No live vLLM server reachable -- start vllm.sh to exercise this test.")
    # Whatever the model actually did, it must not have crashed the pipeline,
    # and per the grounding rules it should either produce no action for
    # "yendas" or flag it in unrecognized_terms (or both) -- never silently
    # invent a typo-corrected metric with no trace of low confidence.
    assert "actions" in decomposition
    flagged = any("yendas" in t.lower() for t in decomposition.get("unrecognized_terms", []))
    metric_names = [a.get("metric_of_interest") for a in decomposition["actions"] if a.get("metric_of_interest")]
    plausible_guess = any(m and "yenda" in m.lower() for m in metric_names)
    assert flagged or not plausible_guess or decomposition["actions"] == [], (
        "'yendas' should either be flagged in unrecognized_terms, or not produce "
        f"a same-word metric guess at all; got actions={decomposition['actions']}, "
        f"unrecognized_terms={decomposition.get('unrecognized_terms')}"
    )


def test_live_metric_author_plausible_metric(real_tree):
    try:
        result = parse_phrase_to_formula_llm("net kills per set", real_tree)
    except LLMUnavailableError:
        pytest.skip("No live vLLM server reachable -- start vllm.sh to exercise this test.")
    assert result.matched
    assert result.formula_expr or result.column_ref
