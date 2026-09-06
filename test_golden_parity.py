"""
test_golden_parity.py
======================
A BEHAVIOUR LOCK for the recruiting (CSV) path, written immediately
before the unification refactor and intended to outlive it.

The other suites assert properties ("rank returns the top 3", "reduce
averages"). This one asserts *the exact numbers the app produces today*,
for a battery of representative decompositions, recorded in
`golden_recruiting.json`. Its only job is to make the sentence "the
recruiting side behaves identically after the refactor" checkable
instead of hopeful.

Run against the real committed tree (the staged copy in
VolleyData_upload/) and the same hand-computed synthetic games
test_integration.py uses, so a diff here is a real behaviour change and
never a data-drift artifact.

To re-record after an INTENTIONAL change:
    GOLDEN_UPDATE=1 pytest test_golden_parity.py
and read the resulting JSON diff before committing it -- that diff is
the whole point of the file.
"""

import json
import math
import os
from typing import Any, Dict, List, Tuple

import pandas as pd
import pytest

import app
import recruiting_data_store
from recruiting_data_store import GameInfo
from recruiting_llm import _validate_and_repair
from recruiting_operations import pipeline_to_dicts
from source_csv import CsvSource
from recruiting_tree import KnowledgeTree

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN_PATH = os.path.join(HERE, "golden_recruiting.json")
_STAGED_TREE_JSON = os.path.join(HERE, "VolleyData_upload", "recruiting_kb_data.json")

KNOWN_PLAYERS = ["#7 Sloan T.", "#22 Azana S."]


# ──────────────────────────────────────────────────────────────
# FIXTURES -- deliberately the same inputs test_integration.py uses
# ──────────────────────────────────────────────────────────────

def _synthetic_row(name: str, attack_k: float, attack_e: float,
                   sets_played: float, pass_pct: float) -> dict:
    return {"Name": name, "Attack K": attack_k, "Attack E": attack_e,
            "Sets Sets Played": sets_played, "Receive Pass%": pass_pct}


@pytest.fixture
def real_tree() -> KnowledgeTree:
    if not os.path.exists(_STAGED_TREE_JSON):
        pytest.skip("staged recruiting_kb_data.json not available")
    with open(_STAGED_TREE_JSON) as handle:
        return recruiting_data_store._tree_from_json_dict(json.load(handle))


@pytest.fixture
def synthetic_games() -> List[Tuple[Any, pd.DataFrame]]:
    """Kills Per Set = Attack K / Sets Sets Played
           Sloan: 12/3=4.000, 9/3=3.000, 5/2=2.500   (mean 3.1667)
           Azana:  8/3=2.667, 15/3=5.000, 6/2=3.000  (mean 3.5556)
    """
    games = [
        (GameInfo(path="synthetic-A", filename="A.csv", opponent="Game A"),
         pd.DataFrame([_synthetic_row("#7 Sloan T.", 12, 2, 3, 0.8),
                       _synthetic_row("#22 Azana S.", 8, 1, 3, 0.7)])),
        (GameInfo(path="synthetic-B", filename="B.csv", opponent="Game B"),
         pd.DataFrame([_synthetic_row("#7 Sloan T.", 9, 3, 3, 0.6),
                       _synthetic_row("#22 Azana S.", 15, 2, 3, 0.9)])),
        (GameInfo(path="synthetic-C", filename="C.csv", opponent="Game C"),
         pd.DataFrame([_synthetic_row("#7 Sloan T.", 5, 1, 2, 0.75),
                       _synthetic_row("#22 Azana S.", 6, 0, 2, 0.65)])),
    ]
    return games


@pytest.fixture
def patched_loader(synthetic_games):
    """Kept as a fixture so the test signatures below are unchanged from
    when they were recorded. Nothing to patch any more: after
    unification the frames go in through CsvSource rather than being
    fetched by the query layer, which is the point."""
    return synthetic_games


# ──────────────────────────────────────────────────────────────
# THE BATTERY -- one entry per behaviour worth freezing
# ──────────────────────────────────────────────────────────────

CASES: Dict[str, dict] = {
    "single_metric_one_player": {
        "actions": [{"metric_of_interest": "Kills Per Set", "skill_group": None,
                     "player": "Sloan", "game_hint": None,
                     "title": "Sloan's Kills Per Set", "pipeline": []}],
    },
    "single_metric_all_players": {
        "actions": [{"metric_of_interest": "Kills Per Set", "skill_group": None,
                     "player": None, "game_hint": None,
                     "title": "Kills Per Set", "pipeline": []}],
    },
    "category_browse": {
        "actions": [{"metric_of_interest": None, "skill_group": "Attack",
                     "player": "Sloan", "game_hint": None,
                     "title": "Sloan's Attacking", "pipeline": []}],
    },
    "rank_top_players": {
        "actions": [{"metric_of_interest": "Kills Per Set", "skill_group": None,
                     "player": None, "game_hint": None, "title": "Top Kills Per Set",
                     "pipeline": [{"op": "rank", "axis": "Player", "descending": True, "limit": 3}]}],
    },
    "rank_within_category": {
        "actions": [{"metric_of_interest": None, "skill_group": "Attack",
                     "player": None, "game_hint": None, "title": "Attack leaders",
                     "pipeline": [{"op": "rank", "axis": "Player", "descending": True, "limit": 3}]}],
    },
    "reduce_mean_across_games": {
        "actions": [{"metric_of_interest": "Kills Per Set", "skill_group": None,
                     "player": "Sloan", "game_hint": None, "title": "Sloan average",
                     "pipeline": [{"op": "reduce", "axis": "Game", "how": "mean"}]}],
    },
    "slice_cross_metric_predicate": {
        "actions": [{"metric_of_interest": "Kills Per Set", "skill_group": None,
                     "player": "Sloan", "game_hint": None, "title": "Games with 10+ kills",
                     "pipeline": [{"op": "slice", "axis": "Game",
                                   "predicate": {"metric": "Kills", "op": ">=", "threshold": 10,
                                                 "decided_by_player": None}}]}],
    },
    "slice_predicate_decided_by_named_player": {
        "actions": [{"metric_of_interest": "Kills Per Set", "skill_group": None,
                     "player": None, "game_hint": None, "title": "Games where Sloan had 10+",
                     "pipeline": [{"op": "slice", "axis": "Game",
                                   "predicate": {"metric": "Kills", "op": ">=", "threshold": 10,
                                                 "decided_by_player": "#7 Sloan T."}}]}],
    },
    "compare_above_team_average": {
        "actions": [{"metric_of_interest": "Kills Per Set", "skill_group": None,
                     "player": None, "game_hint": None, "title": "Above average",
                     "pipeline": [{"op": "compare", "axis": "Player", "baseline": "mean"}]}],
    },
    "two_players_merged": {
        "actions": [
            {"metric_of_interest": "Kills Per Set", "skill_group": None, "player": "Sloan",
             "game_hint": None, "title": "Sloan", "pipeline": []},
            {"metric_of_interest": "Kills Per Set", "skill_group": None, "player": "Azana",
             "game_hint": None, "title": "Azana", "pipeline": []},
        ],
    },
    "two_metrics_same_player": {
        "actions": [
            {"metric_of_interest": "Kills Per Set", "skill_group": None, "player": "Sloan",
             "game_hint": None, "title": "KPS", "pipeline": []},
            {"metric_of_interest": "Kills", "skill_group": None, "player": "Sloan",
             "game_hint": None, "title": "Kills", "pipeline": []},
        ],
    },
    "missing_metric": {
        "actions": [{"metric_of_interest": "Blocks Per Possession", "skill_group": None,
                     "player": "Sloan", "game_hint": None, "title": "nope", "pipeline": []}],
    },
    "gibberish_metric": {
        "actions": [{"metric_of_interest": "florbnak", "skill_group": None,
                     "player": "Sloan", "game_hint": None, "title": "nope", "pipeline": []}],
        "unrecognized_terms": ["florbnak"],
    },
    "unresolvable_game_hint": {
        "actions": [{"metric_of_interest": "Kills Per Set", "skill_group": None,
                     "player": "Sloan", "game_hint": "Nonexistent U",
                     "title": "bad game", "pipeline": []}],
    },
}


# ──────────────────────────────────────────────────────────────
# DETERMINISTIC SERIALISATION
# ──────────────────────────────────────────────────────────────

def _round(value: Any) -> Any:
    """Floats are rounded to 6dp so the lock survives platform-level
    float formatting without loosening far enough to hide a real change."""
    if isinstance(value, float):
        return "NaN" if math.isnan(value) else round(value, 6)
    if pd.isna(value) if not isinstance(value, (list, dict, str)) else False:
        return None
    return value


def _frame_to_records(frame: pd.DataFrame) -> dict:
    if frame is None:
        return {"columns": [], "rows": []}
    return {
        "columns": list(frame.columns),
        "rows": [[_round(v) for v in row] for row in frame.itertuples(index=False, name=None)],
    }


def _snapshot(action_results: List[dict], consolidated: pd.DataFrame) -> dict:
    entries = []
    for result in action_results:
        entries.append({
            "title": result["action"].get("title"),
            "status": result.get("status"),
            "error": result.get("error"),
            "is_gibberish": result.get("is_gibberish"),
            "raw_metric_name": result.get("raw_metric_name"),
            "is_category": result.get("is_category"),
            "resolved_player": result.get("resolved_player"),
            "game_note": result.get("game_note"),
            # action_games is a list of Game-axis LABELS now rather than
        # GameInfo objects. The values are the same strings, which is why
        # the recorded snapshot is expected to be unchanged.
        "games": [str(g) for g in result.get("action_games") or []],
            "pipeline": pipeline_to_dicts(result.get("pipeline") or []),
            "pipeline_notes": result.get("pipeline_notes") or [],
            "result": _frame_to_records(result.get("result_df")),
        })
    return {"actions": entries, "consolidated": _frame_to_records(consolidated)}


def _run_case(raw: dict, tree: KnowledgeTree, source) -> dict:
    decomposition = _validate_and_repair(dict(raw), tree)
    action_results = app.execute_query_actions(decomposition, tree, source)
    consolidated = app.consolidate_action_results(action_results)
    return _snapshot(action_results, consolidated)


# ──────────────────────────────────────────────────────────────
# THE LOCK
# ──────────────────────────────────────────────────────────────

def _current(tree, synthetic_games) -> dict:
    source = CsvSource(synthetic_games)
    return {name: _run_case(raw, tree, source) for name, raw in sorted(CASES.items())}


def test_recruiting_behaviour_matches_the_golden_snapshot(real_tree, synthetic_games, patched_loader):
    produced = _current(real_tree, synthetic_games)

    if os.environ.get("GOLDEN_UPDATE"):
        with open(GOLDEN_PATH, "w") as handle:
            json.dump(produced, handle, indent=2, sort_keys=True)
        pytest.skip(f"re-recorded {GOLDEN_PATH} -- review the diff before committing")

    if not os.path.exists(GOLDEN_PATH):
        pytest.fail(f"no golden file at {GOLDEN_PATH}; record one with GOLDEN_UPDATE=1")

    with open(GOLDEN_PATH) as handle:
        expected = json.load(handle)

    assert set(produced) == set(expected), "the set of locked cases changed"
    differing = [name for name in sorted(expected) if produced[name] != expected[name]]
    if differing:
        first = differing[0]
        raise AssertionError(
            f"recruiting behaviour changed in {len(differing)} case(s): {differing}\n\n"
            f"--- {first} expected ---\n{json.dumps(expected[first], indent=2, sort_keys=True)}\n\n"
            f"--- {first} produced ---\n{json.dumps(produced[first], indent=2, sort_keys=True)}"
        )


def test_the_battery_actually_produces_numbers(real_tree, synthetic_games, patched_loader):
    """Guards the lock itself: a snapshot of fourteen empty frames would
    pass the comparison above forever while asserting nothing."""
    produced = _current(real_tree, synthetic_games)
    with_rows = [name for name, snap in produced.items()
                 if any(entry["result"]["rows"] for entry in snap["actions"])]
    assert len(with_rows) >= 10, f"only {len(with_rows)} cases produced rows: {sorted(produced)}"
