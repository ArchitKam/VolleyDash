import pandas as pd

from recruiting_encoding import (
    EncodingAssignment, default_encoding, reconcile_encoding, set_game_order,
    slot_options, varying_axes,
)


def _df(rows):
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────
# default_encoding -- 0/1/2/3 varying axes
# ──────────────────────────────────────────────────────────────

def test_zero_varying_axes_single_player_single_game():
    df = _df([{"Player": "Sloan", "Game": "Vegas Aces", "Value": 12}])
    assert varying_axes(df) == []
    enc = default_encoding(df)
    assert enc == EncodingAssignment()


def test_one_varying_axis_game_only():
    df = _df([
        {"Player": "Sloan", "Game": "Vegas Aces", "Value": 12},
        {"Player": "Sloan", "Game": "Mavs 816", "Value": 8},
    ])
    assert varying_axes(df) == ["Game"]
    enc = default_encoding(df)
    assert enc.position == "Game"
    assert enc.color is None
    assert enc.facet is None


def test_one_varying_axis_player_only_no_game_column():
    # A plain single-metric, single-game query's frame has no Metric
    # column at all -- Player is the only varying axis here.
    df = _df([
        {"Player": "Sloan", "Value": 12},
        {"Player": "Azana", "Value": 9},
    ])
    assert varying_axes(df) == ["Player"]
    enc = default_encoding(df)
    assert enc.position == "Player"
    assert enc.color is None
    assert enc.facet is None


def test_two_varying_axes_player_and_game_game_takes_position():
    df = _df([
        {"Player": "Sloan", "Game": "Vegas Aces", "Value": 12},
        {"Player": "Sloan", "Game": "Mavs 816", "Value": 8},
        {"Player": "Azana", "Game": "Vegas Aces", "Value": 5},
        {"Player": "Azana", "Game": "Mavs 816", "Value": 7},
    ])
    assert set(varying_axes(df)) == {"Player", "Game"}
    enc = default_encoding(df)
    assert enc.position == "Game"
    assert enc.color == "Player"
    assert enc.facet is None


def test_two_varying_axes_player_and_metric_larger_cardinality_takes_position():
    # No Game column varying (or present) here -- among Player (2 values)
    # and Metric (3 values), Metric has the larger cardinality.
    rows = []
    for player in ("Sloan", "Azana"):
        for metric in ("Kills", "Aces", "Digs"):
            rows.append({"Player": player, "Metric": metric, "Value": 1})
    df = _df(rows)
    assert set(varying_axes(df)) == {"Player", "Metric"}
    enc = default_encoding(df)
    assert enc.position == "Metric"
    assert enc.color == "Player"
    assert enc.facet is None


def test_two_varying_axes_player_and_metric_player_larger_takes_position():
    # Player has 3 distinct values, Metric only 2 -- Player is larger.
    rows = [
        {"Player": "Sloan", "Metric": "Kills", "Value": 1},
        {"Player": "Azana", "Metric": "Kills", "Value": 1},
        {"Player": "Yuki", "Metric": "Aces", "Value": 1},
    ]
    df = _df(rows)
    assert set(varying_axes(df)) == {"Player", "Metric"}
    enc = default_encoding(df)
    assert enc.position == "Player"
    assert enc.color == "Metric"
    assert enc.facet is None


def test_three_varying_axes_smallest_cardinality_facets_game_takes_position():
    # Metric has the smallest cardinality (2) vs Player (3) and Game (4).
    rows = []
    for player in ("Sloan", "Azana", "Yuki"):
        for game in ("G1", "G2", "G3", "G4"):
            for metric in ("Kills", "Aces"):
                rows.append({"Player": player, "Game": game, "Metric": metric, "Value": 1})
    df = _df(rows)
    assert set(varying_axes(df)) == {"Player", "Game", "Metric"}
    enc = default_encoding(df)
    assert enc.facet == "Metric"
    assert enc.position == "Game"
    assert enc.color == "Player"


def test_three_varying_axes_game_smallest_cardinality_facets_larger_of_rest_takes_position():
    # Game has the smallest cardinality (2) here -- once it facets away,
    # Game is no longer in remaining, so position picks the larger of
    # Player (4) vs Metric (2): Player.
    rows = []
    for player in ("P1", "P2", "P3", "P4"):
        for game in ("G1", "G2"):
            for metric in ("Kills", "Aces"):
                rows.append({"Player": player, "Game": game, "Metric": metric, "Value": 1})
    df = _df(rows)
    enc = default_encoding(df)
    assert enc.facet == "Game"
    assert enc.position == "Player"
    assert enc.color == "Metric"


# ──────────────────────────────────────────────────────────────
# reconcile_encoding -- swap instead of duplicate
# ──────────────────────────────────────────────────────────────

def test_reconcile_swaps_when_new_axis_already_assigned_elsewhere():
    current = EncodingAssignment(position="Game", color="Player")
    updated = reconcile_encoding(current, "color", "Game")
    # "Game" moves into color; whatever was in color ("Player") takes
    # over position instead of "Game" existing in both slots at once.
    assert updated.position == "Player"
    assert updated.color == "Game"


def test_reconcile_swaps_facet_and_position():
    current = EncodingAssignment(position="Metric", color="Player", facet="Game")
    updated = reconcile_encoding(current, "position", "Game")
    assert updated.position == "Game"
    assert updated.facet == "Metric"
    assert updated.color == "Player"  # untouched, wasn't involved in the swap


def test_reconcile_no_swap_when_new_axis_unassigned():
    current = EncodingAssignment(position="Game", color=None)
    updated = reconcile_encoding(current, "color", "Player")
    assert updated.position == "Game"
    assert updated.color == "Player"


def test_reconcile_clearing_a_slot_to_none_does_not_swap():
    current = EncodingAssignment(position="Game", color="Player")
    updated = reconcile_encoding(current, "position", None)
    assert updated.position is None
    assert updated.color == "Player"


def test_reconcile_rejects_unknown_slot():
    import pytest
    current = EncodingAssignment()
    with pytest.raises(ValueError):
        reconcile_encoding(current, "bogus_slot", "Game")


# ──────────────────────────────────────────────────────────────
# set_game_order / slot_options
# ──────────────────────────────────────────────────────────────

def test_set_game_order_toggles_and_validates():
    import pytest
    current = EncodingAssignment(position="Game")
    assert set_game_order(current, "original").game_order == "original"
    assert set_game_order(current, "value").game_order == "value"
    with pytest.raises(ValueError):
        set_game_order(current, "chronological")


def test_slot_options_includes_none_plus_varying_axes():
    df = _df([
        {"Player": "Sloan", "Game": "Vegas Aces", "Value": 12},
        {"Player": "Sloan", "Game": "Mavs 816", "Value": 8},
    ])
    assert slot_options(df) == [None, "Game"]
