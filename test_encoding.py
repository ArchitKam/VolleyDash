import pandas as pd

from recruiting_encoding import (
    EncodingAssignment, default_encoding, reconcile_encoding, render,
    resolve_clicked_point, set_game_order, slot_options, varying_axes,
)


def _df(rows):
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────
# default_encoding -- fixed mapping (Game -> position, Player -> color,
# Metric -> facet), applied uniformly regardless of cardinality; an axis
# that doesn't vary in this particular result just isn't assigned.
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
    # column at all -- Player is the only varying axis here. Game isn't
    # even present, so it can't take position -- Player has nowhere to
    # go but color, even alone.
    df = _df([
        {"Player": "Sloan", "Value": 12},
        {"Player": "Azana", "Value": 9},
    ])
    assert varying_axes(df) == ["Player"]
    enc = default_encoding(df)
    assert enc.position is None
    assert enc.color == "Player"
    assert enc.facet is None


def test_one_varying_axis_metric_only():
    df = _df([
        {"Player": "Sloan", "Metric": "Kills", "Value": 12},
        {"Player": "Sloan", "Metric": "Aces", "Value": 2},
    ])
    assert varying_axes(df) == ["Metric"]
    enc = default_encoding(df)
    assert enc.position is None
    assert enc.color is None
    assert enc.facet == "Metric"


def test_two_varying_axes_player_and_game_fixed_roles():
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


def test_two_varying_axes_player_and_metric_no_game_present():
    # Game isn't even a column here -- Metric still goes to facet, Player
    # still goes to color, regardless of either's cardinality.
    rows = []
    for player in ("Sloan", "Azana"):
        for metric in ("Kills", "Aces", "Digs"):
            rows.append({"Player": player, "Metric": metric, "Value": 1})
    df = _df(rows)
    assert set(varying_axes(df)) == {"Player", "Metric"}
    enc = default_encoding(df)
    assert enc.position is None
    assert enc.color == "Player"
    assert enc.facet == "Metric"


def test_three_varying_axes_always_the_same_fixed_roles():
    rows = []
    for player in ("Sloan", "Azana", "Yuki"):
        for game in ("G1", "G2", "G3", "G4"):
            for metric in ("Kills", "Aces"):
                rows.append({"Player": player, "Game": game, "Metric": metric, "Value": 1})
    df = _df(rows)
    assert set(varying_axes(df)) == {"Player", "Game", "Metric"}
    enc = default_encoding(df)
    assert enc.position == "Game"
    assert enc.color == "Player"
    assert enc.facet == "Metric"


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


# ──────────────────────────────────────────────────────────────
# render() -- color_map (consistent colors) and highlights (brushing,
# one or more simultaneous selections each in its own color)
# ──────────────────────────────────────────────────────────────

def _comparison_df():
    return _df([
        {"Player": "Sloan", "Game": "Vegas Aces", "Value": 12},
        {"Player": "Sloan", "Game": "Mavs 816", "Value": 8},
        {"Player": "Kendal", "Game": "Vegas Aces", "Value": 5},
        {"Player": "Kendal", "Game": "Mavs 816", "Value": 7},
    ])


def test_render_applies_color_map_per_trace():
    df = _comparison_df()
    enc = EncodingAssignment(position="Game", color="Player")
    color_map = {"Sloan": "#E21833", "Kendal": "#B8860B"}
    _, fig = render(df, enc, color_map=color_map)[0]
    by_name = {trace.name: trace.marker.color for trace in fig.data}
    assert by_name == {"Sloan": "#E21833", "Kendal": "#B8860B"}


def test_render_highlight_outlines_only_the_matching_bar():
    df = _comparison_df()
    enc = EncodingAssignment(position="Game", color="Player")
    highlights = [("#1", "#00BFFF", {"player": "Sloan", "game": "Vegas Aces", "metric": None})]
    _, fig = render(df, enc, highlights=highlights)[0]

    sloan_trace = next(t for t in fig.data if t.name == "Sloan")
    kendal_trace = next(t for t in fig.data if t.name == "Kendal")

    # Sloan's trace: Vegas Aces (index 0) gets the outline, Mavs 816 doesn't.
    assert list(sloan_trace.x) == ["Vegas Aces", "Mavs 816"]
    assert sloan_trace.marker.line.width == (4, 0)
    assert sloan_trace.marker.line.color == ("#00BFFF", "rgba(0,0,0,0)")

    # Kendal's trace never matched the highlighted player at all --
    # untouched (None), not even a zero-width array.
    assert kendal_trace.marker.line.width is None


def test_render_highlight_only_applies_to_the_matching_facet_panel():
    rows = []
    for player in ("Sloan", "Kendal"):
        for game in ("Vegas Aces", "Mavs 816"):
            for metric in ("Kills Per Set", "Aces Per Set"):
                rows.append({"Player": player, "Game": game, "Metric": metric, "Value": 1})
    df = _df(rows)
    enc = EncodingAssignment(position="Game", color="Player", facet="Metric")
    highlights = [("#1", "#00BFFF", {"player": "Sloan", "game": "Vegas Aces", "metric": "Kills Per Set"})]

    panels = render(df, enc, highlights=highlights)
    by_title = dict(panels)
    assert set(by_title) == {"Aces Per Set", "Kills Per Set"}

    kills_sloan_trace = next(t for t in by_title["Kills Per Set"].data if t.name == "Sloan")
    assert kills_sloan_trace.marker.line.width == (4, 0)

    # The OTHER metric's panel must not light up just because it shares
    # the same player/game -- the highlight is scoped to its own metric.
    for trace in by_title["Aces Per Set"].data:
        assert trace.marker.line.width is None


def test_render_position_axis_defaults_to_value_sorted_order():
    # "value" (the dataclass default) sorts EVERY position axis by Value
    # descending, not just Game -- so a Rank'd pipeline result's order
    # carries straight into the chart without an extra step.
    df = _df([
        {"Player": "Sloan", "Value": 5},
        {"Player": "Kendal", "Value": 12},
        {"Player": "Azana", "Value": 8},
    ])
    enc = EncodingAssignment(position="Player")
    _, fig = render(df, enc)[0]
    assert list(fig.data[0].x) == ["Kendal", "Azana", "Sloan"]


def test_render_position_axis_original_order_preserves_row_order():
    df = _df([
        {"Player": "Sloan", "Value": 5},
        {"Player": "Kendal", "Value": 12},
        {"Player": "Azana", "Value": 8},
    ])
    enc = EncodingAssignment(position="Player", game_order="original")
    _, fig = render(df, enc)[0]
    assert list(fig.data[0].x) == ["Sloan", "Kendal", "Azana"]


def test_render_highlight_stamps_badge_annotation_on_the_matching_bar():
    df = _comparison_df()
    enc = EncodingAssignment(position="Game", color="Player", game_order="original")
    highlights = [("#1", "#00BFFF", {"player": "Sloan", "game": "Vegas Aces", "metric": None})]
    _, fig = render(df, enc, highlights=highlights)[0]

    assert len(fig.layout.annotations) == 1
    annotation = fig.layout.annotations[0]
    assert annotation.text == "#1"
    assert annotation.x == "Vegas Aces"
    assert annotation.font.color == "#00BFFF"


def test_render_no_highlight_leaves_figures_plain():
    df = _comparison_df()
    enc = EncodingAssignment(position="Game", color="Player")
    _, fig = render(df, enc)[0]
    for trace in fig.data:
        assert trace.marker.line.width is None


def test_render_multiple_highlights_each_get_their_own_color():
    df = _comparison_df()
    # game_order="original" -- pin row order explicitly so this test's own
    # index assumptions (Vegas Aces=0, Mavs 816=1) hold regardless of the
    # "value" order default (see test_render_position_axis_defaults_to_
    # value_sorted_order below, which covers that default specifically).
    enc = EncodingAssignment(position="Game", color="Player", game_order="original")
    highlights = [
        ("#1", "#00BFFF", {"player": "Sloan", "game": "Vegas Aces", "metric": None}),
        ("#2", "#39FF14", {"player": "Kendal", "game": "Mavs 816", "metric": None}),
    ]
    _, fig = render(df, enc, highlights=highlights)[0]

    sloan_trace = next(t for t in fig.data if t.name == "Sloan")
    kendal_trace = next(t for t in fig.data if t.name == "Kendal")

    # Sloan: Vegas Aces (index 0) lit up in the FIRST selection's color.
    assert sloan_trace.marker.line.width == (4, 0)
    assert sloan_trace.marker.line.color == ("#00BFFF", "rgba(0,0,0,0)")

    # Kendal: Mavs 816 (index 1) lit up in the SECOND selection's color --
    # both selections apply simultaneously, each to its own bar.
    assert kendal_trace.marker.line.width == (0, 4)
    assert kendal_trace.marker.line.color == ("rgba(0,0,0,0)", "#39FF14")


def test_render_two_highlights_on_the_same_bar_the_later_one_wins():
    df = _comparison_df()
    enc = EncodingAssignment(position="Game", color="Player")
    highlights = [
        ("#1", "#00BFFF", {"player": "Sloan", "game": "Vegas Aces", "metric": None}),
        ("#2", "#39FF14", {"player": "Sloan", "game": "Vegas Aces", "metric": None}),
    ]
    _, fig = render(df, enc, highlights=highlights)[0]
    sloan_trace = next(t for t in fig.data if t.name == "Sloan")
    assert sloan_trace.marker.line.color == ("#39FF14", "rgba(0,0,0,0)")


# ──────────────────────────────────────────────────────────────
# resolve_clicked_point -- click a bar, get back {player, game, metric}
# ──────────────────────────────────────────────────────────────

def test_resolve_clicked_point_maps_color_and_position_to_correct_keys():
    df = _comparison_df()
    enc = EncodingAssignment(position="Game", color="Player")
    _, fig = render(df, enc)[0]

    # Simulate clicking the Sloan trace's "Vegas Aces" bar.
    curve_number = next(i for i, t in enumerate(fig.data) if t.name == "Sloan")
    points = [{"curve_number": curve_number, "x": "Vegas Aces", "y": 12}]

    result = resolve_clicked_point(points, enc, fig, panel_metric="Kills Per Set")
    assert result == {"player": "Sloan", "game": "Vegas Aces", "metric": "Kills Per Set"}


def test_resolve_clicked_point_returns_none_for_empty_points():
    df = _comparison_df()
    enc = EncodingAssignment(position="Game", color="Player")
    _, fig = render(df, enc)[0]
    assert resolve_clicked_point([], enc, fig, panel_metric="Kills Per Set") is None


def test_resolve_clicked_point_uses_panel_metric_when_metric_not_an_axis():
    # Neither position nor color is "Metric" here -- the panel's own
    # metric (passed in by the caller, since this action/panel already
    # IS one specific metric) fills that key instead of being left None.
    df = _comparison_df()
    enc = EncodingAssignment(position="Game", color="Player")
    _, fig = render(df, enc)[0]
    curve_number = next(i for i, t in enumerate(fig.data) if t.name == "Kendal")
    points = [{"curve_number": curve_number, "x": "Mavs 816", "y": 7}]
    result = resolve_clicked_point(points, enc, fig, panel_metric="Aces Per Set")
    assert result["metric"] == "Aces Per Set"


# ── chronological Game axis ────────────────────────────────────

class TestGameAxisOrder:
    """A season on an x-axis reads in the order it was played. Sorting
    games by value turns a timeline into a ranking, and you cannot see a
    run of form in a bar chart sorted by height."""

    @staticmethod
    def _frame():
        import pandas as pd
        return pd.DataFrame([
            {"Game": "Ohio State (Nov 8)", "Player": "A", "Value": 5.0},
            {"Game": "Rutgers (Oct 3)", "Player": "A", "Value": 1.0},
            {"Game": "Illinois (Oct 10)", "Player": "A", "Value": 9.0},
        ])

    def test_a_game_axis_defaults_to_natural_order_not_value(self):
        from recruiting_encoding import default_encoding

        encoding = default_encoding(self._frame())
        assert encoding.position == "Game"
        assert encoding.game_order == "original"

    def test_the_canonical_order_wins_over_the_frames_row_order(self):
        """The evaluator emits rows sorted by IDENTITY, so "original"
        without a canonical order was alphabetical -- which put Nov 8
        before Oct 3 and looked deliberate."""
        from recruiting_encoding import default_encoding, render

        frame = self._frame()
        season = ["Rutgers (Oct 3)", "Illinois (Oct 10)", "Ohio State (Nov 8)"]
        panels = render(frame, default_encoding(frame), position_order=season)
        assert panels
        drawn = list(panels[0][1].data[0].x)
        assert drawn == season, drawn

    def test_a_game_missing_from_the_canonical_order_is_still_drawn(self):
        """An unknown game must not silently vanish from the chart."""
        from recruiting_encoding import default_encoding, render

        frame = self._frame()
        panels = render(frame, default_encoding(frame),
                        position_order=["Rutgers (Oct 3)"])
        drawn = list(panels[0][1].data[0].x)
        assert drawn[0] == "Rutgers (Oct 3)"
        assert set(drawn) == set(frame["Game"]), "every game must appear"

    def test_by_value_still_available(self):
        from recruiting_encoding import default_encoding, render, set_game_order

        frame = self._frame()
        encoding = set_game_order(default_encoding(frame), "value")
        panels = render(frame, encoding)
        assert list(panels[0][1].data[0].y) == [9.0, 5.0, 1.0]


# ── the fixed encoding rule ────────────────────────────────────

class TestFixedEncodingDefaults:
    """Player -> colour, Game -> x, Metric -> panel. Absolute, not
    cardinality-dependent."""

    @staticmethod
    def _df(rows):
        import pandas as pd
        return pd.DataFrame(rows)

    def test_the_rule_holds_when_all_three_vary(self):
        from recruiting_encoding import default_encoding

        e = default_encoding(self._df([
            {"Metric": "Kills", "Game": "A", "Player": "P1", "Value": 1},
            {"Metric": "Aces", "Game": "B", "Player": "P2", "Value": 2},
        ]))
        assert (e.position, e.color, e.facet) == ("Game", "Player", "Metric")

    def test_an_axis_that_cannot_distinguish_anything_is_left_unassigned(self):
        from recruiting_encoding import default_encoding

        e = default_encoding(self._df([
            {"Metric": "Kills", "Game": "A", "Player": "P1", "Value": 1},
            {"Metric": "Kills", "Game": "B", "Player": "P2", "Value": 2},
        ]))
        assert (e.position, e.color) == ("Game", "Player")
        assert e.facet is None, "one metric is a panel of one"

    def test_a_single_game_broken_out_by_set_puts_set_on_x(self):
        """Set was invisible to the chart layer entirely -- AXES did not
        list it -- so a per-set drill-down drew unlabelled bars at
        x = 0, 1, 2 and lost which set was which."""
        from recruiting_encoding import default_encoding, render

        df = self._df([
            {"Metric": "Kills", "Game": "Purdue", "Set": f"Set {i}", "Player": "P1", "Value": v}
            for i, v in enumerate([2, 1, 3], start=1)
        ])
        e = default_encoding(df)
        assert e.position == "Set"
        assert list(render(df, e)[0][1].data[0].x) == ["Set 1", "Set 2", "Set 3"]

    def test_sets_are_ordered_1_2_3_not_tallest_first(self):
        from recruiting_encoding import default_encoding

        df = self._df([
            {"Metric": "Kills", "Game": "P", "Set": "Set 1", "Player": "A", "Value": 9},
            {"Metric": "Kills", "Game": "P", "Set": "Set 2", "Player": "A", "Value": 1},
        ])
        assert default_encoding(df).game_order == "original"

    def test_an_axis_that_varies_but_is_drawn_nowhere_is_reported(self):
        """Rows differing only on an undrawn axis land on the same mark:
        several values where the reader sees one."""
        from recruiting_encoding import default_encoding, unassigned_axes

        df = self._df([
            {"Metric": "Kills", "Game": g, "Set": s, "Player": "P1", "Value": 1}
            for g in ("A", "B") for s in ("Set 1", "Set 2")
        ])
        assert unassigned_axes(df, default_encoding(df)) == ["Set"]

    def test_nothing_is_reported_when_every_varying_axis_is_drawn(self):
        from recruiting_encoding import default_encoding, unassigned_axes

        df = self._df([
            {"Metric": "Kills", "Game": "A", "Player": "P1", "Value": 1},
            {"Metric": "Aces", "Game": "B", "Player": "P2", "Value": 2},
        ])
        assert unassigned_axes(df, default_encoding(df)) == []
