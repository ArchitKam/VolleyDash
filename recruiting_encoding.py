"""
recruiting_encoding.py
========================
Decides chart encoding (which axis becomes x-position, color, or panel
split) from which axes still vary in a query result. The default is a
fixed mapping -- Game -> position, Player -> color, Metric -> facet --
applied uniformly to however many of those actually vary (0 through 3),
not per-situation special-casing; the coach can still override any slot.

Pure pandas + dataclasses + plotly.express, no Streamlit or LLM imports
-- same Streamlit-agnostic module pattern already used by
recruiting_llm.py, directly testable on its own.

Operates on the long-format result frame recruiting_operations.py's
module docstring describes: a Player x Game x Metric -> Value cube (see
app.py's run_metric_query/run_category_query/prepare_pipeline_frame for
what actually produces it) -- NOT the wide table tidy_data() pivots for
the table display. Not every axis is always present: a plain single-
metric, single-game query's result has no "Metric" column at all, and a
column that IS present but happens to be constant in this particular
result (e.g. one game was named) doesn't count as "varying" -- see
varying_axes().
"""

from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

AXES = ("Player", "Game", "Metric")
_AXIS_KEY = {"Player": "player", "Game": "game", "Metric": "metric"}
HIGHLIGHT_OUTLINE_COLOR = "#FFFFFF"
HIGHLIGHT_OUTLINE_WIDTH = 4


@dataclass(frozen=True)
class EncodingAssignment:
    position: Optional[str] = None   # one of AXES, or None
    color: Optional[str] = None
    facet: Optional[str] = None
    game_order: str = "value"        # "value" (sort by Value) or "original" (row order)


def varying_axes(df: pd.DataFrame) -> List[str]:
    """Which of Player/Game/Metric are present in df AND have more than
    one distinct (non-null) value in THIS result -- only these are ever
    eligible for a slot. Absent columns and constant-valued ones are
    both excluded, for the same reason: encoding an axis that can't
    actually distinguish any rows would just be a legend/facet of one."""
    return [axis for axis in AXES if axis in df.columns and df[axis].nunique(dropna=True) > 1]


def default_encoding(df: pd.DataFrame) -> EncodingAssignment:
    """
    Fixed default, independent of cardinality: Game -> position, Player
    -> color, Metric -> facet -- whichever of those actually vary in this
    result (see varying_axes); an axis that doesn't vary (absent, or
    constant in this particular result) simply isn't assigned. The coach
    can still override any slot from there via reconcile_encoding/the UI
    dropdowns.
    """
    axes = varying_axes(df)
    return EncodingAssignment(
        position="Game" if "Game" in axes else None,
        color="Player" if "Player" in axes else None,
        facet="Metric" if "Metric" in axes else None,
    )


def slot_options(df: pd.DataFrame) -> List[Optional[str]]:
    """Same option list for every slot's dropdown: 'unassigned' plus
    whichever axes actually vary in this result."""
    return [None] + varying_axes(df)


def reconcile_encoding(current: EncodingAssignment, changed_slot: str,
                        new_axis: Optional[str]) -> EncodingAssignment:
    """
    Applies a manual override to one slot ('position'/'color'/'facet').
    If new_axis already occupies a DIFFERENT slot, swaps it with whatever
    was in changed_slot instead of duplicating an axis across two slots
    -- e.g. moving "Game" into position when position already held
    "Metric" swaps them: position becomes Game, color/facet (wherever
    Game was) becomes Metric. Setting a slot to None never triggers a
    swap (nothing to hand off). game_order is untouched here -- see
    set_game_order.
    """
    if changed_slot not in ("position", "color", "facet"):
        raise ValueError(f"Unknown slot '{changed_slot}' (expected 'position', 'color', or 'facet').")

    slots = {"position": current.position, "color": current.color, "facet": current.facet}
    displaced = slots[changed_slot]

    if new_axis is not None:
        for other_slot, other_axis in slots.items():
            if other_slot != changed_slot and other_axis == new_axis:
                slots[other_slot] = displaced
                break

    slots[changed_slot] = new_axis
    return replace(current, position=slots["position"], color=slots["color"], facet=slots["facet"])


def set_game_order(current: EncodingAssignment, order: str) -> EncodingAssignment:
    """Toggle between sorting the Game axis by Value (today's behavior)
    and preserving original row order -- a cheap stand-in for
    chronological order, since Game has no ordinal in the schema (see
    the trend-support caveat in recruiting_operations.py's module
    docstring)."""
    if order not in ("value", "original"):
        raise ValueError(f"Unknown game_order '{order}' (expected 'value' or 'original').")
    return replace(current, game_order=order)


def _apply_highlight(fig: go.Figure, encoding: EncodingAssignment, highlight: Dict[str, Optional[str]]) -> None:
    """
    Outlines whichever bar(s) match `highlight` (a dict with optional
    'player'/'game'/'metric' values) with a thick, distinct border --
    fill color stays whatever the color axis already assigned (identity:
    "this bar is Sloan's"), only the outline changes (selection: "this
    is the one that was clicked"). An axis this chart doesn't actually
    encode is a wildcard (matches anything), so e.g. a highlight with no
    "player" opinion doesn't prevent a Game-only match from lighting up.
    """
    color_key = _AXIS_KEY.get(encoding.color)
    position_key = _AXIS_KEY.get(encoding.position)
    wanted_color = highlight.get(color_key) if color_key else None
    wanted_position = highlight.get(position_key) if position_key else None

    for trace in fig.data:
        if color_key and wanted_color is not None and trace.name != wanted_color:
            continue  # this whole color-group doesn't match at all
        xs = list(trace.x) if trace.x is not None else []
        if not xs:
            continue
        widths, colors = [], []
        for x_val in xs:
            hit = position_key is None or wanted_position is None or x_val == wanted_position
            widths.append(HIGHLIGHT_OUTLINE_WIDTH if hit else 0)
            colors.append(HIGHLIGHT_OUTLINE_COLOR if hit else "rgba(0,0,0,0)")
        trace.marker.line.width = widths
        trace.marker.line.color = colors


def _one_figure(df: pd.DataFrame, encoding: EncodingAssignment, value_col: str,
                 color_map: Optional[Dict[str, str]] = None,
                 highlight: Optional[Dict[str, Optional[str]]] = None) -> go.Figure:
    # position=None only happens when there was nothing to encode at all
    # (see default_encoding) or a manual override cleared it -- fall back
    # to color as the x-axis so a lone color-only assignment still plots
    # as something rather than erroring on a missing x.
    x = encoding.position or encoding.color
    color = encoding.color if encoding.position else None
    if x is None:
        fig = px.bar(df, y=value_col)
    else:
        # barmode="group" -- side-by-side bars per color (e.g. one bar
        # per player at each game), not Plotly's default stacked/
        # "relative" mode, which reads as one combined total rather than
        # a comparison. color_discrete_map (when given) keeps each
        # category's color CONSISTENT across every panel/question rather
        # than Plotly re-assigning colors per figure.
        fig = px.bar(df, x=x, y=value_col, color=color, barmode="group",
                      color_discrete_map=color_map if (color and color_map) else None)

    if highlight:
        _apply_highlight(fig, encoding, highlight)
    return fig


def resolve_clicked_point(points: List[dict], encoding: EncodingAssignment, fig: go.Figure,
                            panel_metric: Optional[str]) -> Optional[Dict[str, Optional[str]]]:
    """
    Turns one Plotly click-selection point (as reported by
    st.plotly_chart's on_select payload) back into a
    {"player":..., "game":..., "metric":...} dict, using the SAME axis-
    role mapping _apply_highlight uses -- so "click a bar" and "highlight
    a bar" always agree on which axis means what, regardless of which
    slot the coach has each axis assigned to. `panel_metric` fills the
    "metric" key when Metric isn't itself an encoded axis in THIS chart
    (the common case: the panel/action already IS one specific metric).
    Returns None if points is empty or the point doesn't map to a real
    trace (should not happen in practice, but never raises).
    """
    if not points:
        return None
    point = points[0]
    curve_number = point.get("curve_number")
    if curve_number is None or curve_number >= len(fig.data):
        return None

    trace = fig.data[curve_number]
    result: Dict[str, Optional[str]] = {"player": None, "game": None, "metric": panel_metric}

    if encoding.color is not None:
        result[_AXIS_KEY[encoding.color]] = trace.name
    if encoding.position is not None:
        result[_AXIS_KEY[encoding.position]] = point.get("x")

    return result


def render(df: pd.DataFrame, encoding: EncodingAssignment, value_col: str = "Value",
           color_map: Optional[Dict[str, str]] = None,
           highlight: Optional[Dict[str, Optional[str]]] = None) -> List[Tuple[str, go.Figure]]:
    """
    Renders df per encoding into one or more Plotly figures -- one entry
    normally, one per distinct facet value if encoding.facet is set.
    Returns [] if nothing is computable (no non-null values). Knows
    nothing about Streamlit layout -- the caller decides how to arrange
    multiple panels (st.columns, tabs, whatever fits).

    `highlight`, if given, only actually gets applied to the panel whose
    facet value matches highlight["metric"] (when faceted by Metric) --
    a highlight for "Kills Per Set" shouldn't light up a bar in the
    "Aces Per Set" panel just because both happen to share a player/game.
    """
    valid = df[df[value_col].notna()].copy()
    if valid.empty:
        return []

    if encoding.position == "Game" and encoding.game_order == "original":
        game_order_values = list(dict.fromkeys(valid["Game"]))
        valid["Game"] = pd.Categorical(valid["Game"], categories=game_order_values, ordered=True)
        valid = valid.sort_values("Game")

    if encoding.facet is None:
        return [("", _one_figure(valid, encoding, value_col, color_map, highlight))]

    facet_key = _AXIS_KEY.get(encoding.facet)
    panels: List[Tuple[str, go.Figure]] = []
    for facet_value in sorted(valid[encoding.facet].dropna().unique(), key=str):
        sub = valid[valid[encoding.facet] == facet_value]
        panel_highlight = None
        if highlight:
            wanted_facet = highlight.get(facet_key) if facet_key else None
            if wanted_facet is None or wanted_facet == facet_value:
                panel_highlight = highlight
        panels.append((str(facet_value), _one_figure(sub, encoding, value_col, color_map, panel_highlight)))
    return panels
