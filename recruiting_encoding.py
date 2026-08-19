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
from typing import List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

AXES = ("Player", "Game", "Metric")


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


def _one_figure(df: pd.DataFrame, encoding: EncodingAssignment, value_col: str) -> go.Figure:
    # position=None only happens when there was nothing to encode at all
    # (see default_encoding) or a manual override cleared it -- fall back
    # to color as the x-axis so a lone color-only assignment still plots
    # as something rather than erroring on a missing x.
    x = encoding.position or encoding.color
    color = encoding.color if encoding.position else None
    if x is None:
        return px.bar(df, y=value_col)
    return px.bar(df, x=x, y=value_col, color=color)


def render(df: pd.DataFrame, encoding: EncodingAssignment, value_col: str = "Value") -> List[Tuple[str, go.Figure]]:
    """
    Renders df per encoding into one or more Plotly figures -- one entry
    normally, one per distinct facet value if encoding.facet is set.
    Returns [] if nothing is computable (no non-null values). Knows
    nothing about Streamlit layout -- the caller decides how to arrange
    multiple panels (st.columns, tabs, whatever fits).
    """
    valid = df[df[value_col].notna()].copy()
    if valid.empty:
        return []

    if encoding.position == "Game" and encoding.game_order == "original":
        game_order_values = list(dict.fromkeys(valid["Game"]))
        valid["Game"] = pd.Categorical(valid["Game"], categories=game_order_values, ordered=True)
        valid = valid.sort_values("Game")

    if encoding.facet is None:
        return [("", _one_figure(valid, encoding, value_col))]

    panels: List[Tuple[str, go.Figure]] = []
    for facet_value in sorted(valid[encoding.facet].dropna().unique(), key=str):
        sub = valid[valid[encoding.facet] == facet_value]
        panels.append((str(facet_value), _one_figure(sub, encoding, value_col)))
    return panels
