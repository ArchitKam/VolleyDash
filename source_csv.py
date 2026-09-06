"""
source_csv.py
==============
The Huddle CSV export, adapted to the Source contract.

MEASURE grain: one row is a player's totals for one match, and the
fields are already-computed measures ("Attack K" is a kill COUNT, not a
kill). The export threw the individual actions away before the file was
written, which is why:

  * there is no Set axis here -- identity is (Player, Game) and nothing
    finer exists to group by. axes() reports that, so the unified UI
    hides the set controls for this source rather than offering a
    drill-down that could only ever return match totals; and

  * a metric is a COLUMN reference or arithmetic over columns, never a
    filter-plus-aggregate. Counting is not available because there is
    nothing left to count.

This class is deliberately thin. It adapts what the CSV already is into
the shape the rest of the system consumes; the arithmetic that turns a
spec into numbers stays in evaluate_csv.py, unchanged from when it lived
in app.py, so that unification did not quietly restate what a blank cell
or a division by zero means to a coach.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from evaluate_csv import game_label as _game_label
from source import FieldRole, FieldSpec, Grain, Source, SourceSchema

PLAYER_COLUMN = "Player"
GAME_COLUMN = "Game"

#: Huddle marks player rows with a leading "#<jersey>"; everything else
#: in the export is a team aggregate or a spacer. The CSV evaluator has
#: always identified players this way, and the identity must not drift
#: between the evaluator and the adapter that feeds it.
_PLAYER_ROW_PREFIX = "#"

NAME_COLUMN = "Name"


def is_player_row(value: object) -> bool:
    return str(value).strip().startswith(_PLAYER_ROW_PREFIX)


class CsvSource(Source):
    """
    One or more Huddle match exports as a single Source.

    Constructed from already-loaded frames rather than paths: fetching a
    CSV is the data store's job (it may come from a private GitHub repo
    or from disk), and keeping that out of here is what lets the same
    adapter serve both without knowing which happened.
    """

    def __init__(self, games: Sequence[Tuple[object, pd.DataFrame]],
                 name_column: str = NAME_COLUMN):
        #: [(GameInfo, frame)] -- GameInfo is duck-typed to .opponent so
        #: this module does not depend on the data store's dataclass.
        self._games = list(games)
        self._name_column = name_column
        self._facts: Optional[pd.DataFrame] = None
        self._schema: Optional[SourceSchema] = None

    # ── contract ──────────────────────────────────────────────

    @property
    def grain(self) -> Grain:
        return Grain.MEASURE

    @property
    def schema(self) -> SourceSchema:
        if self._schema is None:
            self._schema = self._build_schema(self.facts())
        return self._schema

    def identity_fields(self) -> Dict[str, str]:
        """No Set: see the module docstring. This is the single place the
        absence is declared, and axes() derives the rest."""
        return {"Player": PLAYER_COLUMN, "Game": GAME_COLUMN}

    def facts(self) -> pd.DataFrame:
        if self._facts is None:
            self._facts = self._build_facts()
        return self._facts

    def measures(self) -> pd.DataFrame:
        """At this grain facts ARE the measures: every numeric column is
        already a per-(player, match) quantity, so there is nothing to
        derive separately the way sets played must be for .dvw."""
        return self.facts()

    def game_labels(self) -> List[str]:
        """Export order, not alphabetical -- the games are loaded in the
        order the coach selected them, and re-sorting would silently
        reorder every chart's Game axis."""
        seen, labels = set(), []
        for game, _ in self._games:
            label = _game_label(game)
            if label not in seen:
                seen.add(label)
                labels.append(label)
        return labels

    def player_labels(self) -> List[str]:
        facts = self.facts()
        if facts.empty:
            return []
        return sorted(facts[PLAYER_COLUMN].dropna().unique())

    # ── construction ──────────────────────────────────────────

    def frames(self) -> List[Tuple[object, pd.DataFrame]]:
        """The underlying (GameInfo, frame) pairs, for the MEASURE-grain
        evaluator, which works a row at a time against one match's frame
        rather than against the stacked facts table."""
        return list(self._games)

    def _build_facts(self) -> pd.DataFrame:
        frames = []
        for game, frame in self._games:
            if frame is None or frame.empty or self._name_column not in frame.columns:
                continue
            players = frame[frame[self._name_column].map(is_player_row)].copy()
            if players.empty:
                continue
            players[PLAYER_COLUMN] = players[self._name_column].astype(str).str.strip()
            players[GAME_COLUMN] = _game_label(game)
            frames.append(players)

        if not frames:
            return pd.DataFrame(columns=[PLAYER_COLUMN, GAME_COLUMN])
        # sort=False: a column present in only some exports must not
        # reorder the ones shared by all.
        return pd.concat(frames, ignore_index=True, sort=False)

    def _build_schema(self, facts: pd.DataFrame) -> SourceSchema:
        """
        Every numeric export column is a measure; the identity columns
        are identities. Nothing is a filterable DIMENSION, because a
        measure-grain row has no attributes to filter on -- which is the
        schema-level statement of why event metrics are not expressible
        here, and why event_spec's validator rejects them for this
        source rather than accepting one that would always return zero.
        """
        fields: Dict[str, FieldSpec] = {
            PLAYER_COLUMN: FieldSpec(PLAYER_COLUMN, FieldRole.IDENTITY, "Player (jersey + name)"),
            GAME_COLUMN: FieldSpec(GAME_COLUMN, FieldRole.IDENTITY, "Match (opponent)"),
        }
        for column in facts.columns:
            if column in fields or column == self._name_column:
                continue
            if pd.api.types.is_numeric_dtype(facts[column]):
                fields[column] = FieldSpec(column, FieldRole.MEASURE, column)
        return SourceSchema(fields=fields)


