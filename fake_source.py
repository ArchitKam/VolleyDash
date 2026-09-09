"""
fake_source.py
===============
A hand-built EVENT-grain Source for tests, so the spec/evaluator/query
suites run with no .dvw file present and with data small enough that
every expected number can be worked out by hand in the test itself.

Deliberately mirrors DvwSource's shape (same identity columns, same
skill/evaluation vocabulary structure, same sets-played measure) rather
than being a loose stub -- a fake that does not match the real adapter's
contract would let a broken evaluator pass.
"""

from typing import Dict, List, Optional

import pandas as pd

from source import FieldRole, FieldSpec, Grain, Source, SourceSchema

PLAYER_COLUMN = "player_label"
GAME_COLUMN = "match_label"


class FakeEventSource(Source):
    """
    Build from a list of (player, game, skill, evaluation_code) tuples
    plus a {(player, game): sets_played} mapping.
    """

    def __init__(self, rows: List[tuple], sets_played: Optional[Dict[tuple, int]] = None):
        self._frame = pd.DataFrame(
            rows, columns=[PLAYER_COLUMN, GAME_COLUMN, "skill", "evaluation_code"]
        )
        self._sets_played = sets_played or {}

    @property
    def grain(self) -> Grain:
        return Grain.EVENT

    def identity_fields(self) -> Dict[str, str]:
        return {"Player": PLAYER_COLUMN, "Game": GAME_COLUMN}

    def facts(self) -> pd.DataFrame:
        return self._frame

    def measures(self) -> pd.DataFrame:
        if not self._sets_played:
            return pd.DataFrame()
        return pd.DataFrame(
            [{PLAYER_COLUMN: player, GAME_COLUMN: game, "sets_played": value}
             for (player, game), value in self._sets_played.items()]
        )

    def matches(self):
        return sorted(self._frame[GAME_COLUMN].unique())

    @property
    def schema(self) -> SourceSchema:
        frame = self._frame
        fields = {
            PLAYER_COLUMN: FieldSpec(PLAYER_COLUMN, FieldRole.IDENTITY, "Player"),
            GAME_COLUMN: FieldSpec(GAME_COLUMN, FieldRole.IDENTITY, "Match"),
        }
        for column in ("skill", "evaluation_code"):
            values = frozenset(str(v) for v in frame[column].dropna().unique())
            fields[column] = FieldSpec(column, FieldRole.DIMENSION, column, values=values)

        vocabulary = {
            str(skill): frozenset(str(c) for c in group["evaluation_code"].dropna().unique())
            for skill, group in frame.dropna(subset=["skill"]).groupby("skill")
        }
        return SourceSchema(fields=fields,
                             dependent_values={("skill", "evaluation_code"): vocabulary})


def sample_source() -> FakeEventSource:
    """
    Two players, two matches, hand-countable.

    Sloan  @ Game A: 3 Attack # (kills), 1 Attack = (error), 1 Attack + -> 5 attacks
    Sloan  @ Game B: 1 Attack #, 1 Attack =                              -> 2 attacks
    Kendal @ Game A: 2 Attack #, 2 Serve # (aces)                        -> 2 attacks
    Kendal @ Game B: no attacks at all (only a reception)
    """
    rows = [
        ("Sloan", "Game A", "Attack", "#"),
        ("Sloan", "Game A", "Attack", "#"),
        ("Sloan", "Game A", "Attack", "#"),
        ("Sloan", "Game A", "Attack", "="),
        ("Sloan", "Game A", "Attack", "+"),
        ("Sloan", "Game B", "Attack", "#"),
        ("Sloan", "Game B", "Attack", "="),
        ("Kendal", "Game A", "Attack", "#"),
        ("Kendal", "Game A", "Attack", "#"),
        ("Kendal", "Game A", "Serve", "#"),
        ("Kendal", "Game A", "Serve", "#"),
        ("Kendal", "Game B", "Reception", "#"),
    ]
    sets_played = {
        ("Sloan", "Game A"): 3, ("Sloan", "Game B"): 2,
        ("Kendal", "Game A"): 3, ("Kendal", "Game B"): 4,
    }
    return FakeEventSource(rows, sets_played)


SET_COLUMN = "set_label"


class FakeSetSource(FakeEventSource):
    """
    A fake with the Set axis, so set scoping is unit-testable without the
    real corpus.

    Rows are (player, game, set, skill, evaluation_code), and sets played
    is derived rather than passed in: at this grain it is exactly "one
    row per (player, match, set) the player appeared in", which is the
    same rule the real DVW source applies -- and deriving it here means
    the fake cannot drift into asserting a shape the real source does not
    produce.
    """

    def __init__(self, rows: List[tuple]):
        self._frame = pd.DataFrame(
            rows, columns=[PLAYER_COLUMN, GAME_COLUMN, SET_COLUMN, "skill", "evaluation_code"]
        )
        self._sets_played = {}

    def identity_fields(self) -> Dict[str, str]:
        return {"Player": PLAYER_COLUMN, "Game": GAME_COLUMN, "Set": SET_COLUMN}

    def measures(self) -> pd.DataFrame:
        appearances = self._frame[[PLAYER_COLUMN, GAME_COLUMN, SET_COLUMN]].drop_duplicates()
        appearances = appearances.copy()
        appearances["sets_played"] = 1
        return appearances

    @property
    def schema(self) -> SourceSchema:
        base = super().schema
        fields = dict(base.fields)
        fields[SET_COLUMN] = FieldSpec(SET_COLUMN, FieldRole.IDENTITY, "Set")
        return SourceSchema(fields=fields, dependent_values=base.dependent_values)


def sample_set_source() -> FakeSetSource:
    """
    One match, three sets, hand-countable per set.

    Sloan  -- Set 1: 2 kills   Set 2: 1 kill    Set 3: 3 kills   (6 total, 3 sets)
    Kendal -- Set 1: 1 kill    Set 3: 0 kills   (1 total, 2 sets; did not play set 2)
    """
    rows = [
        ("Sloan", "Game A", "Set 1", "Attack", "#"),
        ("Sloan", "Game A", "Set 1", "Attack", "#"),
        ("Sloan", "Game A", "Set 2", "Attack", "#"),
        ("Sloan", "Game A", "Set 3", "Attack", "#"),
        ("Sloan", "Game A", "Set 3", "Attack", "#"),
        ("Sloan", "Game A", "Set 3", "Attack", "#"),
        ("Kendal", "Game A", "Set 1", "Attack", "#"),
        ("Kendal", "Game A", "Set 3", "Attack", "="),
    ]
    return FakeSetSource(rows)
