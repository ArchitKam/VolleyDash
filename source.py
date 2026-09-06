"""
source.py
=================
The Source boundary: what the rest of the system is allowed to assume
about an input file, so nothing above this line knows whether it is
reading a pre-aggregated Huddle CSV or a per-action DataVolley .dvw.

A Source exposes three things:
  facts()   -- a DataFrame of rows
  schema    -- what fields exist and what values they may take
  grain     -- what ONE ROW represents

Grain is the load-bearing concept, because it decides which metric
kinds are even expressible:

  MEASURE grain (Huddle CSV): one row per player per match, and the
    fields are already-computed measures ("Attack K" is a kill COUNT).
    A metric is either a direct column reference or arithmetic over
    columns -- there is no finer detail to filter on, because the
    export already threw it away.

  EVENT grain (DataVolley .dvw): one row per action. The fields are
    attributes of a single event (which skill, which evaluation code,
    which zone), so a metric is a FILTER plus an AGGREGATE -- "count
    the rows where skill=Attack and evaluation_code=#". Counting is
    how a measure gets constructed here rather than read off.

"formula" is expressible at BOTH grains, and that is the whole point of
putting the boundary here: a formula composes over other METRICS, not
over raw fields, so "([Kills] - [Attack Errors]) / [Sets Played]" is
the same expression in both worlds. Only the resolution of its leaf
tokens differs, and that is the evaluator's job (evaluate.py),
not the formula's.

Schema note -- vocabularies are DERIVED, NOT DECLARED. The permitted
values of a field come from the data actually loaded, not from a table
copied out of a format specification. A validator that rejects a
(skill, evaluation_code) pair therefore rejects it because that pair is
not present in these matches -- a true, checkable statement -- rather
than because a document said it should not exist. The tradeoff is
recorded honestly in field_values()'s docstring.

Pure types and protocol only: no pandas logic, no I/O, no Streamlit.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, FrozenSet, List, Optional

import pandas as pd


class Grain(Enum):
    """What a single row of facts() represents."""

    MEASURE = "measure"   # one row per player per match; fields are computed measures
    EVENT = "event"       # one row per action; fields are attributes of that action


class FieldRole(Enum):
    """
    Why a field exists, which is what tells the evaluator how it may be
    used -- not its dtype, which cannot distinguish "the player this row
    belongs to" from "the zone the ball landed in" when both are
    strings.
    """

    IDENTITY = "identity"      # who/which match this row belongs to (Player, Game)
    DIMENSION = "dimension"    # a categorical attribute you may filter on
    MEASURE = "measure"        # a numeric quantity you may aggregate


@dataclass(frozen=True)
class FieldSpec:
    """
    One field a Source offers, plus the values it may take.

    `values` is None for a field whose domain is open (a free numeric
    measure, or an identity column). For a closed categorical dimension
    it is the observed domain, and a validator may reject anything
    outside it -- see field_values().
    """

    name: str
    role: FieldRole
    description: str = ""
    values: Optional[FrozenSet[str]] = None

    def permits(self, value: str) -> bool:
        return self.values is None or value in self.values


@dataclass(frozen=True)
class SourceSchema:
    """
    The full description of what a Source offers, and the object every
    validator runs against. This is the abstract replacement for
    RECRUITING_COLUMN_SCHEMA being a module-level constant in
    recruiting_tree.py: same job (say what is real), but obtained from
    the source at hand instead of hard-coded for one file format.

    `dependent_values` carries the case a flat field -> values mapping
    cannot express: a vocabulary where the legal values of one field
    depend on the value of another. In DataVolley that is exactly the
    evaluation code, whose meaning AND legality depend on the skill it
    attaches to -- "#" is an ace on a Serve, a perfect pass on a
    Reception and a kill on an Attack, and is not a legal code at all
    for some skills. Keyed by (governing_field, dependent_field) ->
    {governing_value: allowed dependent values}.
    """

    fields: Dict[str, FieldSpec] = field(default_factory=dict)
    dependent_values: Dict[tuple, Dict[str, FrozenSet[str]]] = field(default_factory=dict)

    def has_field(self, name: str) -> bool:
        return name in self.fields

    def field_names(self) -> List[str]:
        return sorted(self.fields)

    def field_values(self, name: str) -> Optional[FrozenSet[str]]:
        """
        The values this field is known to take. DERIVED from the loaded
        data, so it answers "what is in these matches", not "what the
        format permits in principle" -- a legal-but-never-scouted value
        will be absent and will be rejected by validation. That is the
        deliberate tradeoff: a metric is accepted only when the data can
        actually compute it, and the error message says which values
        were really seen instead of pointing at a spec.
        """
        spec = self.fields.get(name)
        return spec.values if spec else None

    def dependent_field_values(
        self, governing_field: str, dependent_field: str, governing_value: str,
    ) -> Optional[FrozenSet[str]]:
        """Allowed values of `dependent_field` GIVEN that
        `governing_field` == `governing_value`; None when no such
        dependency is declared (then field_values applies)."""
        table = self.dependent_values.get((governing_field, dependent_field))
        if table is None:
            return None
        return table.get(governing_value)


class Source(ABC):
    """
    One input format, adapted. Implementations live in
    source_csv.py / source_dvw.py.

    Deliberately narrow: everything above this line consumes facts() and
    schema and never opens a file, so swapping the format -- or adding a
    third one -- touches exactly one class.
    """

    @property
    @abstractmethod
    def grain(self) -> Grain:
        """What one row of facts() represents."""

    @property
    @abstractmethod
    def schema(self) -> SourceSchema:
        """Fields available, and the values they may take."""

    @abstractmethod
    def facts(self) -> pd.DataFrame:
        """
        The rows themselves, at this source's grain. Must carry the
        identity columns named by identity_fields() so the evaluator can
        group without knowing the format.
        """

    @abstractmethod
    def identity_fields(self) -> Dict[str, str]:
        """
        Maps the cube's axis names to this source's own column names,
        e.g. {"Player": "player_label", "Game": "match_label"}. The
        evaluator groups by these, which is what lets one evaluator
        serve both grains.
        """

    def measures(self) -> pd.DataFrame:
        """
        Per-(Player, Game) quantities that are NOT derivable by counting
        facts() rows -- for DataVolley, sets played, which is recorded in
        the match's roster metadata and cannot be reconstructed from
        action rows without undercounting anyone who was on court but
        never touched the ball.

        Default: no such measures. Shape when present is the cube's own:
        the identity columns plus one column per measure.
        """
        return pd.DataFrame()

    def game_labels(self) -> List[str]:
        """
        Every Game on the cube's Game axis, in the order they should be
        offered and plotted.

        Part of the contract rather than something callers dig out of an
        adapter-specific attribute: the router needs the list of game
        names, and reading it off a concrete adapter's own match objects
        would tie the query layer to one implementation. The default
        derives it from facts(); an adapter that knows a better ORDER
        (chronological, say, rather than alphabetical) overrides it.
        """
        facts = self.facts()
        game_column = self.identity_fields()["Game"]
        if facts.empty or game_column not in facts.columns:
            return []
        return sorted(facts[game_column].dropna().unique())
