"""
source_dvw.py
=====================
EVENT-grain Source over DataVolley .dvw files, via pydatavolley (with
dvw_patch.py's corrections applied -- without them 4 of the 22 real
match files in this project do not parse at all).

One row of facts() is ONE ACTION by one player: a serve, a reception, an
attack, a block, a dig, a set, a freeball. That is a fundamentally finer
grain than the Huddle CSV export, where a row is already a whole match's
totals for one player, and it is why an "event" metric is a filter plus
an aggregate rather than a column reference.

────────────────────────────────────────────────────────────────
DECISIONS TAKEN FROM THE DATA, NOT FROM THE FORMAT DOCS
Each of these was measured across all 22 match files in this project
before being written down; the numbers are in the comments so a later
reader can re-check rather than trust.
────────────────────────────────────────────────────────────────

PLAYER IDENTITY INCLUDES THE JERSEY NUMBER.
  One roster here carries the SAME name at BOTH #17 and #44 in the same
  match. Keying a player by name alone silently merges two different
  roster entries (one of whom played 5 sets and one 0), so the label is
  "#44 <name>". This also matches the CSV world's "#7 <name>"
  convention, so the router's player resolution (which strips a leading
  #token) works unchanged against both. Real names are data and live
  in the match files, which stay out of this repo.

GAME IDENTITY INCLUDES THE DATE.
  Maryland plays Penn State, Indiana and Rutgers TWICE each in this
  corpus, so the opponent name alone is not a key. The label is
  "Penn State (Oct 5)".

NON-ACTION ROWS ARE DROPPED.
  Roughly a third of the rows returned by get_plays() have skill = NaN:
  they are substitutions, timeouts and rotation markers (raw codes like
  "*P26>LUp", "az6"). They are not actions by a player and must not be
  counted as any. Filtering to a non-null skill is also what silently
  removes the trailing pseudo-set (see below).

set_number CARRIES A TRAILING PSEUDO-SET.
  get_plays() emits one extra set_number beyond the sets actually
  played -- a 3-0 sweep shows set_number 4 and a five-setter shows 6 --
  always holding exactly 1 row with no skill. Every real set count in
  this file therefore comes from rows that have a skill, never from
  max(set_number).

SETS PLAYED COMES FROM ROSTER METADATA, NOT FROM COUNTING ACTIONS.
  Two candidate methods were checked against the file's own per-set
  roster metadata across 769 player-rows in all 22 files:
    - counting distinct sets in which a player performed an action
      UNDERCOUNTS: metadata exceeded it in 113 cases (on court, never
      touched the ball).
    - reading the six-player rotation columns (home_p1..home_p6)
      catastrophically undercounts LIBEROS: Maryland's libero (#24,
      role "L" in the metadata) records 5 sets of digs and receptions
      while never once appearing in a rotation column, so that method
      scores her 0 sets and makes every per-set metric of hers a
      division by zero.
  The [3PLAYERS-H]/[3PLAYERS-V] metadata has an explicit per-set
  participation entry per player ("*" = played, "1".."6" = started in
  that rotation position, blank = did not play), it agreed with the
  action-based count everywhere the latter was non-zero (0 violations
  in 769 rows), and it is the only one of the three that is right for
  liberos. So it is what sets_played uses.

  WITH ONE CORRECTION, which counting the five columns naively gets
  wrong: a scout can enter the NEXT set's starting lineup before the
  match ends, and that entry is saved even if the set is never played.
  Maryland's 3-0 loss to Minnesota has a full six-player rotation
  (positions 1..6) sitting in the FOURTH participation column, for a
  set 4 that does not exist -- two of those six recorded no action in
  the match at all. Counting all five columns therefore reports 4 sets
  played in a 3-set match. The number of sets actually played is
  home_setswon + visiting_setswon from the [3TEAMS] metadata, and only
  that many columns are counted. With the cap applied, metadata and
  action counts reconcile exactly on every player who touched the ball
  (verified across all 22 files, both teams).
"""

import os
import re
from functools import lru_cache
from typing import Dict, FrozenSet, List, Optional

import pandas as pd

import dvw_patch  # noqa: F401  -- must be imported before DataVolley is constructed
from datavolley.read_dv import DataVolley

from source import FieldRole, FieldSpec, Grain, Source, SourceSchema

PLAYER_COLUMN = "player_label"
GAME_COLUMN = "match_label"
SET_COLUMN = "set_label"
TEAM_COLUMN = "team"

# Per-set participation columns in the [3PLAYERS-*] metadata. A blank
# entry means the player did not play that set; anything else ("*", or a
# starting rotation position "1".."6") means they did.
PARTICIPATION_COLUMNS = ["set1", "set2", "set3", "set4", "set5"]

# Dimensions worth filtering an event on. Any of these that turns out to
# be entirely empty in the loaded matches is dropped from the schema
# rather than advertised as filterable-but-useless.
CANDIDATE_DIMENSIONS = [
    "skill", "evaluation_code", "point_phase", "attack_phase",
    "set_type", "attack_code", "start_zone", "end_zone", "end_subzone",
    "num_players_numeric", "setter_position",
]

# The pair whose legality is conditional: an evaluation code means a
# different thing per skill, and is not even legal for every skill.
SKILL_FIELD = "skill"
EVALUATION_FIELD = "evaluation_code"

# Names whose generic shortening would be WRONG rather than merely ugly.
# The comma rule below turns "University of California, Los Angeles" into
# "California" -- which is a different school. These are display names
# only; MatchInfo keeps the full legal name as the key.
_TEAM_ALIASES = {
    "university of california los angeles": "UCLA",
    "university of california, los angeles": "UCLA",
    "university of southern california": "USC",
    "university of illinois urbana-champaign": "Illinois",
    "university of wisconsin-madison": "Wisconsin",
    "pennsylvania state university": "Penn State",
}

_UNIVERSITY_NOISE = [
    # Campus qualifier first ("Indiana University, Bloomington" ->
    # "Indiana University"), otherwise the trailing-University rule
    # below cannot fire because the string no longer ends in it.
    (re.compile(r",.*$"), ""),
    (re.compile(r"^University of ", re.IGNORECASE), ""),
    (re.compile(r" State University$", re.IGNORECASE), " State"),
    (re.compile(r" University$", re.IGNORECASE), ""),
]

_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def shorten_team_name(name: str) -> str:
    """
    "Ohio State University" -> "Ohio State", "University of Maryland" ->
    "Maryland". Purely cosmetic (chart axes and the router's game hints
    read badly with the full legal names), deterministic, and never
    applied to anything used as a key -- MatchInfo keeps the full name.
    """
    if not name:
        return ""
    short = name.strip()
    alias = _TEAM_ALIASES.get(short.lower())
    if alias:
        return alias
    for pattern, replacement in _UNIVERSITY_NOISE:
        short = pattern.sub(replacement, short)
    return short.strip().strip(",") or name.strip()


def format_match_day(day: Optional[str]) -> str:
    """"11/08/2025" -> "Nov 8". Returns "" for anything unparseable
    rather than guessing a date."""
    if not day or not isinstance(day, str):
        return ""
    parts = day.strip().split("/")
    if len(parts) != 3:
        return ""
    try:
        month, dom = int(parts[0]), int(parts[1])
    except ValueError:
        return ""
    if not 1 <= month <= 12:
        return ""
    return f"{_MONTHS[month - 1]} {dom}"


_MISSING_TOKENS = {"", "nan", "none", "<na>"}


def _clean_token(value: Optional[object]) -> str:
    """"" for anything that is not real content. Guards specifically
    against str(NaN) == "nan", which is truthy and would otherwise sail
    through an `if value:` check and become part of a label."""
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in _MISSING_TOKENS else text


def player_label(number: Optional[object], name: Optional[object]) -> str:
    """
    "#44 <name>" -- jersey number included because names are NOT unique
    on a real roster (see module docstring).

    Returns "" when the row identifies no player, which is how the
    rally-outcome rows get excluded: every skill="Point" row (a
    scoreline marker like "*p01:03", belonging to a TEAM rather than a
    player) carries a null player number AND name. Formatting those
    naively yields the label "#nan nan", which then behaves as a
    phantom roster member -- it appeared in all 22 files with 3 to 5
    "sets played" and would have polluted every ranking and every team
    average.
    """
    number, name = _clean_token(number), _clean_token(name)
    if number and name:
        return f"#{number} {name}"
    return name or (f"#{number}" if number else "")


def set_label(number: Optional[object]) -> str:
    """
    "Set 3". A label rather than the bare integer because this is an
    identity value on a cube axis, and the axis is displayed: bare 1..5
    reads as a count next to a column of counts.

    Returns "" for a row with no identifiable set, which the caller
    drops for the same reason it drops a row with no identifiable
    player -- it cannot be attributed, and keeping it would inflate
    whichever bucket pandas grouped the blank into.
    """
    token = _clean_token(number)
    if not token:
        return ""
    try:
        return f"Set {int(float(token))}"
    except (TypeError, ValueError):
        return ""


def set_sort_key(label: str) -> tuple:
    """Orders "Set 10" after "Set 9" rather than between "Set 1" and
    "Set 2", which is what sorting the strings would do."""
    try:
        return (0, int(label.split()[-1]))
    except (ValueError, IndexError):
        return (1, 0)


class MatchInfo:
    """One parsed .dvw file, and the labels derived from it."""

    def __init__(self, path: str, home_team: str, visiting_team: str, day: str,
                 team_of_interest: Optional[str] = None):
        self.path = path
        self.filename = os.path.basename(path)
        self.home_team = home_team
        self.visiting_team = visiting_team
        self.day = day
        self.team_of_interest = team_of_interest

    @property
    def opponent(self) -> str:
        """The team that is NOT the team of interest. With no team of
        interest set there is no meaningful opponent, so the visiting
        team stands in -- documented rather than silently assumed."""
        if self.team_of_interest == self.home_team:
            return self.visiting_team
        if self.team_of_interest == self.visiting_team:
            return self.home_team
        return self.visiting_team

    @property
    def short_opponent(self) -> str:
        """Just the other team, e.g. "Purdue" -- what a coach calls the
        match, and what the CSV dashboard puts on its Game axis."""
        return shorten_team_name(self.opponent)

    @property
    def label(self) -> str:
        """
        The Game axis value: opponent plus date, e.g. "Purdue (Nov 15)".

        The date is not decoration. Three opponents are played twice in
        this corpus, and two matches sharing an axis value would
        silently add their stats together -- a doubled kill count looks
        like a real number.
        """
        day = format_match_day(self.day)
        return f"{self.short_opponent} ({day})" if day else self.short_opponent

    def __repr__(self) -> str:
        return f"MatchInfo({self.label!r}, {self.filename!r})"


@lru_cache(maxsize=64)
def _read_match(path: str):
    """Parse one .dvw once. Cached because the Streamlit app re-runs its
    whole script on every interaction and a season is ~22 files; a parse
    is ~0.13s, so uncached this would cost ~3s per click."""
    dv = DataVolley(path)
    plays = dv.get_plays()
    match_row = dv.match_info.to_dict("records")
    day = match_row[0].get("day", "") if match_row else ""
    roster = pd.concat(
        [dv.players_home.assign(_team=dv.home_team),
         dv.players_visiting.assign(_team=dv.visiting_team)],
        ignore_index=True,
    )
    # Sets actually contested, from [3TEAMS]. Needed to ignore a
    # pre-entered lineup for a set that was never played -- see the
    # module docstring's sets-played note.
    sets_played = int(dv.home_setswon) + int(dv.visiting_setswon)
    return plays, dv.home_team, dv.visiting_team, day, roster, sets_played


def discover_matches(directory: str) -> List[str]:
    """Every .dvw in a directory, sorted. No recursion: the corpus keeps
    duplicates of the same match in sibling folders, and silently
    loading a match twice would double every count."""
    if not os.path.isdir(directory):
        return []
    return sorted(
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if name.lower().endswith(".dvw")
    )


class DvwSource(Source):
    """
    EVENT-grain Source over one or more .dvw files.

    `team_of_interest` scopes the roster to one team -- a .dvw records
    BOTH teams' actions, so without it "every player" means both benches
    at once and a "team average" straddles two teams. None means keep
    both, which is legitimate for scouting an opponent but must be an
    explicit choice.
    """

    def __init__(self, paths: List[str], team_of_interest: Optional[str] = None):
        self.paths = list(paths)
        self.team_of_interest = team_of_interest
        self._facts: Optional[pd.DataFrame] = None
        self._measures: Optional[pd.DataFrame] = None
        self._schema: Optional[SourceSchema] = None
        self._matches: Optional[List[MatchInfo]] = None

    # ── Source contract ───────────────────────────────────────

    @property
    def grain(self) -> Grain:
        return Grain.EVENT

    @property
    def schema(self) -> SourceSchema:
        if self._schema is None:
            self._schema = self._build_schema(self.facts())
        return self._schema

    def identity_fields(self) -> Dict[str, str]:
        """Set is an identity axis here and absent for the CSV source --
        that difference is the whole reason axes() asks the source."""
        return {"Player": PLAYER_COLUMN, "Game": GAME_COLUMN, "Set": SET_COLUMN}

    def facts(self) -> pd.DataFrame:
        if self._facts is None:
            self._load()
        return self._facts

    def measures(self) -> pd.DataFrame:
        if self._measures is None:
            self._load()
        return self._measures

    # ── loading ───────────────────────────────────────────────

    def matches(self) -> List[MatchInfo]:
        if self._matches is None:
            self._load()
        return self._matches

    def game_labels(self) -> List[str]:
        """File order rather than alphabetical: the corpus is named by
        date, so this keeps a season roughly chronological in the game
        picker and along a chart's Game axis."""
        seen, labels = set(), []
        for match in self.matches():
            if match.label not in seen:
                seen.add(match.label)
                labels.append(match.label)
        return labels

    def teams(self) -> List[str]:
        """
        Every team appearing in the loaded files -- what a caller picks
        team_of_interest from -- most-played first.

        The order matters: a scouting corpus is ABOUT one team, which
        appears in every file while opponents appear in one or two. That
        team is the right default, and sorting alphabetically would have
        buried it.
        """
        counts: Dict[str, int] = {}
        for path in self.paths:
            _, home, visiting, _, _, _ = _read_match(path)
            for team in {home, visiting}:
                counts[team] = counts.get(team, 0) + 1
        return sorted(counts, key=lambda team: (-counts[team], team))

    def default_team(self) -> Optional[str]:
        """The team this corpus is about, or None if there are no files.
        Without it, MatchInfo.opponent falls back to the visiting team
        and a match Maryland played away is labelled "Maryland"."""
        teams = self.teams()
        return teams[0] if teams else None

    def _load(self) -> None:
        fact_frames, measure_frames, matches = [], [], []

        for path in self.paths:
            plays, home, visiting, day, roster, sets_contested = _read_match(path)
            info = MatchInfo(path, home, visiting, day, self.team_of_interest)
            matches.append(info)

            # Actions only: drop substitutions/timeouts/rotation markers
            # (skill is NaN on those), which also drops the trailing
            # pseudo-set. See module docstring.
            actions = plays[plays["skill"].notna()].copy()
            if self.team_of_interest is not None:
                actions = actions[actions[TEAM_COLUMN] == self.team_of_interest]

            actions[PLAYER_COLUMN] = [
                player_label(n, nm)
                for n, nm in zip(actions["player_number"], actions["player_name"])
            ]
            actions[GAME_COLUMN] = info.label
            actions[SET_COLUMN] = [set_label(n) for n in actions["set_number"]]
            # A row with no identifiable player cannot be attributed to
            # anyone; keeping it would inflate whichever bucket pandas
            # grouped the blank label into.
            actions = actions[actions[PLAYER_COLUMN].str.strip() != ""]
            fact_frames.append(actions)

            measure_frames.append(self._sets_played_frame(roster, info, sets_contested))

        self._facts = (
            pd.concat(fact_frames, ignore_index=True) if fact_frames else pd.DataFrame()
        )
        self._measures = (
            pd.concat(measure_frames, ignore_index=True) if measure_frames else pd.DataFrame()
        )
        self._matches = matches

    def _sets_played_frame(self, roster: pd.DataFrame, info: MatchInfo,
                            sets_contested: int) -> pd.DataFrame:
        """
        Sets played for one match, reported PER SET: one row per
        (player, match, set) the player was actually on court for, with
        sets_played = 1.

        Reported at the finest grain the data supports rather than
        pre-summed to the match, because the coarser answer is
        recoverable from this one (sum over Set) and the finer one is
        not recoverable from the coarser. That is what makes a per-set
        rate correct when a set is selected: "Kills Per Set" in set 3
        divides by the 1 set she played in that slice, not by the 4 she
        played in the match.

        Source of truth is the roster metadata's per-set participation
        entries -- the only one of the three available methods that is
        correct for liberos (module docstring). Only the first
        `sets_contested` participation columns are read: a lineup
        entered for a set that was never played still sits in the file
        and would otherwise be counted as a set played.
        """
        columns = [PLAYER_COLUMN, GAME_COLUMN, SET_COLUMN, "sets_played"]
        if roster.empty:
            return pd.DataFrame(columns=columns)

        rows = roster
        if self.team_of_interest is not None:
            rows = rows[rows["_team"] == self.team_of_interest]

        countable = PARTICIPATION_COLUMNS[:max(0, sets_contested)]

        records = []
        for _, entry in rows.iterrows():
            label = player_label(entry.get("player_number"), entry.get("player_name"))
            if not label.strip():
                continue
            for offset, column in enumerate(countable, start=1):
                if column not in entry.index:
                    continue
                if str(entry[column]).strip() in ("", "nan", "None"):
                    continue
                records.append({
                    PLAYER_COLUMN: label,
                    GAME_COLUMN: info.label,
                    SET_COLUMN: set_label(offset),
                    "sets_played": 1,
                })
        return pd.DataFrame(records, columns=columns)

    # ── schema derivation ─────────────────────────────────────

    def _build_schema(self, facts: pd.DataFrame) -> SourceSchema:
        fields: Dict[str, FieldSpec] = {
            PLAYER_COLUMN: FieldSpec(PLAYER_COLUMN, FieldRole.IDENTITY, "Player (jersey + name)"),
            GAME_COLUMN: FieldSpec(GAME_COLUMN, FieldRole.IDENTITY, "Match (opponent + date)"),
        }

        if not facts.empty:
            for name in CANDIDATE_DIMENSIONS:
                if name not in facts.columns:
                    continue
                values = self._observed_values(facts, name)
                if not values:
                    # Present in the frame but never populated in these
                    # matches -- advertising it would let a metric be
                    # authored that can only ever return zero.
                    continue
                fields[name] = FieldSpec(
                    name=name, role=FieldRole.DIMENSION,
                    description=f"{name} ({len(values)} observed values)",
                    values=values,
                )

        dependent: Dict[tuple, Dict[str, FrozenSet[str]]] = {}
        if not facts.empty and SKILL_FIELD in facts.columns and EVALUATION_FIELD in facts.columns:
            dependent[(SKILL_FIELD, EVALUATION_FIELD)] = self._evaluation_vocabulary(facts)

        return SourceSchema(fields=fields, dependent_values=dependent)

    @staticmethod
    def _observed_values(facts: pd.DataFrame, column: str) -> FrozenSet[str]:
        series = facts[column].dropna()
        values = {str(v).strip() for v in series.unique()}
        return frozenset(v for v in values if v not in ("", "nan", "None"))

    @staticmethod
    def _evaluation_vocabulary(facts: pd.DataFrame) -> Dict[str, FrozenSet[str]]:
        """
        skill -> the evaluation codes actually recorded for THAT skill.

        Keyed on the pair rather than on the code alone because the code
        is meaningless without its skill: "#" is an ace on a Serve, a
        perfect pass on a Reception and a kill on an Attack, and several
        skills never take some codes at all (in this corpus Set only
        ever takes #, - and =). A flat list of six codes would happily
        accept "a Set with evaluation /", which cannot occur.
        """
        pairs = facts[[SKILL_FIELD, EVALUATION_FIELD]].dropna()
        vocabulary: Dict[str, FrozenSet[str]] = {}
        for skill, group in pairs.groupby(SKILL_FIELD):
            codes = {str(c).strip() for c in group[EVALUATION_FIELD].unique()}
            vocabulary[str(skill)] = frozenset(c for c in codes if c not in ("", "nan", "None"))
        return vocabulary
