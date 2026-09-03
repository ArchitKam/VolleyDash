"""
dvw_patch.py
=============
Corrects two real defects in pydatavolley 2.3 that make it unusable on
part of a real season's worth of .dvw files. Applied as a monkeypatch
rather than a fork so the dependency stays `pip install pydatavolley`
and an upstream fix later just makes this a no-op.

Import this module (or call apply_patches()) BEFORE constructing
datavolley.read_dv.DataVolley. volley_source_dvw.py already does.

────────────────────────────────────────────────────────────────
DEFECT 1 -- get_set() crashes on any partially-scored set row
────────────────────────────────────────────────────────────────
Measured: 4 of 22 real match files in this project fail to parse AT ALL
with `ValueError: 10 columns passed, passed data had 12 columns`.

The upstream get_set() builds one list per set row, appending as it
parses, and on ANY exception appends a fixed 9 Nones -- WITHOUT
discarding what the try block already appended:

    set_data = [idx]                       # 1 element
    try:
        for each of 4 quarter fields:      # appends 2 each on success
            set_data.append(home); set_data.append(visitor)
    except Exception:
        for _ in range(9):                 # blindly appends 9 more
            set_data.append(None)

So the element count depends on HOW FAR the try got before raising:
  - fails on field 1 (an unplayed set, "True;;;;;25;"):
        1 + 0 + 9 = 10  -> matches the 10 column labels, accidentally OK
  - fails partway  (a played set, "True;6-8;;;12-15;15;"):
        1 + 2 + 9 = 12  -> 12 values for 10 labels -> ValueError

It is NOT "5-set matches" as such; it is any set row where some quarter
score fields are populated and others are blank. That is exactly the
shape of a 15-point deciding set: DataVolley records the score at fixed
scoreboard milestones, and a set to 15 never reaches the 16/21 markers,
so those fields are empty while the 8-point and final fields are filled.

The all-or-nothing handler also DESTROYS real data on rows it does not
crash on: a row it gives up on has its scores AND its duration nulled,
even though they are right there in the file.

The fix parses each field independently -- a blank field becomes None,
a populated one is kept -- so the row is always exactly 10 values and
no recorded score is thrown away.

The one intentional behavioural difference on files that already
parsed: upstream nulls the last field of any row it gave up on, so an
unplayed set reads as all-NaN; this parser reports the value actually
present in the file. See the column-naming note below for why that
value is not what upstream's label claims anyway.

────────────────────────────────────────────────────────────────
MEASURED, NOT ASSUMED -- two upstream labels that do not hold
────────────────────────────────────────────────────────────────
Both checked across all 22 match files in this project (110 set rows):

  - Field 5, which upstream names "duration", is EXACTLY "25" for sets
    1-4 and EXACTLY "15" for set 5 in 110 of 110 rows. It is the set's
    target score, not a duration in minutes -- no set in a season takes
    precisely 25 minutes every time. The column keeps upstream's name
    here for drop-in compatibility, but do not read it as minutes.

  - Field 0, which upstream treats as a played flag, is the string
    "True" in 110 of 110 rows -- including sets that were never played
    (a 3-0 sweep still emits rows for sets 4 and 5). It cannot be used
    to tell whether a set happened. The reliable signal is whether the
    four score-pair fields are populated, which is what this parser
    preserves faithfully instead of collapsing to NaN.

────────────────────────────────────────────────────────────────
DEFECT 2 -- np.NaN was removed in NumPy 2.0
────────────────────────────────────────────────────────────────
read_dv.py references np.NaN, which NumPy 2.0 removed in favour of
np.nan, so on NumPy >= 2 EVERY file fails with AttributeError before
any of the above matters. np.NaN was only ever an alias of np.nan, so
restoring the alias is exactly faithful, and is skipped entirely on
NumPy 1.x where the attribute already exists.
"""

import re
from typing import Any, List, Optional

import numpy as np
import pandas as pd

# Same 10 labels upstream uses, in the same order -- callers (and
# DataVolley.sets_info) see an unchanged frame shape.
SET_COLUMNS = [
    "set", "home1", "visitor1", "home2", "visitor2",
    "home3", "visitor3", "home4", "visitor4", "duration",
]

# A set row holds 4 quarter-score fields (index 1..4) then duration
# (index 5); index 0 is the played flag, which upstream ignores and so
# does this.
_QUARTER_FIELD_INDEXES = (1, 2, 3, 4)
_DURATION_FIELD_INDEX = 5
_MAX_SETS = 5


def _parse_int(text: Optional[str]) -> Optional[int]:
    """int() or None -- never raises. A blank/absent/garbage field is
    genuinely absent data (an unreached scoreboard milestone), not an
    error to abort the whole row over."""
    if text is None:
        return None
    text = text.strip()
    if not text:
        return None
    try:
        return int(text)
    except (TypeError, ValueError):
        return None


def _parse_score_pair(field: Optional[str]) -> List[Optional[int]]:
    """"12-15" -> [12, 15]; "" or malformed -> [None, None]. Always
    returns exactly 2 values so the row's width is fixed no matter what
    the file contains."""
    if field is None:
        return [None, None]
    parts = field.strip().split("-")
    if len(parts) != 2:
        return [None, None]
    return [_parse_int(parts[0]), _parse_int(parts[1])]


def get_set(rows_list: List[str]) -> pd.DataFrame:
    """
    Drop-in replacement for datavolley.helpers.get_set.

    Always emits exactly len(SET_COLUMNS) values per set row, parsing
    every field independently, so a partially-scored set (the deciding
    15-point set, above all) neither crashes the read nor loses the
    scores it does have.

    Also stops at the next [SECTION] header instead of blindly reading
    5 lines past [3SET]: upstream assumes the section is always padded
    to 5 rows, and would silently parse a following section's lines as
    set data on a file that isn't.
    """
    sets_index = rows_list.index("[3SET]\n")
    sets_data = []

    for idx in range(1, _MAX_SETS + 1):
        line_index = sets_index + idx
        if line_index >= len(rows_list):
            break
        raw_line = rows_list[line_index].strip()
        # A new section header means this file simply recorded fewer
        # than 5 set rows -- stop rather than misread it as set data.
        if raw_line.startswith("["):
            break

        fields = raw_line.split(";")
        row: List[Any] = [idx]
        for field_index in _QUARTER_FIELD_INDEXES:
            field = fields[field_index] if field_index < len(fields) else None
            row.extend(_parse_score_pair(field))
        duration_field = (
            fields[_DURATION_FIELD_INDEX] if _DURATION_FIELD_INDEX < len(fields) else None
        )
        row.append(_parse_int(duration_field))
        sets_data.append(row)

    return pd.DataFrame(data=sets_data, columns=SET_COLUMNS)


# ──────────────────────────────────────────────────────────────
# DEFECT 3 -- jersey number 0 is discarded as if it were "no player"
#
# read_dv.py builds the player number like this:
#
#   plays['player_number'] = plays['code'].str[1:3].str.extract(r'(\d{2})') \
#       .astype(float).fillna(0).astype(int).astype(str)
#   plays['player_number'] = np.where(plays['player_number'] == '0',
#                                     np.nan, plays['player_number'])
#
# fillna(0) turns "the regex matched nothing" into the string "0", and
# the next line then throws away everything equal to "0". That works
# only while no player wears number 0 -- and it uses the same value to
# mean "no number here" and "the number zero".
#
# Jersey 0 is legal in NCAA volleyball and IS used in this corpus:
# Rutgers' #0 plays in both Rutgers matches. Her scout codes are written
# with the jersey zero-padded to the two-character field ("a00AT+..."),
# which extracts as "00" -> 0 -> discarded. player_number becomes NaN,
# and because calculate_skill returns NaN whenever player_number is NaN,
# her SKILL is nulled too -- so all 161 of her actions across the two
# files disappear from the play-by-play entirely.
#
# Measured with an independent decoder written straight from the code
# grammar: 20 of 22 files agreed with pydatavolley exactly (21,313
# actions); the 2 disagreements were these two files, missing exactly
# her 75 and 86 actions.
#
# The repair restores the player, the skill, and every field that is
# derived from skill, recomputing the derived ones with the SAME
# expressions read_dv uses so a repaired file is what upstream would
# have produced had the collision never happened. Files with no jersey-0
# codes return early and are left bit-identical.
# ──────────────────────────────────────────────────────────────

# team char, two-digit jersey, skill char. Same grammar the independent
# decoder used.
_ACTION_CODE = re.compile(r"^[*a]\d{2}[SRAEDBF]")

_SKILL_BY_CODE = {
    "S": "Serve", "R": "Reception", "E": "Set", "A": "Attack",
    "D": "Dig", "B": "Block", "F": "Freeball",
}


def _roster_lookup(dv) -> dict:
    """(team, player_number) -> (player_name, player_id) from the match's
    own [3PLAYERS-*] metadata."""
    lookup = {}
    for roster, team in ((dv.players_home, dv.home_team),
                          (dv.players_visiting, dv.visiting_team)):
        if roster is None or roster.empty:
            continue
        for _, entry in roster.iterrows():
            number = str(entry.get("player_number", "")).strip()
            if not number:
                continue
            lookup[(team, number)] = (entry.get("player_name"), entry.get("player_id"))
    return lookup


def _recompute_skill_derived_fields(dv, plays: pd.DataFrame) -> None:
    """
    Re-derive everything read_dv gates on `skill`, using its own
    expressions verbatim. Fields NOT gated on skill (start_zone,
    end_zone, end_subzone, the scores) are pure slices of `code` and
    were already correct, so they are left alone.
    """
    code = plays["code"].astype(str)

    plays["set_code"] = np.where(plays["skill"] == "Set", code.str[6:8], np.nan)
    plays["set_code"] = np.where(
        (plays["skill"] == "Set") & (plays["set_code"] != "~~"), plays["set_code"], np.nan)

    plays["set_type"] = np.where(plays["skill"] == "Set", code.str[8:9], np.nan)
    plays["set_type"] = np.where(
        (plays["skill"] == "Set") & (plays["set_type"] != "~~"), plays["set_type"], np.nan)

    plays["attack_code"] = code.str[6:8]
    plays["attack_code"] = np.where(
        (plays["skill"] == "Attack") & (plays["attack_code"] != "~~"), plays["attack_code"], np.nan)

    plays["num_players_numeric"] = np.where(plays["skill"] == "Attack", code.str[13:14], np.nan)
    plays["num_players_numeric"] = np.where(
        (plays["skill"] == "Attack") & (plays["num_players_numeric"] != "~~"),
        plays["num_players_numeric"], np.nan)

    plays["serving_team"] = np.where(
        (plays["skill"] == "Serve") & (code.str[0:1] == "*"), dv.home_team, None)
    plays["serving_team"] = np.where(
        (plays["skill"] == "Serve") & (code.str[0:1] == "a"), dv.visiting_team,
        plays["serving_team"])
    plays["serving_team"] = plays.groupby(["set_number", "rally_number"])["serving_team"].ffill()

    plays["receiving_team"] = np.where(
        plays["serving_team"] == dv.home_team, dv.visiting_team, dv.home_team)
    plays["receiving_team"] = np.where(
        plays["serving_team"].isna(), np.nan, plays["receiving_team"])

    plays["point_phase"] = np.where(
        plays["serving_team"] == plays["team"], "Serve", "Reception")

    plays["attack_phase"] = np.where(
        (plays["skill"] == "Attack") & (plays["skill"].shift(2) == "Reception")
        & (plays["skill"].shift(1) == "Set") & (plays["team"].shift(2) == plays["team"]),
        "Reception", np.nan)
    plays["attack_phase"] = np.where(
        (plays["skill"] == "Attack") & (plays["skill"].shift(2) != "Reception")
        & (plays["skill"].shift(1) == "Set") & (plays["serving_team"] != plays["team"])
        & (plays["team"].shift(2) == plays["team"]), "SO-Transition", plays["attack_phase"])
    plays["attack_phase"] = np.where(
        (plays["skill"] == "Attack") & (plays["skill"].shift(2) != "Reception")
        & (plays["skill"].shift(1) == "Set") & (plays["serving_team"] == plays["team"])
        & (plays["team"].shift(2) == plays["team"]), "BP-Transition", plays["attack_phase"])

    plays["possesion_number"] = (
        plays.groupby(["set_number", "rally_number"], group_keys=False)["skill"]
        .apply(lambda x: (x == "Attack").shift(1).cumsum() + 1).fillna(0).astype(int)
    )


def repair_jersey_zero(dv) -> int:
    """
    Restore actions whose player was discarded because their jersey is 0.
    Returns how many rows were repaired (0 leaves the frame untouched).
    """
    plays = getattr(dv, "plays", None)
    if plays is None or plays.empty:
        return 0

    code = plays["code"].astype(str)
    # A real action by a two-digit jersey that nonetheless has no player.
    # Point rows ("ap00:01") do not match: their second character is "p",
    # not a digit, so they keep having no player, which is correct --
    # they belong to a team, not a person.
    candidates = code.str.match(_ACTION_CODE) & plays["player_number"].isna()
    if not candidates.any():
        return 0

    jerseys = code.str[1:3]
    # Canonical form is the roster's: "00" -> "0", matching how
    # [3PLAYERS-*] stores it, so the name lookup below can hit.
    canonical = jerseys.where(~candidates).copy()
    canonical[candidates] = [str(int(v)) for v in jerseys[candidates]]

    plays.loc[candidates, "player_number"] = canonical[candidates]
    plays.loc[candidates, "skill"] = [
        _SKILL_BY_CODE.get(value[3]) for value in code[candidates]
    ]

    lookup = _roster_lookup(dv)
    for index in plays.index[candidates]:
        key = (plays.at[index, "team"], plays.at[index, "player_number"])
        name, player_id = lookup.get(key, (np.nan, np.nan))
        plays.at[index, "player_name"] = name
        if "player_id" in plays.columns:
            plays.at[index, "player_id"] = player_id

    _recompute_skill_derived_fields(dv, plays)
    return int(candidates.sum())


def _patch_get_plays() -> bool:
    """Wrap DataVolley.get_plays so the repair runs once per instance,
    the first time the frame is asked for."""
    from datavolley.read_dv import DataVolley

    if getattr(DataVolley.get_plays, "_jersey_zero_wrapped", False):
        return False

    original = DataVolley.get_plays

    def get_plays(self):
        if not getattr(self, "_jersey_zero_repaired", False):
            self._jersey_zero_repaired = True
            self.jersey_zero_repaired_rows = repair_jersey_zero(self)
        return original(self)

    get_plays._jersey_zero_wrapped = True
    get_plays.__doc__ = original.__doc__
    DataVolley.get_plays = get_plays
    return True


def _patch_numpy_nan_alias() -> bool:
    """Restore np.NaN on NumPy >= 2 (removed there; was an alias of
    np.nan). Returns whether it had to be added."""
    if hasattr(np, "NaN"):
        return False
    np.NaN = np.nan  # type: ignore[attr-defined]
    return True


def apply_patches() -> dict:
    """
    Idempotent. Returns what was actually patched, so a caller (or a
    test) can assert the patch really took effect rather than assume it.

    read_dv.py does `from .helpers import get_set`, which binds a SECOND
    name in read_dv's own namespace -- patching only datavolley.helpers
    would leave the copy the parser actually calls untouched, so both
    are rebound.
    """
    applied = {"numpy_nan_alias": _patch_numpy_nan_alias()}

    from datavolley import helpers as dv_helpers
    from datavolley import read_dv as dv_read

    applied["helpers.get_set"] = dv_helpers.get_set is not get_set
    applied["read_dv.get_set"] = dv_read.get_set is not get_set
    dv_helpers.get_set = get_set
    dv_read.get_set = get_set
    applied["get_plays.jersey_zero"] = _patch_get_plays()
    return applied


apply_patches()
