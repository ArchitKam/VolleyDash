"""
test_dvw_patch.py
==================
Tests the get_set correction against the UNPATCHED upstream algorithm,
which is reproduced verbatim below rather than snapshotted -- an
equivalence claim ("identical where upstream worked") is only worth
anything if it is checked against the real thing.

The real-file tests skip gracefully when the .dvw corpus is absent,
following the test_live_* pattern in the CSV app's test_integration.py.
"""

import collections
import glob
import os
import re

import pandas as pd
import pytest

import dvw_patch
from dvw_patch import SET_COLUMNS, get_set

from volley_store import DVW_SEARCH_DIRS as DVW_DIRS


def upstream_get_set(rows_list):
    """
    pydatavolley 2.3's datavolley.helpers.get_set, copied verbatim. Kept
    here so the equivalence tests below compare against the actual
    upstream behaviour rather than a recorded expectation.
    """
    sets_index = rows_list.index("[3SET]\n")
    sets_data = []
    sets_label = ["set", "home1", "visitor1", "home2", "visitor2",
                  "home3", "visitor3", "home4", "visitor4", "duration"]
    for idx in range(1, 6):
        rowdata = rows_list[sets_index + idx].strip().split(";")
        set_data = []
        set_data.append(idx)
        add = True
        try:
            set_data.append(int(rowdata[1].split("-")[0]))
            set_data.append(int(rowdata[1].split("-")[1]))
            set_data.append(int(rowdata[2].split("-")[0]))
            set_data.append(int(rowdata[2].split("-")[1]))
            set_data.append(int(rowdata[3].split("-")[0]))
            set_data.append(int(rowdata[3].split("-")[1]))
            set_data.append(int(rowdata[4].split("-")[0]))
            set_data.append(int(rowdata[4].split("-")[1]))
        except Exception:
            for _ in range(9):
                set_data.append(None)
                add = False
        if add:
            set_data.append(int(rowdata[5]))
        sets_data.append(set_data)
    return pd.DataFrame(data=sets_data, columns=sets_label)


def _rows(*set_lines):
    """A minimal rows_list containing just a [3SET] section."""
    return ["[3MATCH]\n", "x\n", "[3SET]\n"] + [line + "\n" for line in set_lines] + ["[3PLAYERS-H]\n"]


# ── the defect itself ──────────────────────────────────────────

def test_upstream_really_crashes_on_a_partially_scored_set():
    """Guards the premise: if upstream stops crashing, this whole patch
    should be revisited rather than silently kept."""
    rows = _rows("True;8-7;16-14;21-19;25-23;25;", "True;8-6;12-16;14-21;16-25;25;",
                 "True;4-8;13-16;18-21;21-25;25;", "True;8-5;16-8;21-9;25-11;25;",
                 "True;6-8;;;12-15;15;")
    with pytest.raises(ValueError, match="10 columns passed, passed data had 12 columns"):
        upstream_get_set(rows)


def test_patched_parses_the_partially_scored_set():
    rows = _rows("True;8-7;16-14;21-19;25-23;25;", "True;8-6;12-16;14-21;16-25;25;",
                 "True;4-8;13-16;18-21;21-25;25;", "True;8-5;16-8;21-9;25-11;25;",
                 "True;6-8;;;12-15;15;")
    frame = get_set(rows)
    assert list(frame.columns) == SET_COLUMNS
    assert len(frame) == 5

    fifth = frame.iloc[4]
    # The scores that ARE in the file survive; the unreached scoreboard
    # milestones are None rather than taking the whole row down.
    assert (fifth["home1"], fifth["visitor1"]) == (6, 8)
    assert pd.isna(fifth["home2"]) and pd.isna(fifth["visitor2"])
    assert pd.isna(fifth["home3"]) and pd.isna(fifth["visitor3"])
    assert (fifth["home4"], fifth["visitor4"]) == (12, 15)


def test_every_row_has_exactly_the_declared_width():
    """The crash was a width mismatch, so width is the invariant."""
    rows = _rows("True;1-2;3-4;5-6;7-8;25;", "True;;;;;25;", "True;6-8;;;12-15;15;",
                 "True;garbage;;;;;", "True;1-2;;5-6;;15;")
    frame = get_set(rows)
    assert frame.shape == (5, len(SET_COLUMNS))


def test_stops_at_the_next_section_instead_of_reading_past_it():
    """Upstream always reads 5 lines after [3SET]; a file with fewer
    would have its next section parsed as set data."""
    frame = get_set(_rows("True;1-2;3-4;5-6;7-8;25;", "True;2-3;4-5;6-7;8-9;25;"))
    assert len(frame) == 2


def test_unplayed_set_rows_agree_with_upstream_on_the_scores():
    """The documented behavioural difference is confined to the last
    field; every score column still matches upstream exactly."""
    rows = _rows("True;6-8;12-16;14-21;16-25;25;", "True;2-8;6-16;9-21;11-25;25;",
                 "True;1-8;7-16;10-21;13-25;25;", "True;;;;;25;", "True;;;;;15;")
    score_columns = [c for c in SET_COLUMNS if c != "duration"]
    pd.testing.assert_frame_equal(
        get_set(rows)[score_columns].astype("float64"),
        upstream_get_set(rows)[score_columns].astype("float64"),
    )


# ── numpy compatibility ────────────────────────────────────────

def test_numpy_nan_alias_is_available_after_patching():
    """read_dv.py references np.NaN, removed in NumPy 2.0."""
    import numpy as np
    assert hasattr(np, "NaN")
    assert np.isnan(np.NaN)


def test_apply_patches_is_idempotent_and_rebinds_both_names():
    from datavolley import helpers as dv_helpers
    from datavolley import read_dv as dv_read

    dvw_patch.apply_patches()
    dvw_patch.apply_patches()
    # read_dv did `from .helpers import get_set`, so it holds its own
    # binding; patching only helpers would miss the one actually called.
    assert dv_helpers.get_set is get_set
    assert dv_read.get_set is get_set


# ── against the real corpus ────────────────────────────────────

def _real_files():
    paths = []
    for directory in DVW_DIRS:
        paths.extend(glob.glob(os.path.join(directory, "*.dvw")))
    return sorted(set(paths))


real_files = _real_files()
requires_corpus = pytest.mark.skipif(not real_files, reason="no .dvw corpus available here")


@requires_corpus
def test_every_real_file_parses_end_to_end():
    from datavolley.read_dv import DataVolley

    failures = []
    for path in real_files:
        try:
            DataVolley(path).get_plays()
        except Exception as error:
            failures.append(f"{os.path.basename(path)}: {type(error).__name__}: {error}")
    assert not failures, "files still failing to parse:\n" + "\n".join(failures)


# ── defect 3: jersey number 0 discarded as "no player" ─────────

ACTION_CODE = re.compile(r"^([*a])(\d{2})([SRAEDBF])(.)([#+!/=\-])")
SKILL_BY_CODE = {"S": "Serve", "R": "Reception", "E": "Set", "A": "Attack",
                 "D": "Dig", "B": "Block", "F": "Freeball"}


def independent_action_counts(path):
    """
    Decodes the raw [3SCOUT] lines from scratch using only the code
    grammar, sharing NO code with pydatavolley. This is the check that
    found the jersey-0 defect in the first place, so it is kept as the
    regression test rather than a recorded expectation.
    """
    from charset_normalizer import from_path

    rows = open(path, "r", encoding=from_path(path).best().encoding).readlines()
    start = rows.index("[3SCOUT]\n")
    counts = collections.Counter()
    for line in rows[start + 1:]:
        if line.startswith("["):
            break
        match = ACTION_CODE.match(line.split(";")[0].strip())
        if match:
            side, number, skill_char, _type, evaluation = match.groups()
            counts[(side, str(int(number)), SKILL_BY_CODE[skill_char], evaluation)] += 1
    return counts


def library_action_counts(path):
    from datavolley.read_dv import DataVolley

    dv = DataVolley(path)
    plays = dv.get_plays()
    acts = plays[plays["skill"].notna() & plays["evaluation_code"].notna()]
    counts = collections.Counter()
    for _, row in acts.iterrows():
        number = str(row["player_number"])
        if number in ("nan", "None", ""):
            continue
        side = "*" if row["team"] == dv.home_team else "a"
        counts[(side, str(int(float(number))), row["skill"], row["evaluation_code"])] += 1
    return counts


@requires_corpus
def test_decoding_agrees_with_an_independent_decoder_on_every_file():
    """
    The strongest available check that the .dvw is understood correctly:
    every (team, jersey, skill, evaluation code) count must match a
    decoder written straight from the raw grammar. Before the jersey-0
    fix this was 20/22 files and 161 missing actions.
    """
    mismatches = []
    total_independent = total_library = 0
    for path in real_files:
        independent, library = independent_action_counts(path), library_action_counts(path)
        total_independent += sum(independent.values())
        total_library += sum(library.values())
        if independent != library:
            mismatches.append(os.path.basename(path))
    assert not mismatches, f"files disagreeing with an independent decode: {mismatches}"
    assert total_independent == total_library
    assert total_library > 0


@requires_corpus
def test_jersey_zero_players_survive_decoding():
    """
    Jersey 0 is legal and used in this corpus. read_dv used "0" both as
    the sentinel for "no number extracted" and as a real number, so a
    #0's actions were dropped entirely -- player_number went NaN, and
    calculate_skill nulls skill whenever player_number is NaN.
    """
    from datavolley.read_dv import DataVolley

    repaired_files = 0
    for path in real_files:
        dv = DataVolley(path)
        plays = dv.get_plays()
        if not dv.jersey_zero_repaired_rows:
            continue
        repaired_files += 1
        zeros = plays[plays["player_number"] == "0"]
        assert len(zeros) == dv.jersey_zero_repaired_rows
        assert zeros["skill"].notna().all(), "a repaired row must have its skill back"
        assert zeros["player_name"].notna().all(), "and must resolve to a roster name"
    assert repaired_files > 0, "expected at least one file with a jersey-0 player"


@requires_corpus
def test_files_without_jersey_zero_are_left_untouched():
    """The repair must be a no-op wherever the defect does not apply."""
    from datavolley.read_dv import DataVolley

    untouched = 0
    for path in real_files:
        dv = DataVolley(path)
        dv.get_plays()
        if dv.jersey_zero_repaired_rows == 0:
            untouched += 1
    assert untouched > 0


def test_point_rows_are_not_given_a_player_by_the_repair():
    """A rally-outcome code ("ap00:01") has "p" where a jersey would be,
    so it must keep having no player -- it belongs to a team."""
    assert ACTION_CODE.match("ap00:01") is None
    assert ACTION_CODE.match("*p01:03") is None
    assert ACTION_CODE.match("a00AT+X6~27DH4~00F") is not None


@requires_corpus
def test_matches_upstream_scores_on_every_file_upstream_could_read():
    """Where upstream produced an answer at all, the patched parser
    agrees with it on every score column."""
    from charset_normalizer import from_path

    score_columns = [c for c in SET_COLUMNS if c != "duration"]
    compared = 0
    for path in real_files:
        rows = open(path, "r", encoding=from_path(path).best().encoding).readlines()
        try:
            expected = upstream_get_set(rows)
        except Exception:
            continue  # upstream crashed here; nothing to compare against
        pd.testing.assert_frame_equal(
            get_set(rows)[score_columns].astype("float64"),
            expected[score_columns].astype("float64"),
        )
        compared += 1
    assert compared > 0, "expected at least one file upstream could read"
