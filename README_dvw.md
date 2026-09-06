# _volley_dash — DataVolley (.dvw) source for VolleyDash

An **event-grain** twin of the CSV recruiting dashboard, reading DataVolley
play-by-play files instead of pre-aggregated Huddle exports.

```bash
cd .../recruiting_analysis
pip install pydatavolley "numpy<2.0"      # see "Parser corrections" below
streamlit run _volley_dash/app.py
pytest _volley_dash -q                    # 76 tests
```

The `.dvw` folder is set in the sidebar, or via `VOLLEY_DVW_DIR`.

## Why a Source abstraction

The two file formats differ in **grain** — what one row means — and grain
decides which metric kinds are even expressible:

| | Huddle CSV | DataVolley `.dvw` |
|---|---|---|
| One row is | a player's totals for a match | a single action |
| Fields are | already-computed measures | attributes of one event |
| A metric is | a column, or arithmetic over columns | a **filter + aggregate**, or arithmetic over metrics |

`volley_source.py` states that boundary (`Grain`, `SourceSchema`, `Source`);
`volley_source_dvw.py` implements it. Everything above the boundary — the tree
engine, the Slice/Reduce/Rank/Compare algebra, chart encoding, and the query
router — is **imported unchanged** from the parent app rather than forked.

`formula` works at both grains because it composes over *metrics*, not raw
fields: `[Kills] / [Sets Played]` is the same expression in both worlds, and
only token resolution differs.

## Parser corrections (`dvw_patch.py`)

pydatavolley 2.3 could not read this corpus correctly. Three defects, all
patched at import (no fork):

1. **4 of 22 real files failed outright** with `10 columns passed, passed data
   had 12 columns`. `get_set()` builds each row inside a `try` and, on any
   exception, appends a fixed 9 `None`s *without discarding what the try
   already appended*. A row failing on the first field gives 10 values
   (accidentally right); one failing partway gives 12. It is not "5-set
   matches" — it is any set row with some quarter-score fields filled and
   others blank, which is exactly a 15-point deciding set. The handler also
   nulled scores that were present in the file. Fixed by parsing each field
   independently.
2. **`np.NaN` was removed in NumPy 2.0**, so on NumPy ≥ 2 every file failed
   before that mattered. Restored as the alias it always was.
3. **Jersey number 0 was discarded as if it meant "no player".** `read_dv`
   builds `player_number` with `.fillna(0)` and then drops everything equal to
   `"0"` — using one value to mean both "the regex matched nothing" and "the
   number zero". Jersey 0 is legal and used here (Rutgers' #0 plays in both
   Rutgers matches, coded `a00AT+…`), so her `player_number` went `NaN`, and
   since `calculate_skill` nulls `skill` whenever `player_number` is `NaN`,
   **all 161 of her actions vanished from the play-by-play**. Found by decoding
   the raw scout codes independently and diffing: 20/22 files agreed exactly,
   the 2 that didn't were missing exactly her 75 and 86 actions. The repair
   restores the player, the skill, and every skill-derived field (recomputed
   with `read_dv`'s own expressions); files without a jersey-0 player are
   left untouched.

### How the decoding is verified

`test_dvw_patch.py` decodes the raw `[3SCOUT]` lines from scratch, sharing no
code with pydatavolley, and asserts every `(team, jersey, skill, evaluation
code)` count matches — **22/22 files, 21,474 actions, zero delta**. That is the
regression test for all three defects, and it is what caught the third.

Two upstream labels are also wrong, and are documented rather than trusted:
`duration` is really the set's **target score** (exactly 25 for sets 1–4 and 15
for set 5 in 110/110 rows), and the "played" flag is `"True"` on all 110 rows
including sets never played.

## Decisions taken from the data

Every one of these was measured across all 22 files before being coded:

- **Player identity includes the jersey number.** One roster carries the same
  name at two different numbers; keying on name alone merges two players.
- **Game identity includes the date.** Three opponents are played twice.
- **Rally-outcome rows are dropped.** Every `skill="Point"` row has no player;
  formatting it naively produced a phantom `#nan nan` player that appeared in
  all 22 files with 3–5 "sets played".
- **Sets played comes from roster metadata, capped by sets actually contested.**
  Counting actions undercounts (113 cases of on-court-but-no-touch); counting
  the six-player rotation columns scores every **libero 0**, making per-set
  metrics a division by zero. The metadata is right for both — but a lineup
  entered for a set that was never played is stored in the file, so only the
  first `home_setswon + visiting_setswon` columns are counted. With that cap,
  metadata and action counts reconcile exactly (475 exact, 0 violations).
- **The evaluation vocabulary is keyed on `(skill, code)`, not on the code.**
  `#` is an ace on a Serve, a perfect pass on a Reception, a kill on an Attack —
  and some skills never take some codes (`Set` only takes `#`, `-`, `=`).
- **Primitive names are evidence-based.** Verified against rally outcomes:
  Attack `#` wins 100% of rallies (kill), Attack `=` and `/` win 0% (error,
  blocked), Serve `#` 100% (ace), Block `#` 100% (stuff), every `=` is 0%.
  Non-terminal codes sit at 30–70% and are named neutrally rather than given
  outcome words the data doesn't support.

## Layout

| file | role |
|---|---|
| `volley_source.py` | the contract: `Grain`, `FieldSpec`, `SourceSchema`, `Source` |
| `volley_source_dvw.py` | `.dvw` adapter (EVENT grain) |
| `volley_event_spec.py` | `event` / `measure` / metric-`formula` payloads + validators |
| `volley_evaluate.py` | vectorized, source-aware evaluator → the standard tidy cube |
| `volley_seed.py` | the primitives knowledge base + derived metrics |
| `volley_store.py` | match discovery, tree persistence |
| `volley_llm.py` | event-grain metric author (router is reused from the parent) |
| `volley_query.py` | resolution, action execution, pipelines, consolidation |
| `app.py` | Streamlit UI only |

Nothing here modifies the existing CSV app; its suite still passes unchanged.
