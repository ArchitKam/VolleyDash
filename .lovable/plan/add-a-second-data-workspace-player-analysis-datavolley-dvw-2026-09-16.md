# Add a second data workspace: Player Analysis (DataVolley .dvw)

Adds a toggleable workspace alongside the existing Recruiting workspace. Both reuse the same
knowledge-tree engine, operations algebra, charts, LLM router and GitHub persistence; each gets its
own tree file and its own definition of a metric because the two file formats differ in grain
(one row = a player's match totals vs. one row = a single action).

## 1. Source abstraction (`src/lib/source.ts`)

`Grain`, `FieldRole`, `FieldSpec`, `SourceSchema` (fields + `dependentValues` keyed
`` `${governing}|${dependent}` ``), and the `Source` interface exactly as specified: `facts()`,
`measures()`, `identityFields()`, `axes()` (identity axes + `Metric`), `measureAggregation()`,
`gameLabels()`. Schema vocabularies are always derived from the loaded files, never hardcoded.

- `src/lib/source.csv.ts` — wraps the existing recruiting logic unchanged: grain `measure`,
  identity `{Player, Game}`, `measures() === facts()`, every numeric CSV column a `measure` field.
- `src/lib/source.dvw.ts` — grain `event`, identity `{Player, Game, Set}`, facts = decoded actions,
  measures = per-(player, game, set) `sets_played` rows, schema derived from observed values
  including the per-skill evaluation-code table.

## 2. DVW parser (`src/lib/dvw.ts`, written from scratch)

Sections located by exact `[3MATCH]` / `[3TEAMS]` / `[3PLAYERS-H]` / `[3PLAYERS-V]` / `[3SET]` /
`[3SCOUT]` matches, parsed per the spec. Baked-in fixes:

- `[3SET]` fields parsed **independently** so a partially scored (15-point) set can't corrupt row width;
  stop early on a `[` line; field 5 kept but named target score; the "played" flag ignored in favour of
  populated score pairs.
- Jersey number: a genuine `0` is kept; "position isn't digits" is a separate `null` case — never conflated.
- `code` decoded at fixed offsets (team, jersey, skill letter, evaluation code restricted to `# + ! - / =`,
  attack/set codes, set type, zones, num-players, custom code after the last `~`). `code[1] === 'p'` →
  `Point` score-marker row with no player.
- Derived fields: `rally_number`, `team`, `point_won_by` (backfilled per set from the next score marker),
  `serving_team` / `receiving_team` (forward-filled within set+rally), `point_phase`, `attack_phase`
  (straightforward version), `possession_number`, `home_team_score` / `visiting_team_score`.
- Rows with no resolvable skill (including `Point` markers, subs, timeouts) dropped before facts.
- Labels: player `#{jersey} {name}`; game `{shortened opponent} ({Mon D})` with the listed shortening
  rules and explicit overrides; set `Set {n}`; games ordered by parsed date, never by label or filename.
- `sets_played` from roster participation columns, capped at `home_sets_won + visiting_sets_won`,
  emitted one row per participated set — never counted from actions or rotation slots.

## 3. Metric model + evaluator

- `MetricSpec` gains `{ kind: "event", where, aggregate, field? }` and `{ kind: "measure", measure }`;
  `GraphNode` carries the extra payload; persistence round-trips all four kinds.
- Validation against the live schema: unknown field / unobserved value rejected, and a
  (skill, evaluation_code) pair rejected via `dependentValues` with a message naming the codes actually
  seen for that skill.
- `src/lib/evaluateEvent.ts` — vectorized evaluation over the whole (Player, Game[, Set]) universe:
  universe = roster-participation tuples ∪ tuples with at least one action; event metrics mask + group +
  count, missing filled with a real `0`; measures aggregated per `measureAggregation`; formulas do
  array-wise bracket substitution with divide-by-zero → blank. Nested-metric resolution and
  circular-reference detection are generalized from the CSV path, not duplicated.
- A player who did not play produces **no row** (renders blank/em-dash); a player who played and
  recorded nothing produces `0`.

## 4. Seed tree for the player-analysis workspace

`buildDvwSeedTree(schema)` builds the exact eight branches and primitive leaves listed in the spec,
skipping any primitive the loaded files don't actually contain, then adds the derived formula leaves
once the primitives exist. Branch labelled `Receive` while the skill value stays `Reception`.
Coach-style aliases on every leaf; `authored_by: "system:dvw_import"`.

## 5. LLM event-metric author + router

- New event-metric authoring prompt built from the live schema (per-skill codes, other dimension fields
  and their observed values, committed metric labels, branch names), returning the event / formula /
  unmatched JSON shapes. Everything it returns is re-validated by the mechanical validator.
- Router: optional `set_hint`, and `supportsSets` passed to the prompt builder so a source without a Set
  axis is never asked about sets.

## 6. Execution layer

`compute.ts` generalized to run against any `Source`, dispatching on grain:

- Player resolution gains a jersey-number tier (`#7` or `7` matches a label starting `#7 `), for both workspaces.
- A game hint matching multiple games returns **all** of them.
- Set scoping (hint or multiselect) narrows the data so per-set denominators rebase, distinct from a
  pipeline splitting by the Set axis; both may apply together.
- No Set axis but a set was requested → ignored plus the exact warning text from the spec.
- Category aggregation unchanged.

## 7. Persistence + file access (`data.functions.ts`)

`volley_kb_data.json` with top-level key `"volley"` (a file carrying the other workspace's key fails
loudly and seeds fresh), same GET-sha-then-PUT save. New `listDvwFiles` (returns `{path, filename}`
only) and `getDvwFile` (raw text). Parsing and per-path caching happen client-side in the store,
mirroring the CSV cache.

## 8. UI

- Workspace select above the tabs; switching resets every workspace-scoped piece of state (query,
  decomposition, results, chart encodings, player filter, games and sets selections, KB selection/expansion,
  wizard state).
- Team select shown for player analysis, defaulting to the team appearing in the most files; changing it
  resets the same state and re-derives roster + schema.
- Sets multiselect rendered only when the source has a Set axis.
- Assembly warnings as a banner; a zero-branch tree as a hard error.
- Header caption per workspace, and per-workspace table formatting (event/measure = whole numbers,
  formula = percentage when the label reads like a rate/efficiency else 2 decimals, blank ≠ 0).

## Verification note

There are no real `.dvw` files in the repo yet, so the parser, schema derivation, seed-tree validation
and event evaluator will be built to spec and unit-tested against synthetic fixtures I write from the
grammar above. Everything that depends on real file quirks stays unverified until you add files to
`dvw/`; I'll list those parts explicitly in the summary.
