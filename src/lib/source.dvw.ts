import {
  compareMatchesByDate,
  gameLabelFor,
  setLabel,
  type DvwMatch,
} from "./dvw";
import {
  emptySchema,
  noteDependent,
  type FactRow,
  type MeasureRow,
  type Source,
  type SourceSchema,
} from "./source";

/** Categorical action fields offered as filters, in the order a coach reads them. */
const DIMENSIONS: Array<{ name: string; description: string }> = [
  { name: "skill", description: "Which action this row is (Serve, Reception, Set, Attack, Dig, Block, Freeball)" },
  { name: "evaluation_code", description: "Outcome grade of the action — its meaning depends on the skill" },
  { name: "attack_code", description: "Attack call/combination code (attack rows only)" },
  { name: "set_code", description: "Set call code (set rows only)" },
  { name: "set_type", description: "Set type (set rows only)" },
  { name: "start_zone", description: "Court zone the action started in" },
  { name: "end_zone", description: "Court zone the action ended in" },
  { name: "end_subzone", description: "Sub-zone the action ended in" },
  { name: "num_players_numeric", description: "Number of blockers (attack rows only)" },
  { name: "point_phase", description: "Whether this player's team was serving or receiving in the rally" },
  { name: "attack_phase", description: "Reception, SO-Transition or BP-Transition (attack rows only)" },
  { name: "point_won_by", description: "Team that won the rally this action belongs to" },
  { name: "serving_team", description: "Team that served the rally" },
  { name: "receiving_team", description: "Team that received the rally" },
  { name: "team", description: "Team that performed the action" },
  { name: "custom_code", description: "Scout's free-text custom code" },
];

const IDENTITY_FIELDS = {
  Player: "player_label",
  Game: "match_label",
  Set: "set_label",
} as const;

export type DvwSourceResult = { source: Source; warnings: string[] };

export function buildDvwSource(matches: DvwMatch[], teamOfInterest: string | null): DvwSourceResult {
  const warnings: string[] = [];
  for (const m of matches) warnings.push(...m.warnings);

  const relevant = (
    teamOfInterest
      ? matches.filter((m) => m.home_team === teamOfInterest || m.visiting_team === teamOfInterest)
      : matches
  )
    .slice()
    .sort(compareMatchesByDate);

  if (matches.length > 0 && relevant.length === 0 && teamOfInterest) {
    warnings.push(`No loaded match file records actions for "${teamOfInterest}".`);
  }

  const facts: FactRow[] = [];
  const measures: MeasureRow[] = [];
  const gameLabels: string[] = [];
  const setNumbers = new Set<number>();
  const players = new Set<string>();

  for (const match of relevant) {
    const label = gameLabelFor(match, teamOfInterest);
    if (!gameLabels.includes(label)) gameLabels.push(label);
    const side =
      teamOfInterest && match.visiting_team === teamOfInterest
        ? "visiting"
        : teamOfInterest && match.home_team === teamOfInterest
          ? "home"
          : null;

    for (const a of match.actions) {
      if (side && a.team_side !== side) continue;
      if (!a.player_label || a.set_number === null) continue;
      setNumbers.add(a.set_number);
      players.add(a.player_label);
      facts.push({
        player_label: a.player_label,
        match_label: label,
        set_label: setLabel(a.set_number),
        skill: a.skill,
        evaluation_code: a.evaluation_code,
        attack_code: a.attack_code,
        set_code: a.set_code,
        set_type: a.set_type,
        start_zone: a.start_zone,
        end_zone: a.end_zone,
        end_subzone: a.end_subzone,
        num_players_numeric: a.num_players_numeric,
        point_phase: a.point_phase,
        attack_phase: a.attack_phase,
        point_won_by: a.point_won_by,
        serving_team: a.serving_team,
        receiving_team: a.receiving_team,
        team: a.team,
        custom_code: a.custom_code,
        rally_number: a.rally_number,
        possession_number: a.possession_number,
      });
    }

    // Sets played comes from roster participation columns, never from counting actions
    // (a player on court who never touched the ball would undercount) and never from the
    // six on-court rotation slots (a libero would read zero).
    const setsInMatch = Math.max(0, match.home_sets_won + match.visiting_sets_won);
    for (const p of match.players) {
      if (side && p.side !== side) continue;
      if (!p.player_label) continue;
      players.add(p.player_label);
      // Only the first `setsInMatch` columns: a scout can pre-enter the next set's lineup.
      p.participation.slice(0, setsInMatch).forEach((marker, idx) => {
        if (!marker.trim()) return;
        setNumbers.add(idx + 1);
        measures.push({
          Player: p.player_label,
          Game: label,
          Set: setLabel(idx + 1),
          measure: "sets_played",
          value: 1,
        });
      });
    }
  }

  const schema = deriveSchema(facts);
  const setLabels = [...setNumbers].sort((a, b) => a - b).map(setLabel);
  const playerLabels = [...players].sort((a, b) => a.localeCompare(b));

  const source: Source = {
    grain: "event",
    schema,
    facts: () => facts,
    identityFields: () => ({ ...IDENTITY_FIELDS }),
    axes: () => [...Object.keys(IDENTITY_FIELDS), "Metric"],
    measures: () => measures,
    measureAggregation: () => "sum",
    gameLabels: () => gameLabels,
    setLabels: () => setLabels,
    playerLabels: () => playerLabels,
    warnings: () => warnings,
  };
  return { source, warnings };
}

/** Vocabularies come from the loaded files only — nothing is taken from a format spec. */
function deriveSchema(facts: FactRow[]): SourceSchema {
  const schema = emptySchema();
  schema.fields["player_label"] = {
    name: "player_label",
    role: "identity",
    description: "Player, as jersey number plus name",
    values: null,
  };
  schema.fields["match_label"] = {
    name: "match_label",
    role: "identity",
    description: "Match, as opponent plus date",
    values: null,
  };
  schema.fields["set_label"] = {
    name: "set_label",
    role: "identity",
    description: "Set within the match",
    values: null,
  };

  const observed = new Map<string, Set<string>>();
  for (const dim of DIMENSIONS) observed.set(dim.name, new Set<string>());

  for (const row of facts) {
    for (const dim of DIMENSIONS) {
      const v = row[dim.name];
      if (v === null || v === undefined || v === "") continue;
      observed.get(dim.name)!.add(String(v));
    }
    const skill = row["skill"];
    if (skill === null || skill === undefined || skill === "") continue;
    for (const dep of ["evaluation_code", "attack_code", "set_code", "set_type", "attack_phase"]) {
      const v = row[dep];
      if (v === null || v === undefined || v === "") continue;
      noteDependent(schema, "skill", String(skill), dep, String(v));
    }
  }

  for (const dim of DIMENSIONS) {
    const values = observed.get(dim.name)!;
    if (values.size === 0) continue; // never offer a filter on something these files don't record
    schema.fields[dim.name] = {
      name: dim.name,
      role: "dimension",
      description: dim.description,
      values,
    };
  }

  schema.fields["sets_played"] = {
    name: "sets_played",
    role: "measure",
    description: "Sets the player participated in, from the match roster",
    values: null,
  };
  return schema;
}
