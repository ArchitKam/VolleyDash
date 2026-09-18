export type ColumnType = "COUNT" | "PERCENTAGE" | "RATE" | "RATING";

export type ColumnDef = {
  name: string;
  group: string;
  type: ColumnType;
  description: string;
};

export const SKILL_GROUPS = [
  "Attack",
  "Serve",
  "Receive",
  "Set",
  "Dig",
  "Block",
  "Points",
  "Sets",
] as const;

export const COLUMN_SCHEMA: ColumnDef[] = [
  { name: "Attack K", group: "Attack", type: "COUNT", description: "Kills" },
  { name: "Attack E", group: "Attack", type: "COUNT", description: "Attack errors" },
  { name: "Attack TA", group: "Attack", type: "COUNT", description: "Total attack attempts" },
  {
    name: "Attack Atk%",
    group: "Attack",
    type: "PERCENTAGE",
    description: "(K-E)/TA hitting efficiency",
  },
  { name: "Attack K/S", group: "Attack", type: "RATE", description: "Kills per set played" },

  { name: "Serve SA", group: "Serve", type: "COUNT", description: "Service aces" },
  { name: "Serve SE", group: "Serve", type: "COUNT", description: "Service errors" },
  { name: "Serve TA", group: "Serve", type: "COUNT", description: "Total serve attempts" },
  {
    name: "Serve Pct",
    group: "Serve",
    type: "PERCENTAGE",
    description: "Serve success percentage",
  },
  { name: "Serve Eff", group: "Serve", type: "PERCENTAGE", description: "Serve efficiency" },
  { name: "Serve Rtg.", group: "Serve", type: "RATING", description: "Composite serve rating" },

  { name: "Receive 3", group: "Receive", type: "COUNT", description: "Perfect passes" },
  { name: "Receive 2", group: "Receive", type: "COUNT", description: "Good passes" },
  { name: "Receive 1", group: "Receive", type: "COUNT", description: "Poor passes" },
  {
    name: "Receive 0",
    group: "Receive",
    type: "COUNT",
    description: "Errors/lowest quality",
  },
  { name: "Receive TA", group: "Receive", type: "COUNT", description: "Total receive attempts" },
  {
    name: "Receive Pass%",
    group: "Receive",
    type: "PERCENTAGE",
    description: "Passing quality percentage",
  },

  { name: "Set Ast", group: "Set", type: "COUNT", description: "Assists" },
  { name: "Set TA", group: "Set", type: "COUNT", description: "Total set attempts" },
  { name: "Set SE", group: "Set", type: "COUNT", description: "Setting errors" },
  { name: "Set 3", group: "Set", type: "COUNT", description: "Perfect sets" },
  { name: "Set 2", group: "Set", type: "COUNT", description: "Good sets" },
  { name: "Set 1", group: "Set", type: "COUNT", description: "Poor sets" },
  { name: "Set 0", group: "Set", type: "COUNT", description: "Set errors" },
  { name: "Set Rtg.", group: "Set", type: "RATING", description: "Composite setting rating" },

  { name: "Dig DS", group: "Dig", type: "COUNT", description: "Digs succeeded" },
  { name: "Dig DE", group: "Dig", type: "COUNT", description: "Dig errors" },

  { name: "Block BS", group: "Block", type: "COUNT", description: "Solo blocks" },
  { name: "Block BA", group: "Block", type: "COUNT", description: "Assisted blocks" },
  { name: "Block BE", group: "Block", type: "COUNT", description: "Block errors" },
  { name: "Block B/S", group: "Block", type: "RATE", description: "Blocks per set" },

  {
    name: "Points Pts +/-",
    group: "Points",
    type: "COUNT",
    description: "Plus/minus point differential",
  },

  {
    name: "Sets Sets Played",
    group: "Sets",
    type: "COUNT",
    description: "Sets played in this match",
  },
];

export const COLUMN_BY_NAME = new Map(COLUMN_SCHEMA.map((c) => [c.name, c]));

export function columnsForGroup(group: string): ColumnDef[] {
  return COLUMN_SCHEMA.filter((c) => c.group === group);
}
