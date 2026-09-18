import { COLUMN_SCHEMA } from "./columns";
import { isPlayerRow, playerName, type CsvRow } from "./csv";
import { emptySchema, type FactRow, type MeasureRow, type Source, type SourceSchema } from "./source";

export type CsvIdentityRow = { Player: string; Game: string; row: CsvRow };

/** Wraps the existing recruiting logic — one row is a player's whole-match totals. */
export interface CsvSource extends Source {
  grain: "measure";
  csvRows(): CsvIdentityRow[];
}

const IDENTITY_FIELDS = { Player: "Name", Game: "Game" } as const;

export function buildCsvSource(games: Array<{ opponent: string; rows: CsvRow[] }>): CsvSource {
  const identityRows: CsvIdentityRow[] = [];
  const players = new Set<string>();
  const gameLabels: string[] = [];

  for (const game of games) {
    if (!gameLabels.includes(game.opponent)) gameLabels.push(game.opponent);
    for (const row of game.rows.filter(isPlayerRow)) {
      const player = playerName(row);
      players.add(player);
      identityRows.push({ Player: player, Game: game.opponent, row });
    }
  }

  const facts: FactRow[] = identityRows.map((r) => ({
    ...r.row,
    Name: r.Player,
    Game: r.Game,
  }));

  return {
    grain: "measure",
    schema: deriveSchema(),
    facts: () => facts,
    // Every numeric column already IS a measure at this grain.
    measures: () => facts as unknown as MeasureRow[],
    measureAggregation: () => "sum",
    identityFields: () => ({ ...IDENTITY_FIELDS }),
    axes: () => [...Object.keys(IDENTITY_FIELDS), "Metric"],
    gameLabels: () => gameLabels,
    setLabels: () => [],
    playerLabels: () => [...players].sort((a, b) => a.localeCompare(b)),
    warnings: () => [],
    csvRows: () => identityRows,
  };
}

function deriveSchema(): SourceSchema {
  const schema = emptySchema();
  schema.fields["Name"] = {
    name: "Name",
    role: "identity",
    description: "Player, as recorded in the export",
    values: null,
  };
  schema.fields["Game"] = {
    name: "Game",
    role: "identity",
    description: "Match this row's totals belong to",
    values: null,
  };
  // Numeric columns are measures, not filters — there is nothing to filter at this grain.
  for (const col of COLUMN_SCHEMA) {
    schema.fields[col.name] = {
      name: col.name,
      role: "measure",
      description: col.description,
      values: null,
    };
  }
  return schema;
}
