import { COLUMN_BY_NAME, type ColumnType } from "./columns";
import { nodeExpression, type GraphNode } from "./tree";

export const PLAYER_PALETTE = [
  "#E21833",
  "#B8860B",
  "#8B0000",
  "#DAA520",
  "#FFFFFF",
  "#A9A9A9",
  "#FF6B6B",
  "#F0C300",
];

/** Stable color per player: assigned by sorted player name. */
export function buildPlayerColors(players: string[]): Record<string, string> {
  const sorted = [...new Set(players)].sort((a, b) => a.localeCompare(b));
  const out: Record<string, string> = {};
  sorted.forEach((p, i) => {
    out[p] = PLAYER_PALETTE[i % PLAYER_PALETTE.length]!;
  });
  return out;
}

function underlyingType(leaf?: GraphNode): ColumnType | null {
  if (!leaf) return null;
  if (leaf.kind === "column") return COLUMN_BY_NAME.get(leaf.column_ref ?? "")?.type ?? null;
  const expr = nodeExpression(leaf);
  const refs = [...expr.matchAll(/\[([^\]]+)\]/g)].map((m) => m[1] ?? "");
  if (refs.length === 1) return COLUMN_BY_NAME.get(refs[0]!)?.type ?? null;
  return null;
}

export type NumberFormat = "percent1" | "fixed2" | "int";

export function formatKindFor(label: string, leaf?: GraphNode): NumberFormat {
  const l = label.toLowerCase();
  // Counted actions and roster measures are whole numbers by construction.
  if (leaf?.kind === "event" || leaf?.kind === "measure") return "int";
  const type = underlyingType(leaf);
  const percentish =
    l.includes("%") ||
    l.includes("pct") ||
    l.includes("percentage") ||
    l.includes("efficiency") ||
    (leaf?.kind === "formula" && l.includes("rate")) ||
    (type === "RATE" && l.includes("success")) ||
    type === "PERCENTAGE";
  if (percentish) return "percent1";
  if (
    type === "RATE" ||
    type === "RATING" ||
    l.includes("per set") ||
    l.includes("/s") ||
    l.includes("rating") ||
    l.includes("rtg")
  ) {
    return "fixed2";
  }
  if (type === "COUNT") return "int";
  return "fixed2";
}

export function formatValue(value: number | null | undefined, kind: NumberFormat): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  switch (kind) {
    case "percent1":
      return `${(Math.abs(value) <= 1 ? value * 100 : value).toFixed(1)}%`;
    case "int":
      return value.toFixed(0);
    case "fixed2":
      return value.toFixed(2);
  }
}
