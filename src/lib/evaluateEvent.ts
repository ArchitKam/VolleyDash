import { evalArithmetic } from "./evaluate";
import type { FactRow, MeasureRow, Source } from "./source";
import { bracketRefs, findMetric, type GraphTree } from "./tree";

const SEP = "\u0000";

export type IdentityKey = { Player: string; Game: string; Set?: string | undefined };

export type Series = Map<string, number | null>;

export type EvalContext = {
  source: Source;
  tree: GraphTree;
  bySet: boolean;
  /** Identity tuples that legitimately exist: participation rows plus rows with actions. */
  universe: string[];
  facts: FactRow[];
  measures: MeasureRow[];
  cache: Map<string, Series>;
};

export function encodeKey(k: IdentityKey, bySet: boolean): string {
  return bySet ? [k.Player, k.Game, k.Set ?? ""].join(SEP) : [k.Player, k.Game].join(SEP);
}

export function decodeKey(key: string): IdentityKey {
  const [Player = "", Game = "", Set] = key.split(SEP);
  return Set === undefined ? { Player, Game } : { Player, Game, Set };
}

/**
 * Builds the evaluation universe once. Scoping to games/sets NARROWS the underlying
 * data (so per-set rate denominators rebase), which is a different thing from a
 * pipeline asking to split the cube by the Set axis.
 */
export function createEvalContext(opts: {
  source: Source;
  tree: GraphTree;
  games?: string[] | null;
  sets?: string[] | null;
  bySet?: boolean;
}): EvalContext {
  const { source, tree } = opts;
  const idf = source.identityFields();
  const playerField = idf["Player"] ?? "Player";
  const gameField = idf["Game"] ?? "Game";
  const setField = idf["Set"];
  const bySet = !!opts.bySet && !!setField;

  const games = opts.games && opts.games.length > 0 ? new Set(opts.games) : null;
  const sets = opts.sets && opts.sets.length > 0 ? new Set(opts.sets) : null;

  const facts = source.facts().filter((row) => {
    if (games && !games.has(String(row[gameField] ?? ""))) return false;
    if (sets && setField && !sets.has(String(row[setField] ?? ""))) return false;
    return true;
  });
  const measures = source.measures().filter((m) => {
    if (games && !games.has(m.Game)) return false;
    if (sets && m.Set !== undefined && !sets.has(m.Set)) return false;
    return true;
  });

  const universe = new Set<string>();
  for (const m of measures) {
    universe.add(encodeKey({ Player: m.Player, Game: m.Game, Set: m.Set }, bySet));
  }
  for (const row of facts) {
    const player = String(row[playerField] ?? "");
    if (!player) continue;
    universe.add(
      encodeKey(
        {
          Player: player,
          Game: String(row[gameField] ?? ""),
          Set: setField ? String(row[setField] ?? "") : undefined,
        },
        bySet,
      ),
    );
  }

  return { source, tree, bySet, universe: [...universe], facts, measures, cache: new Map() };
}

function keyOfFact(ctx: EvalContext, row: FactRow): string {
  const idf = ctx.source.identityFields();
  const setField = idf["Set"];
  return encodeKey(
    {
      Player: String(row[idf["Player"] ?? "Player"] ?? ""),
      Game: String(row[idf["Game"] ?? "Game"] ?? ""),
      Set: ctx.bySet && setField ? String(row[setField] ?? "") : undefined,
    },
    ctx.bySet,
  );
}

function matches(row: FactRow, where: Record<string, string | string[]>): boolean {
  for (const [field, raw] of Object.entries(where)) {
    const value = row[field];
    const text = value === null || value === undefined ? "" : String(value);
    if (Array.isArray(raw)) {
      if (!raw.includes(text)) return false;
    } else if (text !== raw) return false;
  }
  return true;
}

/** Vectorized: one metric computed across the whole universe at once. */
export function evaluateMetric(ctx: EvalContext, label: string, seen: string[] = []): Series {
  const cached = ctx.cache.get(label);
  if (cached) return cached;
  if (seen.includes(label)) {
    throw new Error(`Circular reference detected: ${[...seen, label].join(" → ")}`);
  }
  const node = findMetric(ctx.tree, label);
  if (!node) throw new Error(`Unknown reference "${label}".`);

  const out: Series = new Map();

  if (node.kind === "event") {
    const distinctSets = new Map<string, Set<string>>();
    const counts = new Map<string, number>();
    for (const row of ctx.facts) {
      if (!matches(row, node.where ?? {})) continue;
      const key = keyOfFact(ctx, row);
      if (node.aggregate === "count_distinct") {
        const field = node.field ?? "";
        const set = distinctSets.get(key) ?? new Set<string>();
        set.add(String(row[field] ?? ""));
        distinctSets.set(key, set);
      } else {
        counts.set(key, (counts.get(key) ?? 0) + 1);
      }
    }
    // A player who played but recorded none of this gets a real, countable 0 — never blank.
    for (const key of ctx.universe) {
      out.set(
        key,
        node.aggregate === "count_distinct"
          ? (distinctSets.get(key)?.size ?? 0)
          : (counts.get(key) ?? 0),
      );
    }
  } else if (node.kind === "measure") {
    const name = node.measure ?? "";
    const how = ctx.source.measureAggregation(name);
    const sums = new Map<string, number>();
    const counts = new Map<string, number>();
    for (const m of ctx.measures) {
      if (m.measure !== name) continue;
      const key = encodeKey({ Player: m.Player, Game: m.Game, Set: m.Set }, ctx.bySet);
      sums.set(key, (sums.get(key) ?? 0) + m.value);
      counts.set(key, (counts.get(key) ?? 0) + 1);
    }
    for (const key of ctx.universe) {
      const sum = sums.get(key) ?? 0;
      out.set(key, how === "mean" ? (counts.get(key) ? sum / counts.get(key)! : 0) : sum);
    }
  } else if (node.kind === "formula") {
    const expr = node.formula_expr ?? "";
    const refs = [...new Set(bracketRefs(expr))];
    const parts = refs.map((ref) => ({
      ref,
      series: evaluateMetric(ctx, ref, [...seen, label]),
    }));
    for (const key of ctx.universe) {
      let substituted = expr;
      let blank = false;
      for (const { ref, series } of parts) {
        const v = series.get(key);
        if (v === null || v === undefined || Number.isNaN(v)) {
          blank = true;
          break;
        }
        substituted = substituted.split(`[${ref}]`).join(`(${v})`);
      }
      if (blank) {
        out.set(key, null);
        continue;
      }
      try {
        const value = evalArithmetic(substituted);
        // Division by zero is blank, never Infinity.
        out.set(key, Number.isFinite(value) ? value : null);
      } catch {
        out.set(key, null);
      }
    }
  } else {
    throw new Error(`"${label}" isn't a metric this data can compute.`);
  }

  ctx.cache.set(label, out);
  return out;
}
