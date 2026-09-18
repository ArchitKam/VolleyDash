export type Axis = "Player" | "Game" | "Set" | "Metric";
export const AXES: Axis[] = ["Player", "Game", "Set", "Metric"];

export type CompareOp = ">" | ">=" | "<" | "<=" | "==" | "!=";
export type ReduceHow = "mean" | "sum" | "min" | "max" | "count" | "median";

export type Op =
  | {
      kind: "Slice";
      axis: Axis;
      keep?: string[];
      predicate?: {
        metric: string;
        op: CompareOp;
        threshold: number;
        decided_by_player?: string | null;
      };
    }
  | { kind: "Reduce"; axis: Axis; how: ReduceHow }
  | { kind: "Rank"; axis: Axis; descending?: boolean; limit?: number }
  | { kind: "Compare"; axis: Axis; how?: "mean"; mode: "difference" | "ratio" };

export type Row = {
  Game: string;
  Player: string;
  /** Only present when the source has a Set axis and the request splits by it. */
  Set?: string | undefined;
  Metric: string;
  Value: number | null;
  Note?: string | undefined;
  NUsed?: number | undefined;
};

export type Frame = Row[];

export const REDUCE_HOWS: ReduceHow[] = ["mean", "sum", "min", "max", "count", "median"];
export const COMPARE_OPS: CompareOp[] = [">", ">=", "<", "<=", "==", "!="];

export function isValidOp(raw: unknown): raw is Op {
  if (!raw || typeof raw !== "object") return false;
  const o = raw as Record<string, unknown>;
  const axis = o["axis"];
  if (typeof axis !== "string" || !AXES.includes(axis as Axis)) return false;
  switch (o["kind"]) {
    case "Slice": {
      const keep = o["keep"];
      const pred = o["predicate"] as Record<string, unknown> | undefined;
      if (Array.isArray(keep) && keep.every((k) => typeof k === "string")) return true;
      if (pred && typeof pred["metric"] === "string" && typeof pred["threshold"] === "number") {
        return COMPARE_OPS.includes(pred["op"] as CompareOp);
      }
      return false;
    }
    case "Reduce":
      return REDUCE_HOWS.includes(o["how"] as ReduceHow);
    case "Rank":
      return o["limit"] === undefined || typeof o["limit"] === "number";
    case "Compare":
      return o["mode"] === "difference" || o["mode"] === "ratio";
    default:
      return false;
  }
}

function agg(values: number[], how: ReduceHow): number | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  switch (how) {
    case "sum":
      return values.reduce((a, b) => a + b, 0);
    case "mean":
      return values.reduce((a, b) => a + b, 0) / values.length;
    case "min":
      return sorted[0]!;
    case "max":
      return sorted[sorted.length - 1]!;
    case "count":
      return values.length;
    case "median": {
      const mid = Math.floor(sorted.length / 2);
      return sorted.length % 2 ? sorted[mid]! : (sorted[mid - 1]! + sorted[mid]!) / 2;
    }
  }
}

function otherAxes(axis: Axis): Axis[] {
  return AXES.filter((a) => a !== axis);
}

function groupKey(row: Row, axes: Axis[]): string {
  return axes.map((a) => row[a] ?? "").join("||");
}

export type ExecContext = {
  /** Values for a metric NOT necessarily in the frame, used by Slice predicates. */
  resolveMetricRows?: (metric: string) => Row[];
};

export type ExecResult = { frame: Frame; warnings: string[] };

function applySlice(
  frame: Frame,
  op: Extract<Op, { kind: "Slice" }>,
  ctx: ExecContext,
): ExecResult {
  const warnings: string[] = [];
  if (op.keep && op.keep.length > 0) {
    const keep = new Set(op.keep.map((k) => k.toLowerCase()));
    return {
      frame: frame.filter((r) => keep.has(String(r[op.axis]).toLowerCase())),
      warnings,
    };
  }
  const pred = op.predicate;
  if (!pred) return { frame, warnings };

  const source =
    ctx.resolveMetricRows?.(pred.metric) ?? frame.filter((r) => r.Metric === pred.metric);
  if (source.length === 0) {
    warnings.push(`Could not evaluate the "${pred.metric}" filter — no values available.`);
    return { frame, warnings };
  }
  const test = (v: number | null) => {
    if (v === null) return false;
    switch (pred.op) {
      case ">":
        return v > pred.threshold;
      case ">=":
        return v >= pred.threshold;
      case "<":
        return v < pred.threshold;
      case "<=":
        return v <= pred.threshold;
      case "==":
        return v === pred.threshold;
      case "!=":
        return v !== pred.threshold;
    }
  };

  if (pred.decided_by_player) {
    const gate = source.filter((r) =>
      r.Player.toLowerCase().includes(pred.decided_by_player!.toLowerCase()),
    );
    const allowed = new Set(gate.filter((r) => test(r.Value)).map((r) => String(r[op.axis])));
    return { frame: frame.filter((r) => allowed.has(String(r[op.axis]))), warnings };
  }

  const allowedPerPlayer = new Map<string, Set<string>>();
  for (const r of source) {
    if (!test(r.Value)) continue;
    const set = allowedPerPlayer.get(r.Player) ?? new Set<string>();
    set.add(String(r[op.axis]));
    allowedPerPlayer.set(r.Player, set);
  }
  return {
    frame: frame.filter((r) => allowedPerPlayer.get(r.Player)?.has(String(r[op.axis])) ?? false),
    warnings,
  };
}

function applyReduce(frame: Frame, op: Extract<Op, { kind: "Reduce" }>): ExecResult {
  const keepAxes = otherAxes(op.axis);
  const groups = new Map<string, Row[]>();
  for (const r of frame) {
    const k = groupKey(r, keepAxes);
    groups.set(k, [...(groups.get(k) ?? []), r]);
  }
  const out: Frame = [];
  for (const rows of groups.values()) {
    const values = rows.map((r) => r.Value).filter((v): v is number => v !== null);
    const first = rows[0]!;
    out.push({
      Game: op.axis === "Game" ? `${op.how} of ${rows.length} games` : first.Game,
      Player: op.axis === "Player" ? `${op.how} of ${rows.length} players` : first.Player,
      Metric: op.axis === "Metric" ? `${op.how} of ${rows.length} metrics` : first.Metric,
      Value: agg(values, op.how),
      NUsed: values.length,
      Note:
        values.length < rows.length
          ? `${values.length} of ${rows.length} values used (rest were blank)`
          : undefined,
    });
  }
  return { frame: out, warnings: [] };
}

function applyRank(frame: Frame, op: Extract<Op, { kind: "Rank" }>): ExecResult {
  const desc = op.descending !== false;
  const keepAxes = otherAxes(op.axis);
  const groups = new Map<string, Row[]>();
  for (const r of frame) {
    const k = groupKey(r, keepAxes);
    groups.set(k, [...(groups.get(k) ?? []), r]);
  }
  const minRows = Math.max(3, op.limit ?? 0);
  const out: Frame = [];
  for (const rows of groups.values()) {
    const withValues = rows.filter((r) => r.Value !== null);
    const withoutValues = rows.filter((r) => r.Value === null);
    withValues.sort((a, b) => (desc ? b.Value! - a.Value! : a.Value! - b.Value!));
    const ordered = [...withValues, ...withoutValues];
    out.push(...ordered.slice(0, Math.min(ordered.length, minRows)));
  }
  return { frame: out, warnings: [] };
}

function applyCompare(frame: Frame, op: Extract<Op, { kind: "Compare" }>): ExecResult {
  const keepAxes = otherAxes(op.axis);
  const baselines = new Map<string, number | null>();
  const groups = new Map<string, Row[]>();
  for (const r of frame) {
    const k = groupKey(r, keepAxes);
    groups.set(k, [...(groups.get(k) ?? []), r]);
  }
  for (const [k, rows] of groups) {
    const values = rows.map((r) => r.Value).filter((v): v is number => v !== null);
    baselines.set(k, agg(values, op.how ?? "mean"));
  }
  const warnings: string[] = [];
  const out = frame.map((r) => {
    const base = baselines.get(groupKey(r, keepAxes));
    if (r.Value === null || base === null || base === undefined) {
      return { ...r, Value: null, Note: r.Note ?? "No baseline available" };
    }
    if (op.mode === "ratio") {
      if (base === 0) {
        warnings.push("A zero baseline made some ratio comparisons blank.");
        return { ...r, Value: null, Note: "Baseline was zero" };
      }
      return { ...r, Value: r.Value / base };
    }
    return { ...r, Value: r.Value - base };
  });
  return { frame: out, warnings };
}

export function runPipeline(frame: Frame, pipeline: Op[], ctx: ExecContext = {}): ExecResult {
  let current = frame;
  const warnings: string[] = [];
  for (const op of pipeline) {
    let res: ExecResult;
    switch (op.kind) {
      case "Slice":
        res = applySlice(current, op, ctx);
        break;
      case "Reduce":
        res = applyReduce(current, op);
        break;
      case "Rank":
        res = applyRank(current, op);
        break;
      case "Compare":
        res = applyCompare(current, op);
        break;
    }
    current = res.frame;
    warnings.push(...res.warnings);
  }
  return { frame: current, warnings: [...new Set(warnings)] };
}

export function describePipeline(pipeline: Op[]): string {
  return pipeline
    .map((op) => {
      switch (op.kind) {
        case "Slice":
          if (op.keep?.length) return `keep only ${op.axis}: ${op.keep.join(", ")}`;
          if (op.predicate) {
            const who = op.predicate.decided_by_player
              ? ` (decided by ${op.predicate.decided_by_player})`
              : "";
            return `keep ${op.axis}s where ${op.predicate.metric} ${op.predicate.op} ${op.predicate.threshold}${who}`;
          }
          return `slice ${op.axis}`;
        case "Reduce":
          return `take the ${op.how} across ${op.axis}`;
        case "Rank":
          return `rank by value ${op.descending === false ? "ascending" : "descending"} across ${op.axis}${
            op.limit ? ` (top ${op.limit})` : ""
          }`;
        case "Compare":
          return `express each value as a ${op.mode} against the ${op.how ?? "mean"} over ${op.axis}`;
      }
    })
    .join(", then ");
}
