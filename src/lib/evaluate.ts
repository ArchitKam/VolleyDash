import { COLUMN_BY_NAME } from "./columns";
import { isPlayerRow, playerName, type CsvRow } from "./csv";
import { bracketRefs, findMetric, nodeExpression, type GraphNode, type GraphTree } from "./tree";

/* -------- safe arithmetic evaluator -------- */

export function evalArithmetic(expr: string): number {
  const src = expr.trim();
  if (!/^[0-9+\-*/().\s]*$/.test(src)) {
    throw new Error("Expression contains characters that are not allowed.");
  }
  let i = 0;
  const peek = () => src[i];
  const skip = () => {
    while (i < src.length && /\s/.test(src[i] ?? "")) i++;
  };
  function parseExpr(): number {
    let v = parseTerm();
    for (;;) {
      skip();
      const c = peek();
      if (c === "+") {
        i++;
        v += parseTerm();
      } else if (c === "-") {
        i++;
        v -= parseTerm();
      } else return v;
    }
  }
  function parseTerm(): number {
    let v = parseUnary();
    for (;;) {
      skip();
      const c = peek();
      if (c === "*") {
        i++;
        v *= parseUnary();
      } else if (c === "/") {
        i++;
        const d = parseUnary();
        v = d === 0 ? NaN : v / d;
      } else return v;
    }
  }
  function parseUnary(): number {
    skip();
    const c = peek();
    if (c === "-") {
      i++;
      return -parseUnary();
    }
    if (c === "+") {
      i++;
      return parseUnary();
    }
    return parsePrimary();
  }
  function parsePrimary(): number {
    skip();
    if (peek() === "(") {
      i++;
      const v = parseExpr();
      skip();
      if (peek() !== ")") throw new Error("Unbalanced parentheses in expression.");
      i++;
      return v;
    }
    const start = i;
    while (i < src.length && /[0-9.]/.test(src[i] ?? "")) i++;
    if (start === i) throw new Error("Malformed expression.");
    const n = Number(src.slice(start, i));
    if (!Number.isFinite(n)) throw new Error("Malformed number in expression.");
    return n;
  }
  const value = parseExpr();
  skip();
  if (i !== src.length) throw new Error("Malformed expression.");
  return value;
}

/* -------- reference resolution -------- */

export type RefValue = { value: number | null; blankRefs: string[] };

function rawColumnValue(row: CsvRow, column: string): number | null {
  const raw = (row[column] ?? "").trim();
  if (raw === "" || raw === "-" || raw === "—") return null;
  const cleaned = raw.replace(/%/g, "").replace(/,/g, "");
  const n = Number(cleaned);
  return Number.isFinite(n) ? n : null;
}

/**
 * Resolve one bracket reference (a raw column or another metric's label) against a row.
 * Recursive with circular-reference detection.
 */
function resolveRef(
  name: string,
  row: CsvRow,
  tree: GraphTree,
  seen: string[],
  fallbackZero: boolean,
  blanks: string[],
): number | null {
  if (COLUMN_BY_NAME.has(name)) {
    const v = rawColumnValue(row, name);
    if (v === null) {
      blanks.push(name);
      return fallbackZero ? 0 : null;
    }
    return v;
  }
  const found = findMetric(tree, name);
  if (!found) throw new Error(`Unknown reference "${name}".`);
  if (seen.includes(name)) {
    throw new Error(`Circular reference detected: ${[...seen, name].join(" → ")}`);
  }
  return evalLeafForRow(found, row, tree, [...seen, name], fallbackZero, blanks);
}

function evalLeafForRow(
  leaf: GraphNode,
  row: CsvRow,
  tree: GraphTree,
  seen: string[],
  fallbackZero: boolean,
  blanks: string[],
): number | null {
  if (leaf.kind === "column") {
    return resolveRef(leaf.column_ref ?? "", row, tree, seen, fallbackZero, blanks);
  }
  const expr = leaf.formula_expr ?? "";
  let substituted = expr;
  for (const ref of bracketRefs(expr)) {
    const v = resolveRef(ref, row, tree, seen, fallbackZero, blanks);
    if (v === null) return null;
    substituted = substituted.split(`[${ref}]`).join(`(${v})`);
  }
  const out = evalArithmetic(substituted);
  return Number.isFinite(out) ? out : null;
}

/** Full-query evaluation: no fallback zeros. */
export function evaluateLeafStrict(
  leaf: GraphNode,
  row: CsvRow,
  tree: GraphTree,
): { value: number | null; blanks: string[] } {
  const blanks: string[] = [];
  const value = evalLeafForRow(leaf, row, tree, [], false, blanks);
  return { value, blanks };
}

/* -------- worked example -------- */

export type WorkedExample = {
  player: string;
  value: number | null;
  substituted: string;
  fallbackZeroed: string[];
  note?: string | undefined;
  error?: string | undefined;
};

export function refsOfExpression(expr: string, tree: GraphTree): string[] {
  const direct = bracketRefs(expr);
  if (direct.length > 0) return direct;
  return COLUMN_BY_NAME.has(expr.trim()) ? [expr.trim()] : [];
}

function countBlanks(row: CsvRow, refs: string[], tree: GraphTree): number {
  let n = 0;
  for (const ref of refs) {
    if (COLUMN_BY_NAME.has(ref)) {
      if (rawColumnValue(row, ref) === null) n++;
    } else {
      const found = findMetric(tree, ref);
      if (found) n += countBlanks(row, refsOfExpression(nodeExpression(found), tree), tree);
    }
  }
  return n;
}

/** Pick one real player row and demonstrate the expression against it. */
export function computeWorkedExample(
  expr: string,
  rows: CsvRow[],
  tree: GraphTree,
  description = "",
): WorkedExample | null {
  const players = rows.filter(isPlayerRow);
  if (players.length === 0) return null;
  const refs = refsOfExpression(expr, tree);
  const leaf: GraphNode = expr.includes("[")
    ? {
        kind: "formula",
        formula_expr: expr,
        human_description: description,
        node_id: "tmp",
        label: "tmp",
        aliases: [],
        authored_by: "tmp",
        created_at: "",
      }
    : {
        kind: "column",
        column_ref: expr.trim(),
        node_id: "tmp",
        label: "tmp",
        aliases: [],
        authored_by: "tmp",
        created_at: "",
      };

  const scored = players
    .map((row) => ({ row, blanks: countBlanks(row, refs, tree) }))
    .sort((a, b) => a.blanks - b.blanks);
  const chosen = scored[0];
  if (!chosen) return null;

  const blanks: string[] = [];
  try {
    const value = evalLeafForRow(leaf, chosen.row, tree, [], chosen.blanks > 0, blanks);
    let substituted = expr;
    for (const ref of refs) {
      const raw = COLUMN_BY_NAME.has(ref)
        ? rawColumnValue(chosen.row, ref)
        : evaluateLeafStrict(findMetric(tree, ref)!, chosen.row, tree).value;
      substituted = substituted.split(`[${ref}]`).join(String(raw ?? 0));
    }
    return {
      player: playerName(chosen.row),
      value,
      substituted,
      fallbackZeroed: [...new Set(blanks)],
      note:
        chosen.blanks > 0
          ? "No single player had every stat recorded, so this is the closest available real example, not a fully clean one."
          : undefined,
    };
  } catch (e) {
    return {
      player: playerName(chosen.row),
      value: null,
      substituted: expr,
      fallbackZeroed: [],
      error: e instanceof Error ? e.message : String(e),
    };
  }
}
