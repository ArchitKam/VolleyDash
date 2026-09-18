import { COLUMN_BY_NAME, COLUMN_SCHEMA, SKILL_GROUPS } from "./columns";
import {
  dependentAllowed,
  fieldValueList,
  type Grain,
  type SourceSchema,
} from "./source";

export type EventAggregate = "count" | "count_distinct";

/** `where` ANDs across fields and ORs within one field's value list. */
export type EventWhere = Record<string, string | string[]>;

export type MetricSpec =
  | { kind: "column"; column_ref: string }
  | { kind: "formula"; formula_expr: string; human_description: string }
  | {
      kind: "event";
      where: EventWhere;
      aggregate: EventAggregate;
      field?: string | undefined;
      human_description?: string | undefined;
    }
  | { kind: "measure"; measure: string; human_description?: string | undefined };

export type MetricKind = MetricSpec["kind"];

/** A single node type: it may carry a spec (queryable metric) and/or parent other nodes. */
export type GraphNode = {
  node_id: string;
  label: string;
  kind?: MetricKind | undefined;
  column_ref?: string | undefined;
  formula_expr?: string | undefined;
  /** event kind */
  where?: EventWhere | undefined;
  aggregate?: EventAggregate | undefined;
  field?: string | undefined;
  /** measure kind */
  measure?: string | undefined;
  human_description?: string | undefined;
  aliases: string[];
  authored_by: string;
  created_at: string;
};

export type Edge = { parent: string; child: string };

export type GraphTree = { nodes: Record<string, GraphNode>; edges: Edge[]; root_id: string };

export const ROOT_LABEL = "Volleyball Metrics";

export function newId(prefix: string): string {
  return `${prefix}_${Math.random().toString(36).slice(2, 10)}`;
}

export function cloneTree(graph: GraphTree): GraphTree {
  return JSON.parse(JSON.stringify(graph)) as GraphTree;
}

export function isGraphTree(value: unknown): value is GraphTree {
  if (!value || typeof value !== "object") return false;
  const v = value as Record<string, unknown>;
  return (
    !!v["nodes"] &&
    typeof v["nodes"] === "object" &&
    Array.isArray(v["edges"]) &&
    typeof v["root_id"] === "string"
  );
}

/* ---------------- legacy shape + migration ---------------- */

type LegacyLeaf = {
  kind: "column" | "formula";
  column_ref?: string;
  formula_expr?: string;
  human_description?: string;
  node_id: string;
  aliases?: string[];
  authored_by?: string;
  created_at?: string;
};
type LegacyBranch = { branch_node_id?: string; metrics?: Record<string, LegacyLeaf> };
export type LegacyTree = { recruiting: Record<string, LegacyBranch> };

export function isLegacyTree(value: unknown): value is LegacyTree {
  if (!value || typeof value !== "object") return false;
  const v = value as Record<string, unknown>;
  return !!v["recruiting"] && typeof v["recruiting"] === "object" && !isGraphTree(value);
}

/** Converts the old two-level `{recruiting:{branch:{metrics:{label:leaf}}}}` into nodes+edges. */
export function migrateLegacyTree(legacy: LegacyTree): GraphTree {
  const rootId = newId("root");
  const graph: GraphTree = {
    nodes: {
      [rootId]: {
        node_id: rootId,
        label: ROOT_LABEL,
        aliases: [],
        authored_by: "system:root",
        created_at: new Date().toISOString(),
      },
    },
    edges: [],
    root_id: rootId,
  };

  for (const [branchLabel, branch] of Object.entries(legacy.recruiting ?? {})) {
    const branchId = branch?.branch_node_id || newId("branch");
    graph.nodes[branchId] = {
      node_id: branchId,
      label: branchLabel,
      aliases: [],
      authored_by: "system:csv_import",
      created_at: new Date().toISOString(),
    };
    graph.edges.push({ parent: rootId, child: branchId });

    for (const [leafLabel, leaf] of Object.entries(branch?.metrics ?? {})) {
      const leafId = leaf.node_id || newId("leaf");
      graph.nodes[leafId] = {
        node_id: leafId,
        label: leafLabel,
        kind: leaf.kind,
        column_ref: leaf.column_ref,
        formula_expr: leaf.formula_expr,
        human_description: leaf.human_description,
        aliases: leaf.aliases ?? [],
        authored_by: leaf.authored_by ?? "unknown",
        created_at: leaf.created_at ?? new Date().toISOString(),
      };
      graph.edges.push({ parent: branchId, child: leafId });
    }
  }
  return graph;
}

/** Accepts either shape and always returns the new graph shape, or null if unusable. */
export function coerceToGraph(value: unknown): GraphTree | null {
  if (isGraphTree(value)) return value;
  if (isLegacyTree(value)) return migrateLegacyTree(value);
  return null;
}

/* ---------------- traversal helpers ---------------- */

export function nodeList(graph: GraphTree): GraphNode[] {
  return Object.values(graph.nodes);
}

export function childIds(graph: GraphTree, id: string): string[] {
  return graph.edges.filter((e) => e.parent === id).map((e) => e.child);
}

export function parentIds(graph: GraphTree, id: string): string[] {
  return graph.edges.filter((e) => e.child === id).map((e) => e.parent);
}

export function childrenOf(graph: GraphTree, id: string): GraphNode[] {
  return childIds(graph, id)
    .map((c) => graph.nodes[c])
    .filter((n): n is GraphNode => !!n);
}

export function parentsOf(graph: GraphTree, id: string): GraphNode[] {
  return parentIds(graph, id)
    .map((p) => graph.nodes[p])
    .filter((n): n is GraphNode => !!n);
}

export function descendantIds(graph: GraphTree, id: string): string[] {
  const out = new Set<string>();
  const walk = (cur: string) => {
    for (const c of childIds(graph, cur)) {
      if (out.has(c)) continue;
      out.add(c);
      walk(c);
    }
  };
  walk(id);
  return [...out];
}

export function ancestorIds(graph: GraphTree, id: string): string[] {
  const out = new Set<string>();
  const walk = (cur: string) => {
    for (const p of parentIds(graph, cur)) {
      if (out.has(p)) continue;
      out.add(p);
      walk(p);
    }
  };
  walk(id);
  return [...out];
}

/** Top-level category nodes (the 8 skill groups): direct children of root. */
export function topLevelCategories(graph: GraphTree): GraphNode[] {
  return childrenOf(graph, graph.root_id);
}

export function categoryLabels(graph: GraphTree): string[] {
  return topLevelCategories(graph).map((n) => n.label);
}

/** Every node that carries a spec, i.e. is directly queryable. */
export function metricNodes(graph: GraphTree): GraphNode[] {
  return nodeList(graph).filter((n) => !!n.kind);
}

/** Every queryable metric transitively beneath a category node. */
export function metricLabelsUnder(graph: GraphTree, categoryId: string): string[] {
  const labels: string[] = [];
  for (const id of descendantIds(graph, categoryId)) {
    const n = graph.nodes[id];
    if (n?.kind) labels.push(n.label);
  }
  return [...new Set(labels)];
}

export function findNode(graph: GraphTree, label: string): GraphNode | null {
  return nodeList(graph).find((n) => n.label === label) ?? null;
}

/** A queryable metric node found by label. */
export function findMetric(graph: GraphTree, label: string): GraphNode | null {
  return metricNodes(graph).find((n) => n.label === label) ?? null;
}

export function nodeDescription(node: GraphNode): string {
  if (node.kind === "formula") return node.human_description ?? "";
  if (node.kind === "column") {
    const col = COLUMN_BY_NAME.get(node.column_ref ?? "");
    return col?.description ?? node.human_description ?? "";
  }
  return node.human_description ?? "";
}

export function describeWhere(where: EventWhere | undefined): string {
  const entries = Object.entries(where ?? {});
  if (entries.length === 0) return "every action";
  return entries
    .map(([field, value]) => `${field} = ${Array.isArray(value) ? value.join(" or ") : value}`)
    .join(" and ");
}

export function nodeExpression(node: GraphNode): string {
  if (node.kind === "column") return node.column_ref ?? "";
  if (node.kind === "formula") return node.formula_expr ?? "";
  if (node.kind === "event") {
    const agg = node.aggregate === "count_distinct" ? `count distinct ${node.field ?? ""}` : "count";
    return `${agg} where ${describeWhere(node.where)}`;
  }
  if (node.kind === "measure") return `measure: ${node.measure ?? ""}`;
  return "";
}

/** One breadcrumb path from root to the node (there may be others when multi-parent). */
export function pathLabels(graph: GraphTree, id: string): string[] {
  const seen = new Set<string>();
  const walk = (cur: string): string[] | null => {
    if (cur === graph.root_id) return [graph.nodes[cur]?.label ?? ROOT_LABEL];
    if (seen.has(cur)) return null;
    seen.add(cur);
    for (const p of parentIds(graph, cur)) {
      const up = walk(p);
      if (up) return [...up, graph.nodes[cur]?.label ?? cur];
    }
    return null;
  };
  return walk(id) ?? [graph.nodes[id]?.label ?? id];
}

/* ---------------- integrity ---------------- */

export function hasEdge(graph: GraphTree, parent: string, child: string): boolean {
  return graph.edges.some((e) => e.parent === parent && e.child === child);
}

/** Returns an error message if this edge cannot be added, else null. */
export function validateEdge(graph: GraphTree, parent: string, child: string): string | null {
  if (parent === child) return "A node can't be its own parent.";
  if (!graph.nodes[parent]) return "That parent node no longer exists.";
  if (!graph.nodes[child]) return "That child node no longer exists.";
  if (hasEdge(graph, parent, child)) {
    return `"${graph.nodes[child]!.label}" is already filed under "${graph.nodes[parent]!.label}".`;
  }
  if (descendantIds(graph, child).includes(parent)) {
    return `That would create a loop: "${graph.nodes[parent]!.label}" already sits beneath "${graph.nodes[child]!.label}".`;
  }
  return null;
}

export function addEdge(graph: GraphTree, parent: string, child: string): GraphTree {
  const next = cloneTree(graph);
  next.edges.push({ parent, child });
  return next;
}

export function removeEdge(graph: GraphTree, parent: string, child: string): GraphTree {
  const next = cloneTree(graph);
  next.edges = next.edges.filter((e) => !(e.parent === parent && e.child === child));
  return next;
}

/** Removes the node and every edge touching it. Never cascades to children. */
export function removeNode(graph: GraphTree, nodeId: string): GraphTree {
  const next = cloneTree(graph);
  delete next.nodes[nodeId];
  next.edges = next.edges.filter((e) => e.parent !== nodeId && e.child !== nodeId);
  return next;
}

export function upsertNode(graph: GraphTree, node: GraphNode): GraphTree {
  const next = cloneTree(graph);
  next.nodes[node.node_id] = node;
  return next;
}

/** Nodes (other than root) with no parents left — unreachable from root. */
export function orphanNodes(graph: GraphTree): GraphNode[] {
  return nodeList(graph).filter(
    (n) => n.node_id !== graph.root_id && parentIds(graph, n.node_id).length === 0,
  );
}

/* ---------------- seeding ---------------- */

export function buildSeedTree(): GraphTree {
  const now = new Date().toISOString();
  const rootId = newId("root");
  const graph: GraphTree = {
    nodes: {
      [rootId]: {
        node_id: rootId,
        label: ROOT_LABEL,
        aliases: [],
        authored_by: "system:root",
        created_at: now,
      },
    },
    edges: [],
    root_id: rootId,
  };

  for (const group of SKILL_GROUPS) {
    const groupId = newId("branch");
    graph.nodes[groupId] = {
      node_id: groupId,
      label: group,
      aliases: [group.toLowerCase()],
      authored_by: "system:csv_import",
      created_at: now,
    };
    graph.edges.push({ parent: rootId, child: groupId });

    for (const col of COLUMN_SCHEMA.filter((c) => c.group === group)) {
      const leafId = newId("leaf");
      graph.nodes[leafId] = {
        node_id: leafId,
        label: col.description,
        kind: "column",
        column_ref: col.name,
        aliases: [col.name.toLowerCase()],
        authored_by: "system:csv_import",
        created_at: now,
      };
      graph.edges.push({ parent: groupId, child: leafId });
    }
  }
  return graph;
}

/* ---------------- validation ---------------- */

export function bracketRefs(expr: string): string[] {
  return [...expr.matchAll(/\[([^\]]+)\]/g)].map((m) => (m[1] ?? "").trim());
}

/**
 * Where a formula's bracket tokens may point. At measure grain that's a raw CSV
 * column or another metric; at event grain it's other METRIC LABELS only.
 */
export type ValidationCtx = { grain?: Grain | undefined; schema?: SourceSchema | undefined };

export function knownReferenceNames(committed: GraphTree, ctx: ValidationCtx = {}): string[] {
  const metrics = metricNodes(committed).map((n) => n.label);
  if (ctx.grain === "event") return metrics;
  return [...COLUMN_SCHEMA.map((c) => c.name), ...metrics];
}

export function validateReference(
  name: string,
  committed: GraphTree,
  ctx: ValidationCtx = {},
): string | null {
  if (ctx.grain !== "event" && COLUMN_BY_NAME.has(name)) return null;
  if (findMetric(committed, name)) return null;
  return `Unknown reference "${name}". Valid names are: ${knownReferenceNames(committed, ctx).join(", ")}`;
}

/** Mechanical check of an event filter against the vocabulary these files actually contain. */
function validateEventSpec(
  spec: Extract<MetricSpec, { kind: "event" }>,
  schema: SourceSchema | undefined,
): string | null {
  if (!schema) return "This workspace has no loaded match data to validate against yet.";
  const entries = Object.entries(spec.where ?? {});
  if (entries.length === 0) return "An event metric needs at least one filter.";
  if (spec.aggregate !== "count" && spec.aggregate !== "count_distinct") {
    return 'An event metric must aggregate with "count" or "count_distinct".';
  }
  if (spec.aggregate === "count_distinct" && !spec.field?.trim()) {
    return "Counting distinct values needs a field to count.";
  }
  if (spec.field && !schema.fields[spec.field]) {
    return `Unknown field "${spec.field}". Fields in this data: ${Object.keys(schema.fields).join(", ")}`;
  }
  for (const [field, raw] of entries) {
    const fieldSpec = schema.fields[field];
    if (!fieldSpec) {
      return `Unknown field "${field}". Fields in this data: ${Object.keys(schema.fields).join(", ")}`;
    }
    const values = Array.isArray(raw) ? raw : [raw];
    if (values.length === 0) return `The filter on "${field}" has no values.`;
    for (const v of values) {
      if (fieldSpec.values && !fieldSpec.values.has(v)) {
        return `"${v}" was never recorded in ${field} in these matches. Values seen: ${fieldValueList(schema, field).join(", ")}`;
      }
    }
  }
  // An evaluation code means something different per skill and isn't legal for every skill.
  const skillRaw = spec.where["skill"];
  const codeRaw = spec.where["evaluation_code"];
  if (skillRaw && codeRaw) {
    const skills = Array.isArray(skillRaw) ? skillRaw : [skillRaw];
    const codes = Array.isArray(codeRaw) ? codeRaw : [codeRaw];
    for (const skill of skills) {
      const legal = dependentAllowed(schema, "skill", skill, "evaluation_code");
      for (const code of codes) {
        if (!legal || !legal.has(code)) {
          return `"${code}" was never recorded on ${skill} actions in these matches. Codes seen for ${skill}: ${
            legal ? [...legal].sort().join(", ") : "(none)"
          }`;
        }
      }
    }
  }
  return null;
}

export function validateSpec(
  spec: MetricSpec,
  committed: GraphTree,
  ctx: ValidationCtx = {},
): string | null {
  if (spec.kind === "column") {
    if (!spec.column_ref?.trim()) return "A column metric needs a column reference.";
    return validateReference(spec.column_ref.trim(), committed, ctx);
  }
  if (spec.kind === "event") return validateEventSpec(spec, ctx.schema);
  if (spec.kind === "measure") {
    if (!spec.measure?.trim()) return "A measure metric needs a measure name.";
    const field = ctx.schema?.fields[spec.measure];
    if (ctx.schema && (!field || field.role !== "measure")) {
      return `"${spec.measure}" isn't a measure in this data.`;
    }
    return null;
  }
  const expr = spec.formula_expr?.trim();
  if (!expr) return "A formula metric needs an expression.";
  const refs = bracketRefs(expr);
  if (refs.length === 0) return "The formula has no [bracketed] references.";
  for (const r of refs) {
    const err = validateReference(r, committed, ctx);
    if (err) return err;
  }
  const skeleton = expr.replace(/\[[^\]]+\]/g, "1");
  if (/[^0-9+\-*/().\s]/.test(skeleton)) {
    return "The formula may only use bracketed references, numbers, + - * / and parentheses.";
  }
  return null;
}

export function specFromParts(
  expr: string,
  description: string,
  ctx: ValidationCtx = {},
): MetricSpec {
  const trimmed = expr.trim();
  const refs = bracketRefs(trimmed);
  const bare = refs.length === 1 && trimmed === `[${refs[0]}]`;
  // At event grain there are no raw columns to point at, so it's always a formula.
  if (ctx.grain !== "event") {
    if (bare && COLUMN_BY_NAME.has(refs[0]!)) {
      return { kind: "column", column_ref: refs[0]! };
    }
    if (!trimmed.includes("[") && COLUMN_BY_NAME.has(trimmed)) {
      return { kind: "column", column_ref: trimmed };
    }
  }
  return { kind: "formula", formula_expr: trimmed, human_description: description };
}

export function nodeFromSpec(
  spec: MetricSpec,
  opts: { label: string; aliases: string[]; authored_by: string; description?: string },
): GraphNode {
  const base = {
    node_id: newId("leaf"),
    label: opts.label,
    aliases: opts.aliases,
    authored_by: opts.authored_by,
    created_at: new Date().toISOString(),
  };
  if (spec.kind === "column") {
    return { ...base, kind: "column", column_ref: spec.column_ref };
  }
  if (spec.kind === "event") {
    return {
      ...base,
      kind: "event",
      where: spec.where,
      aggregate: spec.aggregate,
      field: spec.field,
      human_description: spec.human_description || opts.description || "",
    };
  }
  if (spec.kind === "measure") {
    return {
      ...base,
      kind: "measure",
      measure: spec.measure,
      human_description: spec.human_description || opts.description || "",
    };
  }
  return {
    ...base,
    kind: "formula",
    formula_expr: spec.formula_expr,
    human_description: spec.human_description || opts.description || "",
  };
}

/** Re-points an existing node at a new spec, clearing the other kinds' payloads. */
export function applySpecToNode(
  existing: GraphNode,
  spec: MetricSpec,
  label: string,
  description: string,
): GraphNode {
  const cleared: GraphNode = {
    ...existing,
    label,
    kind: spec.kind,
    column_ref: undefined,
    formula_expr: undefined,
    where: undefined,
    aggregate: undefined,
    field: undefined,
    measure: undefined,
    human_description: description,
  };
  if (spec.kind === "column") return { ...cleared, column_ref: spec.column_ref };
  if (spec.kind === "formula") {
    return { ...cleared, formula_expr: spec.formula_expr, human_description: description };
  }
  if (spec.kind === "event") {
    return { ...cleared, where: spec.where, aggregate: spec.aggregate, field: spec.field };
  }
  return { ...cleared, measure: spec.measure };
}

/** Round-trips every spec kind through persistence. */
export function specOfNode(node: GraphNode): MetricSpec | null {
  if (node.kind === "column") return { kind: "column", column_ref: node.column_ref ?? "" };
  if (node.kind === "formula") {
    return {
      kind: "formula",
      formula_expr: node.formula_expr ?? "",
      human_description: node.human_description ?? "",
    };
  }
  if (node.kind === "event") {
    return {
      kind: "event",
      where: node.where ?? {},
      aggregate: node.aggregate ?? "count",
      field: node.field,
      human_description: node.human_description,
    };
  }
  if (node.kind === "measure") {
    return {
      kind: "measure",
      measure: node.measure ?? "",
      human_description: node.human_description,
    };
  }
  return null;
}

/* ---------------- diff / merge ---------------- */

export type DiffEntry = {
  type: "Added" | "Edited" | "Linked" | "Unlinked" | "Deleted";
  label: string;
  branch: string;
  before?: string | undefined;
  after?: string | undefined;
  detail?: string | undefined;
};

export const DEPENDENTS_STUB = "No known dependents (usage logging not yet wired up)";

function summarize(node: GraphNode): string {
  const expr = nodeExpression(node);
  const desc = nodeDescription(node);
  if (!expr && !desc) return "category node (no formula)";
  return `${expr || "category"} — ${desc}`;
}

export function diffTrees(staging: GraphTree, committed: GraphTree): DiffEntry[] {
  const out: DiffEntry[] = [];
  const label = (g: GraphTree, id: string) => g.nodes[id]?.label ?? id;

  for (const node of nodeList(staging)) {
    const before = committed.nodes[node.node_id];
    if (!before) {
      const parents = parentsOf(staging, node.node_id).map((p) => p.label);
      out.push({
        type: "Added",
        label: node.label,
        branch: parents.join(", ") || "orphaned",
        after: summarize(node),
        detail: `added under: ${parents.join(", ") || "(no parent)"}`,
      });
      continue;
    }
    const changed =
      before.label !== node.label ||
      nodeExpression(before) !== nodeExpression(node) ||
      nodeDescription(before) !== nodeDescription(node);
    if (changed) {
      out.push({
        type: "Edited",
        label: node.label,
        branch: parentsOf(staging, node.node_id)
          .map((p) => p.label)
          .join(", "),
        before: `${before.label}: ${summarize(before)}`,
        after: `${node.label}: ${summarize(node)}`,
      });
    }
  }

  for (const node of nodeList(committed)) {
    if (!staging.nodes[node.node_id]) {
      out.push({
        type: "Deleted",
        label: node.label,
        branch: parentsOf(committed, node.node_id)
          .map((p) => p.label)
          .join(", "),
        before: summarize(node),
      });
    }
  }

  // Edge diff — only between nodes that exist on both sides.
  for (const e of staging.edges) {
    if (hasEdge(committed, e.parent, e.child)) continue;
    if (!committed.nodes[e.parent] || !committed.nodes[e.child]) continue; // part of an Added entry
    out.push({
      type: "Linked",
      label: label(staging, e.child),
      branch: label(staging, e.parent),
      after: `${label(staging, e.child)} now also filed under ${label(staging, e.parent)}`,
    });
  }
  for (const e of committed.edges) {
    if (hasEdge(staging, e.parent, e.child)) continue;
    if (!staging.nodes[e.parent] || !staging.nodes[e.child]) continue; // covered by Deleted
    out.push({
      type: "Unlinked",
      label: label(committed, e.child),
      branch: label(committed, e.parent),
      before: `${label(committed, e.child)} was filed under ${label(committed, e.parent)}`,
    });
  }

  return out;
}
