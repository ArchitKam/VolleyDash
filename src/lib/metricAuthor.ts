import { COLUMN_SCHEMA, SKILL_GROUPS } from "./columns";
import { callLlm } from "./data.functions";
import {
  dependentAllowed,
  fieldValueList,
  filterableFields,
  type SourceSchema,
} from "./source";
import {
  describeWhere,
  metricNodes,
  validateSpec,
  type EventWhere,
  type GraphTree,
  type MetricSpec,
} from "./tree";

export type MetricProposal = {
  matched: true;
  expr: string;
  human_description: string;
  suggested_label: string;
  suggested_branch_group: string;
  suggested_aliases: string[];
  source: "llm" | "rules";
};

export type MetricRejection = {
  matched: false;
  message: string;
  triedRules: boolean;
};

export type MetricAuthorResult = MetricProposal | MetricRejection;

export const RULE_EXAMPLE_PHRASINGS = [
  "net kills per set",
  "kills minus errors",
  "aces per set",
  "hitting efficiency / attack efficiency",
  "blocks per set",
  "dig efficiency / dig rate / dig percentage",
];

type Rule = {
  keywords: string[][];
  expr: string;
  description: string;
  label: string;
  group: string;
  aliases: string[];
};

const RULES: Rule[] = [
  {
    keywords: [["net"], ["kill", "kills"], ["per", "/"], ["set", "sets"]],
    expr: "([Attack K]-[Attack E])/[Sets Sets Played]",
    description: "Kills minus attack errors, divided by sets played.",
    label: "Net Kills Per Set",
    group: "Attack",
    aliases: ["net kills per set", "nkps"],
  },
  {
    keywords: [
      ["kill", "kills"],
      ["minus", "-"],
      ["error", "errors"],
    ],
    expr: "[Attack K]-[Attack E]",
    description: "Kills minus attack errors.",
    label: "Kills Minus Errors",
    group: "Attack",
    aliases: ["kills minus errors"],
  },
  {
    keywords: [
      ["ace", "aces"],
      ["per", "/"],
      ["set", "sets"],
    ],
    expr: "[Serve SA]/[Sets Sets Played]",
    description: "Service aces per set played.",
    label: "Aces Per Set",
    group: "Serve",
    aliases: ["aces per set", "aps"],
  },
  {
    keywords: [["hitting", "attack"], ["efficiency"]],
    expr: "Attack Atk%",
    description: "(K-E)/TA hitting efficiency.",
    label: "Hitting Efficiency",
    group: "Attack",
    aliases: ["hitting efficiency", "attack efficiency"],
  },
  {
    keywords: [
      ["block", "blocks"],
      ["per", "/"],
      ["set", "sets"],
    ],
    expr: "([Block BS]+[Block BA])/[Sets Sets Played]",
    description: "Solo plus assisted blocks per set played.",
    label: "Blocks Per Set",
    group: "Block",
    aliases: ["blocks per set", "bps"],
  },
  {
    keywords: [
      ["dig", "digs"],
      ["efficiency", "rate", "percentage", "pct"],
    ],
    expr: "[Dig DS]/([Dig DS]+[Dig DE])",
    description: "Successful digs as a share of all dig attempts.",
    label: "Dig Efficiency",
    group: "Dig",
    aliases: ["dig efficiency", "dig rate", "dig percentage"],
  },
];

export function ruleBasedAuthor(phrase: string): MetricAuthorResult {
  const text = phrase.toLowerCase();
  for (const rule of RULES) {
    const ok = rule.keywords.every((group) => group.some((k) => text.includes(k)));
    if (ok) {
      return {
        matched: true,
        expr: rule.expr,
        human_description: rule.description,
        suggested_label: rule.label,
        suggested_branch_group: rule.group,
        suggested_aliases: rule.aliases,
        source: "rules",
      };
    }
  }
  return {
    matched: false,
    message: `Couldn't match "${phrase}" to real columns or metrics.`,
    triedRules: true,
  };
}

export function buildAuthorSystemPrompt(committed: GraphTree): string {
  const columns = COLUMN_SCHEMA.map(
    (c) => `[${c.name}] — group ${c.group}, type ${c.type}: ${c.description}`,
  ).join("\n");
  const metrics = metricNodes(committed)
    .map((n) => `[${n.label}]`)
    .join(", ");
  return `You translate a volleyball coach's plain-language stat description into an arithmetic formula.

RAW CSV COLUMNS (valid bracket targets):
${columns}

ALREADY-COMMITTED METRICS (also valid bracket targets):
${metrics || "(none yet)"}

Skill groups a new metric can be filed under: ${SKILL_GROUPS.join(", ")}.

Rules:
- Only use bracketed names from the two lists above. NEVER invent a bracketed name.
- Allowed operators: + - * / and parentheses.
- If a single existing name is all that's needed, "expr" is that single bracketed name with no arithmetic.
- Respond with ONLY JSON, no prose, no code fences.

Success shape:
{"matched": true, "expr": "...", "human_description": "...", "suggested_label": "...", "suggested_branch_group": "...", "suggested_aliases": ["..."]}

Failure shape (cannot be mapped onto real columns/metrics):
{"matched": false, "message": "<brief reason>"}`;
}

export async function authorMetric(
  phrase: string,
  committed: GraphTree,
): Promise<MetricAuthorResult> {
  let text: string;
  try {
    const res = await callLlm({
      data: {
        systemPrompt: buildAuthorSystemPrompt(committed),
        userMessage: phrase,
        maxTokens: 800,
        temperature: 0.1,
      },
    });
    text = res.text;
  } catch {
    // LLM genuinely unreachable -> rule-based fallback.
    return ruleBasedAuthor(phrase);
  }

  try {
    const parsed = JSON.parse(text) as Record<string, unknown>;
    if (parsed["matched"] === true && typeof parsed["expr"] === "string") {
      return {
        matched: true,
        expr: parsed["expr"] as string,
        human_description: String(parsed["human_description"] ?? ""),
        suggested_label: String(parsed["suggested_label"] ?? phrase),
        suggested_branch_group: String(parsed["suggested_branch_group"] ?? ""),
        suggested_aliases: Array.isArray(parsed["suggested_aliases"])
          ? (parsed["suggested_aliases"] as unknown[]).map(String)
          : [],
        source: "llm",
      };
    }
    return {
      matched: false,
      message: String(parsed["message"] ?? "The model could not map that to real columns."),
      triedRules: false,
    };
  } catch {
    return {
      matched: false,
      message: "The model's answer wasn't valid JSON, so no formula could be read from it.",
      triedRules: false,
    };
  }
}

/* ---------------- event-grain authoring (play-by-play workspace) ---------------- */

export type EventMetricProposal = {
  matched: true;
  spec: MetricSpec;
  /** Human-readable form of the spec, for the wizard's expression box. */
  display: string;
  human_description: string;
  suggested_label: string;
  suggested_branch_group: string;
  suggested_aliases: string[];
  source: "llm";
};

export type EventAuthorResult = EventMetricProposal | MetricRejection;

/** The prompt is built from the LIVE schema, so it can only offer values these files contain. */
export function buildEventAuthorSystemPrompt(
  committed: GraphTree,
  schema: SourceSchema,
  branches: string[],
): string {
  const skills = fieldValueList(schema, "skill");
  const codesPerSkill = skills
    .map((skill) => {
      const codes = dependentAllowed(schema, "skill", skill, "evaluation_code");
      return `- ${skill}: evaluation codes seen = ${codes ? [...codes].sort().join(" ") : "(none)"}`;
    })
    .join("\n");
  const dimensions = filterableFields(schema)
    .filter((f) => f.name !== "skill" && f.name !== "evaluation_code")
    .map((f) => `- ${f.name} (${f.description}): ${fieldValueList(schema, f.name).slice(0, 40).join(", ")}`)
    .join("\n");
  const metrics = metricNodes(committed)
    .map((n) => `[${n.label}]`)
    .join(", ");

  return `You translate a volleyball coach's plain-language stat description into a filter over play-by-play actions, or into arithmetic over existing metrics.

Each row of this data is ONE action. Counting rows that match a filter gives a stat.

SKILLS AND THE EVALUATION CODES ACTUALLY OBSERVED FOR EACH:
${codesPerSkill || "(no skills observed)"}

OTHER FILTERABLE FIELDS AND THEIR OBSERVED VALUES:
${dimensions || "(none)"}

ALREADY-COMMITTED METRICS (valid bracket targets for a formula):
${metrics || "(none yet)"}

Branches a new metric can be filed under: ${branches.join(", ")}.

Rules:
- Only use field names and values from the lists above. NEVER invent one. An evaluation code must be one observed FOR THAT SKILL.
- A count of matching actions is an "event" metric. A ratio or per-set rate is a "formula" metric over existing metric labels.
- Respond with ONLY JSON, no prose, no code fences.

Event shape:
{"matched": true, "kind": "event", "where": {"skill": "Attack", "evaluation_code": "#"}, "aggregate": "count", "human_description": "...", "suggested_label": "...", "suggested_branch_group": "...", "suggested_aliases": ["..."]}

Formula shape:
{"matched": true, "kind": "formula", "expr": "[A]/[B]", "human_description": "...", "suggested_label": "...", "suggested_branch_group": "...", "suggested_aliases": ["..."]}

Failure shape:
{"matched": false, "message": "<brief reason>"}`;
}

export async function authorEventMetric(
  phrase: string,
  committed: GraphTree,
  schema: SourceSchema,
  branches: string[],
): Promise<EventAuthorResult> {
  let text: string;
  try {
    const res = await callLlm({
      data: {
        systemPrompt: buildEventAuthorSystemPrompt(committed, schema, branches),
        userMessage: phrase,
        maxTokens: 900,
        temperature: 0.1,
      },
    });
    text = res.text;
  } catch (e) {
    return {
      matched: false,
      message: e instanceof Error ? e.message : String(e),
      triedRules: false,
    };
  }

  let parsed: Record<string, unknown>;
  try {
    parsed = JSON.parse(text) as Record<string, unknown>;
  } catch {
    return {
      matched: false,
      message: "The model's answer wasn't valid JSON, so nothing could be read from it.",
      triedRules: false,
    };
  }

  if (parsed["matched"] !== true) {
    return {
      matched: false,
      message: String(parsed["message"] ?? "The model could not map that onto this data."),
      triedRules: false,
    };
  }

  const description = String(parsed["human_description"] ?? "");
  const common = {
    matched: true as const,
    human_description: description,
    suggested_label: String(parsed["suggested_label"] ?? phrase),
    suggested_branch_group: String(parsed["suggested_branch_group"] ?? ""),
    suggested_aliases: Array.isArray(parsed["suggested_aliases"])
      ? (parsed["suggested_aliases"] as unknown[]).map(String)
      : [],
    source: "llm" as const,
  };

  if (parsed["kind"] === "formula" && typeof parsed["expr"] === "string") {
    const spec: MetricSpec = {
      kind: "formula",
      formula_expr: parsed["expr"],
      human_description: description,
    };
    // Whatever comes back is re-validated by the real mechanical validator.
    const err = validateSpec(spec, committed, { grain: "event", schema });
    if (err) return { matched: false, message: err, triedRules: false };
    return { ...common, spec, display: parsed["expr"] };
  }

  const whereRaw = parsed["where"];
  if (!whereRaw || typeof whereRaw !== "object") {
    return { matched: false, message: "The model returned no usable filter.", triedRules: false };
  }
  const where: EventWhere = {};
  for (const [field, value] of Object.entries(whereRaw as Record<string, unknown>)) {
    where[field] = Array.isArray(value) ? value.map(String) : String(value);
  }
  const spec: MetricSpec = {
    kind: "event",
    where,
    aggregate: parsed["aggregate"] === "count_distinct" ? "count_distinct" : "count",
    field: typeof parsed["field"] === "string" ? parsed["field"] : undefined,
    human_description: description,
  };
  const err = validateSpec(spec, committed, { grain: "event", schema });
  if (err) return { matched: false, message: err, triedRules: false };
  return { ...common, spec, display: describeWhere(where) };
}
