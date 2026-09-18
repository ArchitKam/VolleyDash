import { SKILL_GROUPS } from "./columns";
import { callLlm } from "./data.functions";
import { isValidOp, type Op } from "./pipeline";
import { categoryLabels, metricNodes, type GraphTree } from "./tree";

export type RouterAction = {
  metric_of_interest: string | null;
  skill_group: string | null;
  player: string | string[] | null;
  game_hint: string | null;
  set_hint?: string | null | undefined;
  title: string;
  pipeline: Op[];
  pipelineNote?: string | undefined;
  candidateNewMetric?: string | null | undefined;
};

export type RouterDecomposition = {
  intent_summary: string;
  actions: RouterAction[];
  reasoning: string;
  limitations: string;
  unrecognized_terms: string[];
  raw: unknown;
};

function candidateMetrics(question: string, tree: GraphTree): string[] {
  const q = question.toLowerCase();
  const scored: Array<{ label: string; score: number }> = [];
  for (const node of metricNodes(tree)) {
    const label = node.label;
    const needles = [label.toLowerCase(), ...(node.aliases ?? []).map((a) => a.toLowerCase())];
    let score = 0;
    for (const n of needles) {
      if (!n) continue;
      if (q.includes(n)) score += 3;
      else if (n.split(/\s+/).some((w) => w.length > 3 && q.includes(w))) score += 1;
    }
    if (score > 0) scored.push({ label, score });
  }
  return scored
    .sort((a, b) => b.score - a.score)
    .slice(0, 8)
    .map((s) => s.label);
}

export function buildRouterSystemPrompt(
  tree: GraphTree,
  question: string,
  roster: string[] = [],
  gameOpponents: string[] = [],
  opts: { supportsSets?: boolean; setLabels?: string[] } = {},
): string {
  const candidates = candidateMetrics(question, tree);
  const metricList =
    candidates.length > 0
      ? candidates.join(", ")
      : metricNodes(tree)
          .map((n) => n.label)
          .join(", ");
  return `You decompose a volleyball coach's question into resolvable analytics actions.

CANDIDATE METRIC LABELS: ${metricList || "(none)"}
CATEGORY (skill group) NAMES: ${SKILL_GROUPS.join(", ")}
KNOWN PLAYERS (roster): ${roster.length > 0 ? roster.join(", ") : "(none loaded yet)"}
KNOWN GAMES (opponents): ${gameOpponents.length > 0 ? gameOpponents.join(", ") : "(none loaded yet)"}${
    opts.supportsSets
      ? `\nKNOWN SETS: ${(opts.setLabels ?? []).join(", ") || "(none loaded yet)"}`
      : ""
  }

Rules:
- ONE ACTION PER DISTINCT ASK. A compound question about two different metrics becomes two actions. A question naming two different players for the same metric becomes two actions.
- A category action aggregates every metric nested anywhere beneath that category.
- Each action is EITHER about one exact existing metric label ("metric_of_interest") OR a whole skill-group category ("skill_group") — never both.
- If the phrasing is generic ("how was her serving"), prefer the category.
- If it names a specific stat that is not in the candidate list, still put that stat name verbatim in "metric_of_interest".
- For "player" and "game_hint": if the question's mention clearly refers to one of the KNOWN PLAYERS or KNOWN GAMES listed above (allowing for a nickname, jersey number, possessive, minor misspelling, or partial name), output that list entry's text EXACTLY as it appears in the list above. Only if it does NOT clearly match anything in those lists, fall back to outputting the coach's own wording with any trailing possessive stripped ("Sloan's" → "Sloan") so a secondary matcher can attempt it. Never invent a roster entry or game that isn't in the lists.
${
  opts.supportsSets
    ? `- This data has one row per action, so it CAN be narrowed to a single set. If the question names a set ("in set 3", "the fifth set"), put that set's label from KNOWN SETS in "set_hint". Otherwise leave "set_hint" null.\n`
    : ""
}- "pipeline" is optional and ordered. Only these operation kinds exist:
  {"kind":"Slice","axis":"Player|Game|Metric","keep":["..."]} or {"kind":"Slice","axis":"...","predicate":{"metric":"...","op":">|>=|<|<=|==|!=","threshold":0,"decided_by_player":null}}
  {"kind":"Reduce","axis":"...","how":"mean|sum|min|max|count|median"}
  {"kind":"Rank","axis":"...","descending":true,"limit":3}
  {"kind":"Compare","axis":"...","how":"mean","mode":"difference|ratio"}
- Put any word or phrase with NO plausible mapping to a real metric, category, player, or game into "unrecognized_terms". Never invent a typo-correction for gibberish. Don't guess.
- "unrecognized_terms" must NOT include any word/phrase that was already used to fill in metric_of_interest, skill_group, player, or game_hint in ANY action (including after stripping a possessive, nickname, or jersey number) — only list words/phrases that map to NOTHING at all (true gibberish or unrelated content).
- Respond with ONLY JSON, no prose, no code fences:
{"intent_summary":"...","actions":[{"metric_of_interest":null,"skill_group":null,"player":null,"game_hint":null,${
    opts.supportsSets ? '"set_hint":null,' : ""
  }"title":"...","pipeline":[]}],"reasoning":"...","limitations":"...","unrecognized_terms":[]}`;
}

function sameExceptPlayer(a: RouterAction, b: RouterAction): boolean {
  return (
    a.metric_of_interest === b.metric_of_interest &&
    a.skill_group === b.skill_group &&
    a.game_hint === b.game_hint &&
    (a.set_hint ?? null) === (b.set_hint ?? null) &&
    JSON.stringify(a.pipeline) === JSON.stringify(b.pipeline)
  );
}

/** Deterministic repair/validation pass. Never trust the raw model output. */
export function repairDecomposition(raw: unknown, tree: GraphTree): RouterDecomposition {
  const obj = (raw ?? {}) as Record<string, unknown>;
  const knownLabels = new Set(metricNodes(tree).map((n) => n.label));
  const knownLabelsLower = new Map(metricNodes(tree).map((n) => [n.label.toLowerCase(), n.label]));
  const categories = [...new Set([...SKILL_GROUPS, ...categoryLabels(tree)])];
  const categoriesLower = new Map(categories.map((c) => [c.toLowerCase(), c]));

  const rawActions = Array.isArray(obj["actions"]) ? (obj["actions"] as unknown[]) : [];
  let actions: RouterAction[] = [];

  for (const ra of rawActions) {
    const a = (ra ?? {}) as Record<string, unknown>;
    let metric =
      typeof a["metric_of_interest"] === "string" ? a["metric_of_interest"].trim() : null;
    let group = typeof a["skill_group"] === "string" ? a["skill_group"].trim() : null;

    if (metric) {
      const exact = knownLabelsLower.get(metric.toLowerCase());
      if (exact) metric = exact;
    }
    // A bare category name in metric_of_interest belongs in skill_group.
    if (metric && !knownLabels.has(metric) && categoriesLower.has(metric.toLowerCase())) {
      group = categoriesLower.get(metric.toLowerCase())!;
      metric = null;
    }
    if (group) {
      const g = categoriesLower.get(group.toLowerCase());
      group = g ?? null;
    }
    // Both present: an exactly-known metric label wins.
    if (metric && group) {
      if (knownLabels.has(metric)) group = null;
      else metric = null;
    }
    if (!metric && !group) continue;

    const rawPipeline = Array.isArray(a["pipeline"]) ? (a["pipeline"] as unknown[]) : [];
    let pipeline: Op[] = [];
    let pipelineNote: string | undefined;
    if (rawPipeline.length > 0) {
      if (rawPipeline.every(isValidOp)) pipeline = rawPipeline as Op[];
      else pipelineNote = "The suggested operations were malformed, so no pipeline was applied.";
    }

    const playerRaw = a["player"];
    const player: string | string[] | null = Array.isArray(playerRaw)
      ? (playerRaw as unknown[]).map(String)
      : typeof playerRaw === "string" && playerRaw.trim()
        ? playerRaw.trim()
        : null;

    actions.push({
      metric_of_interest: metric,
      skill_group: group,
      player,
      game_hint:
        typeof a["game_hint"] === "string" && a["game_hint"].trim() ? a["game_hint"].trim() : null,
      set_hint:
        typeof a["set_hint"] === "string" && a["set_hint"].trim() ? a["set_hint"].trim() : null,
      title:
        typeof a["title"] === "string" && a["title"].trim()
          ? a["title"].trim()
          : (metric ?? group ?? "Result"),
      pipeline,
      pipelineNote,
      candidateNewMetric: metric && !knownLabels.has(metric) ? metric : null,
    });
  }

  // Merge actions identical except for player.
  const merged: RouterAction[] = [];
  const used = new Set<number>();
  actions.forEach((a, i) => {
    if (used.has(i)) return;
    const partners = actions
      .map((b, j) => ({ b, j }))
      .filter(
        ({ b, j }) =>
          j !== i &&
          !used.has(j) &&
          sameExceptPlayer(a, b) &&
          typeof a.player === "string" &&
          typeof b.player === "string" &&
          a.player.toLowerCase() !== b.player.toLowerCase(),
      );
    if (partners.length > 0) {
      const players = [a.player as string, ...partners.map((p) => p.b.player as string)];
      partners.forEach((p) => used.add(p.j));
      used.add(i);
      merged.push({ ...a, player: players, title: a.title });
    } else {
      used.add(i);
      merged.push(a);
    }
  });
  actions = merged;

  return {
    intent_summary: String(obj["intent_summary"] ?? ""),
    actions,
    reasoning: String(obj["reasoning"] ?? ""),
    limitations: String(obj["limitations"] ?? ""),
    unrecognized_terms: Array.isArray(obj["unrecognized_terms"])
      ? (obj["unrecognized_terms"] as unknown[]).map(String)
      : [],
    raw,
  };
}

export async function routeQuestion(
  question: string,
  tree: GraphTree,
  roster: string[] = [],
  gameOpponents: string[] = [],
  opts: { supportsSets?: boolean; setLabels?: string[] } = {},
): Promise<RouterDecomposition> {
  const systemPrompt = buildRouterSystemPrompt(tree, question, roster, gameOpponents, opts);
  const first = await callLlm({
    data: { systemPrompt, userMessage: question, maxTokens: 1600, temperature: 0.1 },
  });
  try {
    return repairDecomposition(JSON.parse(first.text), tree);
  } catch {
    const retry = await callLlm({
      data: {
        systemPrompt: `${systemPrompt}\n\nKeep it concise. JSON only.`,
        userMessage: question,
        maxTokens: 1200,
        temperature: 0,
      },
    });
    try {
      return repairDecomposition(JSON.parse(retry.text), tree);
    } catch {
      throw new Error(
        "The router's answer couldn't be read as JSON, so the question wasn't decomposed. Try rephrasing it.",
      );
    }
  }
}
