import { evaluateLeafStrict } from "./evaluate";
import { createEvalContext, decodeKey, evaluateMetric } from "./evaluateEvent";
import { runPipeline, type Axis, type Frame, type Row } from "./pipeline";
import type { RouterAction } from "./queryRouter";
import { resolveGameHints, resolvePlayerHint, resolveSetHint } from "./resolve";
import { supportsSets, type Source } from "./source";
import type { CsvSource } from "./source.csv";
import { findMetric, findNode, metricLabelsUnder, type GraphTree } from "./tree";

export type ActionResult = {
  action: RouterAction;
  title: string;
  metricLabels: string[];
  frame: Frame;
  warnings: string[];
  gamesUsed: string[];
  setsUsed: string[];
  resolvedPlayers: string[];
  missingMetric?: string | undefined;
};

export type ComputeDeps = {
  committed: GraphTree;
  source: Source;
  /** Game labels currently ticked; empty means every game. */
  selectedGames: string[];
  /** Set labels currently ticked; empty means every set together. */
  selectedSets?: string[] | undefined;
  playerFilter?: string[] | undefined;
};

export const NO_SET_AXIS_WARNING =
  "This data has one row per player per match, so it can't be broken down by set — showing match totals instead.";

function metricLabelsFor(action: RouterAction, tree: GraphTree): string[] {
  if (action.metric_of_interest && findMetric(tree, action.metric_of_interest)) {
    return [action.metric_of_interest];
  }
  if (action.skill_group) {
    // Aggregate over every metric transitively beneath the category node.
    const category = findNode(tree, action.skill_group);
    return category ? metricLabelsUnder(tree, category.node_id) : [];
  }
  return [];
}

/**
 * Chart titles describe the stat only — players are shown through the colour key,
 * so any player name (or possessive) the router put in the title is removed.
 */
export function stripPlayersFromTitle(
  title: string,
  players: string[],
  fallback: string,
): string {
  const words = new Set<string>();
  for (const p of players) {
    words.add(p);
    for (const part of p.split(/\s+/)) if (part.length > 2) words.add(part);
  }
  let out = title;
  for (const w of [...words].sort((a, b) => b.length - a.length)) {
    const esc = w.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    out = out.replace(new RegExp(`\\b${esc}(?:'s|’s|s')?\\b`, "gi"), " ");
  }
  out = out
    .replace(/\s+/g, " ")
    .replace(/\s+(vs\.?|and|&)\s+(?=[—–\-,]|$)/gi, " ")
    .replace(/^[\s—–\-,:•]+|[\s—–\-,:]+$/g, "")
    .replace(/\s*([—–])\s*\1*/g, " $1 ")
    .replace(/\s+/g, " ")
    .replace(/\s+(for|of|by|from|about|vs\.?|and|&)$/i, "")
    .replace(/^(for|of|by|from|about)\s+/i, "")
    .replace(/^[\s—–\-,:•]+|[\s—–\-,:]+$/g, "")
    .trim();
  return out.length > 1 ? out : fallback;
}

function resolveGames(action: RouterAction, deps: ComputeDeps): string[] {
  const all = deps.source.gameLabels();
  const selected = deps.selectedGames.length > 0 ? deps.selectedGames : all;
  if (!action.game_hint) return selected;
  // A hint naming an opponent played twice means BOTH matches, not a coin flip.
  const hits = resolveGameHints(action.game_hint, all);
  return hits.length > 0 ? hits : selected;
}

function resolveSets(action: RouterAction, deps: ComputeDeps): string[] {
  if (!supportsSets(deps.source)) return [];
  const all = deps.source.setLabels();
  const selected = deps.selectedSets && deps.selectedSets.length > 0 ? deps.selectedSets : [];
  if (action.set_hint) {
    const hits = resolveSetHint(action.set_hint, all);
    // A set named in the question overrides the picker, for this action only.
    if (hits.length > 0) return hits;
  }
  return selected;
}

function pipelineUsesAxis(action: RouterAction, axis: Axis): boolean {
  return action.pipeline.some((op) => op.axis === axis);
}

/* ---------------- frame builders, one per grain ---------------- */

function buildMeasureFrame(
  labels: string[],
  games: string[],
  deps: ComputeDeps,
): { frame: Frame; roster: string[]; warnings: string[] } {
  const csv = deps.source as CsvSource;
  const frame: Frame = [];
  const roster = new Set<string>();
  const wanted = new Set(games);
  for (const { Player, Game, row } of csv.csvRows()) {
    if (!wanted.has(Game)) continue;
    roster.add(Player);
    for (const label of labels) {
      const found = findMetric(deps.committed, label);
      if (!found) continue;
      try {
        const { value, blanks } = evaluateLeafStrict(found, row, deps.committed);
        frame.push({
          Game,
          Player,
          Metric: label,
          Value: value,
          Note:
            value === null && blanks.length > 0
              ? `Missing ${[...new Set(blanks)].join(", ")} for this player in this match`
              : undefined,
        } satisfies Row);
      } catch (e) {
        frame.push({
          Game,
          Player,
          Metric: label,
          Value: null,
          Note: e instanceof Error ? e.message : String(e),
        });
      }
    }
  }
  return { frame, roster: [...roster], warnings: [] };
}

function buildEventFrame(
  labels: string[],
  games: string[],
  sets: string[],
  bySet: boolean,
  deps: ComputeDeps,
): { frame: Frame; roster: string[]; warnings: string[] } {
  const warnings: string[] = [];
  const ctx = createEvalContext({
    source: deps.source,
    tree: deps.committed,
    games,
    sets,
    bySet,
  });
  const frame: Frame = [];
  const roster = new Set<string>();
  for (const label of labels) {
    let series;
    try {
      series = evaluateMetric(ctx, label);
    } catch (e) {
      warnings.push(`"${label}" couldn't be computed: ${e instanceof Error ? e.message : String(e)}`);
      continue;
    }
    for (const [key, value] of series) {
      const id = decodeKey(key);
      if (!id.Player) continue;
      roster.add(id.Player);
      frame.push({
        Game: id.Game,
        Player: id.Player,
        Set: bySet ? id.Set : undefined,
        Metric: label,
        Value: value,
      });
    }
  }
  return { frame, roster: [...roster], warnings };
}

/** Runs one router action against whatever source the active workspace provides. */
export async function runAction(action: RouterAction, deps: ComputeDeps): Promise<ActionResult> {
  const labels = metricLabelsFor(action, deps.committed);
  const gamesUsed = resolveGames(action, deps);
  const warnings: string[] = [];
  if (action.pipelineNote) warnings.push(action.pipelineNote);

  const hasSets = supportsSets(deps.source);
  const setsUsed = resolveSets(action, deps);
  const wantsSetSplit = pipelineUsesAxis(action, "Set");
  if (!hasSets && (action.set_hint || wantsSetSplit)) warnings.push(NO_SET_AXIS_WARNING);
  // Scoping to sets NARROWS the data (so per-set rates rebase); splitting by the Set
  // axis is a separate request, and both are legal at the same time.
  const bySet = hasSets && (wantsSetSplit || setsUsed.length > 1);

  if (labels.length === 0) {
    return {
      action,
      title: action.title,
      metricLabels: [],
      frame: [],
      warnings,
      gamesUsed,
      setsUsed,
      resolvedPlayers: [],
      missingMetric: action.candidateNewMetric ?? action.metric_of_interest ?? undefined,
    };
  }

  const built =
    deps.source.grain === "event"
      ? buildEventFrame(labels, gamesUsed, setsUsed, bySet, deps)
      : buildMeasureFrame(labels, gamesUsed, deps);
  const frame = built.frame;
  const roster = built.roster.length > 0 ? built.roster : deps.source.playerLabels();
  warnings.push(...built.warnings);

  // Deterministic player resolution against the real roster.
  const mentioned = action.player
    ? Array.isArray(action.player)
      ? action.player
      : [action.player]
    : [];
  const resolvedPlayers: string[] = [];
  for (const m of mentioned) {
    const hit = resolvePlayerHint(m, roster);
    if (hit) resolvedPlayers.push(hit);
    else warnings.push(`I couldn't find a player matching "${m}" in the loaded matches.`);
  }

  let working = frame;
  if (resolvedPlayers.length > 0) {
    working = working.filter((r) => resolvedPlayers.includes(r.Player));
  }
  if (deps.playerFilter && deps.playerFilter.length > 0) {
    working = working.filter((r) => deps.playerFilter!.includes(r.Player));
  }

  // Pre-compute any metric a Slice predicate gates on but that isn't already in the frame.
  const predicateMetrics = action.pipeline
    .filter((op) => op.kind === "Slice" && op.predicate)
    .map((op) => (op as { predicate: { metric: string } }).predicate.metric)
    .filter((m) => !labels.includes(m) && findMetric(deps.committed, m));
  let predicateFrame: Frame = [];
  if (predicateMetrics.length > 0) {
    const extra = [...new Set(predicateMetrics)];
    predicateFrame =
      deps.source.grain === "event"
        ? buildEventFrame(extra, gamesUsed, setsUsed, bySet, deps).frame
        : buildMeasureFrame(extra, gamesUsed, deps).frame;
  }

  const { frame: piped, warnings: pipeWarnings } = runPipeline(working, action.pipeline, {
    resolveMetricRows: (metric) => [...frame, ...predicateFrame].filter((r) => r.Metric === metric),
  });

  return {
    action,
    title: stripPlayersFromTitle(
      action.title,
      [...resolvedPlayers, ...mentioned],
      labels.join(", ") || action.title,
    ),
    metricLabels: labels,
    frame: piped,
    warnings: [...warnings, ...pipeWarnings],
    gamesUsed,
    setsUsed,
    resolvedPlayers,
  };
}
