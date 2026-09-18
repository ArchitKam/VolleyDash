import { useCallback, useEffect, useMemo, useState } from "react";
import { toast } from "sonner";
import { ChevronDown, Loader2, PanelLeftClose, PanelLeftOpen } from "lucide-react";

import { ChartSection } from "@/components/ChartSection";
import { ConsolidatedTable } from "@/components/ConsolidatedTable";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Input } from "@/components/ui/input";
import { runAction, type ActionResult } from "@/lib/compute";
import { buildPlayerColors } from "@/lib/format";
import { authorEventMetric, authorMetric, type MetricProposal } from "@/lib/metricAuthor";
import { routeQuestion, type RouterDecomposition } from "@/lib/queryRouter";
import { useStore } from "@/lib/store";
import { categoryLabels } from "@/lib/tree";

const SAMPLES = [
  "What is Sloan's Kills Per Set in the Vegas Aces game?",
  "How was Sloan's serving throughout all games?",
  "Who has the highest Kills Per Set in the Mavs 816 game?",
];

type ProposalView = {
  expr: string;
  human_description: string;
  suggested_label: string;
  suggested_branch_group: string;
  suggested_aliases: string[];
};

type MissingMetric = {
  requested: string;
  player: string | null;
  game: string | null;
  proposal: ProposalView | null;
  message?: string | undefined;
};

export function AskTab({ onSwitchToKb }: { onSwitchToKb: () => void }) {
  const {
    committed,
    source,
    schema,
    grain,
    gameLabels,
    setLabels,
    hasSetAxis,
    roster,
    setPrefill,
    rosterLoading,
    rosterReady,
    workspaceKey,
  } = useStore();
  const [selected, setSelected] = useState<string[] | null>(null);
  const [selectedSets, setSelectedSets] = useState<string[]>([]);
  const [question, setQuestion] = useState("");
  const [lastQuestion, setLastQuestion] = useState("");
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [decomp, setDecomp] = useState<RouterDecomposition | null>(null);
  const [results, setResults] = useState<ActionResult[]>([]);
  const [missing, setMissing] = useState<MissingMetric[]>([]);
  const [playerFilter, setPlayerFilter] = useState<string[]>([]);
  const [sidebarOpen, setSidebarOpen] = useState(true);

  // Nothing from one workspace may carry over into the other.
  useEffect(() => {
    setSelected(null);
    setSelectedSets([]);
    setQuestion("");
    setLastQuestion("");
    setDecomp(null);
    setResults([]);
    setMissing([]);
    setPlayerFilter([]);
    setError(null);
  }, [workspaceKey]);

  const selectedGames: string[] = useMemo(
    () => (selected === null ? gameLabels : gameLabels.filter((g) => selected.includes(g))),
    [gameLabels, selected],
  );

  const categoryOptions = useMemo(() => (committed ? categoryLabels(committed) : []), [committed]);

  const playerColors = useMemo(() => buildPlayerColors(roster), [roster]);

  const ask = useCallback(
    async (text: string, filter: string[]) => {
      if (!committed || !source || !text.trim()) return;
      setRunning(true);
      setError(null);
      setLastQuestion(text);
      try {
        // The router needs the FULL roster to resolve nicknames/possessives, so wait for
        // the eager all-games warm-up even on the very first question of the session.
        const fullRoster = await rosterReady();
        const decomposition = await routeQuestion(
          text,
          committed,
          fullRoster.length > 0 ? fullRoster : roster,
          gameLabels,
          { supportsSets: hasSetAxis, setLabels },
        );
        setDecomp(decomposition);

        const deps = {
          committed,
          source,
          selectedGames,
          selectedSets,
          playerFilter: filter,
        };
        const actionResults: ActionResult[] = [];
        for (const action of decomposition.actions) {
          actionResults.push(await runAction(action, deps));
        }
        setResults(actionResults);

        // Any player named in the question becomes the starting selection in the Player Key,
        // so the coach can simply tick extra players to compare against.
        if (filter.length === 0) {
          const named = [...new Set(actionResults.flatMap((r) => r.resolvedPlayers))];
          if (named.length > 0) setPlayerFilter(named);
        }


        const gibberish = decomposition.unrecognized_terms.map((t) => t.toLowerCase());
        const missingList: MissingMetric[] = [];
        for (const r of actionResults) {
          if (!r.missingMetric) continue;
          if (gibberish.includes(r.missingMetric.toLowerCase())) continue;
          let proposal: ProposalView | null = null;
          let message: string | undefined;
          if (grain === "event" && schema) {
            const authored = await authorEventMetric(
              r.missingMetric,
              committed,
              schema,
              committed ? categoryOptions : [],
            );
            if (authored.matched) proposal = { ...authored, expr: authored.display };
            else message = authored.message;
          } else {
            const authored: MetricProposal | { matched: false; message: string } =
              await authorMetric(r.missingMetric, committed);
            if (authored.matched) proposal = authored;
            else message = authored.message;
          }
          missingList.push({
            requested: r.missingMetric,
            player:
              r.resolvedPlayers[0] ??
              (typeof r.action.player === "string" ? r.action.player : null),
            game: r.gamesUsed.join(", ") || null,
            proposal,
            message,
          });
        }
        setMissing(missingList);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
        setResults([]);
        setDecomp(null);
        setMissing([]);
      } finally {
        setRunning(false);
      }
    },
    [
      committed,
      source,
      schema,
      grain,
      gameLabels,
      setLabels,
      hasSetAxis,
      selectedGames,
      selectedSets,
      roster,
      rosterReady,
      categoryOptions,
    ],
  );

  const togglePlayer = (player: string) => {
    const next = playerFilter.includes(player)
      ? playerFilter.filter((p) => p !== player)
      : [...playerFilter, player];
    setPlayerFilter(next);
    if (lastQuestion) void ask(lastQuestion, next);
  };

  const gameOptions = gameLabels;

  return (
    <div className="flex gap-4">
      {sidebarOpen ? (
        <aside className="w-56 shrink-0 space-y-3 rounded-lg border border-border bg-card/60 p-3">
          <div className="flex items-center justify-between gap-2">
            <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              Games
            </span>
            <Button
              size="icon"
              variant="ghost"
              className="size-7 text-muted-foreground"
              aria-label="Hide games list"
              onClick={() => setSidebarOpen(false)}
            >
              <PanelLeftClose className="size-4" />
            </Button>
          </div>
          <p className="text-[11px] leading-snug text-muted-foreground">
            Default scope for any part of your question that doesn&apos;t name a game.
          </p>
          <div className="max-h-[26rem] space-y-1 overflow-y-auto pr-1">
            {gameOptions.length === 0 && (
              <p className="text-xs text-muted-foreground">No match files found yet.</p>
            )}
            {gameOptions.map((opponent) => {
              const active = selected === null || selected.includes(opponent);
              return (
                <label
                  key={opponent}
                  data-active={active}
                  className="flex cursor-pointer items-center gap-2 rounded-md px-2 py-1.5 text-sm text-muted-foreground transition-colors hover:bg-secondary/60 data-[active=true]:text-foreground"
                >
                  <Checkbox
                    checked={active}
                    onCheckedChange={() => {
                      const base = selected === null ? gameOptions : selected;
                      const next = active
                        ? base.filter((g) => g !== opponent)
                        : [...base, opponent];
                      setSelected(next);
                    }}
                  />
                  <span className="truncate">{opponent}</span>
                </label>
              );
            })}
          </div>
          {selected !== null && (
            <Button
              size="sm"
              variant="ghost"
              className="w-full justify-start text-xs text-muted-foreground"
              onClick={() => setSelected(null)}
            >
              Reset to all games
            </Button>
          )}

          {hasSetAxis && (
            <div className="space-y-1 border-t border-border pt-3">
              <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                Sets
              </span>
              <p className="text-[11px] leading-snug text-muted-foreground">
                Leave empty for all sets together.
              </p>
              {setLabels.map((label) => (
                <label
                  key={label}
                  className="flex cursor-pointer items-center gap-2 rounded-md px-2 py-1 text-sm text-muted-foreground transition-colors hover:bg-secondary/60"
                >
                  <Checkbox
                    checked={selectedSets.includes(label)}
                    onCheckedChange={() =>
                      setSelectedSets(
                        selectedSets.includes(label)
                          ? selectedSets.filter((s) => s !== label)
                          : [...selectedSets, label],
                      )
                    }
                  />
                  <span>{label}</span>
                </label>
              ))}
            </div>
          )}
        </aside>
      ) : (
        <Button
          size="sm"
          variant="outline"
          className="h-9 shrink-0 text-muted-foreground"
          onClick={() => setSidebarOpen(true)}
        >
          <PanelLeftOpen className="mr-1 size-4" /> Games
        </Button>
      )}

      <div className="min-w-0 flex-1 space-y-6">


      <section className="space-y-2">
        <div className="flex flex-col gap-2 sm:flex-row">
          <Input
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") void ask(question, playerFilter);
            }}
            placeholder="What is Sloan's Kills Per Set in the Vegas Aces game?"
            aria-label="Ask about a metric, a player, and/or a game"
          />
          <Button onClick={() => void ask(question, playerFilter)} disabled={running}>
            {running || rosterLoading ? <Loader2 className="mr-1 size-4 animate-spin" /> : null}
            {rosterLoading && running ? "Loading match data…" : "Ask 🏐"}
          </Button>
        </div>
        <p className="text-xs text-muted-foreground">Ask about a metric, a player, and/or a game</p>
        <div className="flex flex-wrap gap-2">
          {SAMPLES.map((s) => (
            <Button
              key={s}
              size="sm"
              variant="outline"
              onClick={() => {
                setQuestion(s);
                void ask(s, playerFilter);
              }}
            >
              {s}
            </Button>
          ))}
        </div>
      </section>

      {error && (
        <div className="rounded-lg border border-primary bg-card p-3 text-sm text-primary">
          {error}
        </div>
      )}

      {(results.length > 0 || decomp) && committed && (
        <>
          <section className="space-y-2">
            <h3 className="section-heading text-lg">📊 Consolidated Analytical Matrix</h3>
            <ConsolidatedTable
              results={results}
              committed={committed}
              unrecognized={decomp?.unrecognized_terms ?? []}
            />
          </section>

          {missing.length > 0 && (
            <section className="space-y-2">
              <h3 className="section-heading text-lg">⚠️ Missing Metric Detected</h3>
              {missing.map((m) => (
                <div
                  key={m.requested}
                  className="grid gap-4 rounded-lg border border-gold/60 bg-card p-4 md:grid-cols-2"
                >
                  <div className="space-y-1 text-sm">
                    <div className="font-semibold text-gold">Router interpretation</div>
                    <div>Requested stat: {m.requested}</div>
                    <div>Target player: {m.player ?? "—"}</div>
                    <div>Target game: {m.game ?? "—"}</div>
                  </div>
                  <div className="space-y-2 text-sm">
                    {m.proposal ? (
                      <>
                        <div className="font-semibold text-gold">Proposed metric</div>
                        <code className="block text-xs text-primary">{m.proposal.expr}</code>
                        <div className="text-muted-foreground">{m.proposal.human_description}</div>
                        <div className="text-xs text-muted-foreground">
                          Suggested group: {m.proposal.suggested_branch_group || "—"}
                        </div>
                        <Button
                          size="sm"
                          onClick={() => {
                            setPrefill({
                              expr: m.proposal!.expr,
                              description: m.proposal!.human_description,
                              label: m.proposal!.suggested_label || m.requested,
                              aliases: m.proposal!.suggested_aliases.join(", "),
                              branch: m.proposal!.suggested_branch_group,
                            });
                            toast.info(
                              "Pre-filled in the Knowledge Base tab — switch tabs to finish staging it.",
                            );
                            onSwitchToKb();
                          }}
                        >
                          🚀 Pre-fill in Knowledge Base tab
                        </Button>
                      </>
                    ) : (
                      <div className="text-muted-foreground">
                        {m.message ?? "No formula could be proposed for this stat."}
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </section>
          )}

          <section className="space-y-2">
            <h3 className="section-heading text-lg">📈 Interactive Visualizations</h3>
            <div className="grid gap-4 md:grid-cols-[14rem_1fr]">
              <div className="space-y-2 rounded-lg border border-border bg-card p-3">
                <div className="text-sm font-semibold text-gold">Player Key</div>
                {roster.map((p) => (
                  <label key={p} className="flex items-center gap-2 text-sm">
                    <Checkbox
                      checked={playerFilter.includes(p)}
                      onCheckedChange={() => togglePlayer(p)}
                    />
                    <span
                      className="inline-block size-3 rounded-sm"
                      style={{ backgroundColor: playerColors[p] }}
                    />
                    {p}
                  </label>
                ))}
                {playerFilter.length > 0 && (
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => {
                      setPlayerFilter([]);
                      if (lastQuestion) void ask(lastQuestion, []);
                    }}
                  >
                    Clear player filter
                  </Button>
                )}
              </div>
              <div className="space-y-4">
                {results
                  .filter((r) => r.frame.some((row) => row.Value !== null))
                  .map((r) => (
                    <ChartSection key={r.title} result={r} />
                  ))}
              </div>
            </div>
          </section>

          {decomp && (
            <Collapsible>
              <CollapsibleTrigger asChild>
                <Button variant="outline" size="sm">
                  <ChevronDown className="mr-1 size-4" /> ⚙️ LLM Router Decomposition
                </Button>
              </CollapsibleTrigger>
              <CollapsibleContent className="mt-2 space-y-2 rounded-lg border border-border bg-card p-3 text-sm">
                <div>
                  <span className="text-gold">Intent:</span> {decomp.intent_summary}
                </div>
                <div>
                  <span className="text-gold">Reasoning:</span> {decomp.reasoning}
                </div>
                {decomp.limitations && (
                  <div>
                    <span className="text-gold">Limitations:</span> {decomp.limitations}
                  </div>
                )}
                <pre className="overflow-x-auto rounded bg-black/60 p-2 text-xs">
                  {JSON.stringify(decomp.raw, null, 2)}
                </pre>
              </CollapsibleContent>
            </Collapsible>
          )}
        </>
      )}
      </div>
    </div>
  );
}
