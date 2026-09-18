import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import { Loader2 } from "lucide-react";

import { AskTab } from "@/components/AskTab";
import { KbTab } from "@/components/KbTab";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Toaster } from "@/components/ui/sonner";
import {
  useStore,
  VolleyProvider,
  WORKSPACE_CAPTIONS,
  WORKSPACE_LABELS,
  type Workspace,
} from "@/lib/store";
import { topLevelCategories } from "@/lib/tree";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "VolleyDash — Recruiting Knowledge Base" },
      {
        name: "description",
        content:
          "Mixed-initiative volleyball recruiting analytics: ask questions about player stats and curate a mechanically validated metrics knowledge base.",
      },
      { property: "og:title", content: "VolleyDash — Recruiting Knowledge Base" },
      {
        property: "og:description",
        content:
          "Ask plain-language questions about volleyball match stats and manage a validated metrics tree with staging and diffs.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: Index,
});

function Index() {
  return (
    <VolleyProvider>
      <Dashboard />
      <Toaster />
    </VolleyProvider>
  );
}

function Dashboard() {
  const {
    loading,
    bootError,
    warnings,
    workspace,
    setWorkspace,
    team,
    setTeam,
    teamOptions,
    committed,
  } = useStore();
  const [tab, setTab] = useState("ask");
  const emptyKb = !loading && !!committed && topLevelCategories(committed).length === 0;

  return (
    <main className="mx-auto min-h-screen max-w-7xl px-4 py-8">
      <header className="mb-6">
        <h1 className="section-heading text-3xl">
          🏐 {workspace === "player" ? "Player Analysis" : "Recruiting"} Knowledge Base
        </h1>
        <p className="mt-2 text-sm text-muted-foreground">{WORKSPACE_CAPTIONS[workspace]}</p>
      </header>

      <div className="mb-6 flex flex-wrap items-end gap-3">
        <div className="space-y-1">
          <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            Data source
          </span>
          <Select value={workspace} onValueChange={(v) => setWorkspace(v as Workspace)}>
            <SelectTrigger className="w-[280px]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="recruiting">{WORKSPACE_LABELS.recruiting}</SelectItem>
              <SelectItem value="player">{WORKSPACE_LABELS.player}</SelectItem>
            </SelectContent>
          </Select>
        </div>

        {workspace === "player" && teamOptions.length > 0 && (
          <div className="space-y-1">
            <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              Team
            </span>
            <Select value={team ?? ""} onValueChange={setTeam}>
              <SelectTrigger className="w-[240px]">
                <SelectValue placeholder="Pick a team" />
              </SelectTrigger>
              <SelectContent>
                {teamOptions.map((t) => (
                  <SelectItem key={t} value={t}>
                    {t}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        )}
      </div>

      {warnings.length > 0 && (
        <div className="mb-4 space-y-1 rounded-lg border border-border bg-card p-3 text-sm text-muted-foreground">
          {warnings.map((w) => (
            <div key={w}>{w}</div>
          ))}
        </div>
      )}

      {emptyKb && (
        <div className="mb-4 rounded-lg border border-primary bg-card p-3 text-sm text-primary">
          This knowledge base has no categories, so every question will be dropped.
        </div>
      )}

      {bootError && (
        <div className="mb-4 rounded-lg border border-primary bg-card p-3 text-sm text-primary">
          {bootError}
        </div>
      )}

      {loading ? (
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <Loader2 className="size-4 animate-spin" /> Loading the knowledge base…
        </div>
      ) : (
        <Tabs value={tab} onValueChange={setTab}>
          <TabsList>
            <TabsTrigger value="ask">Ask a Question</TabsTrigger>
            <TabsTrigger value="kb">Knowledge Base</TabsTrigger>
          </TabsList>
          <TabsContent value="ask" className="mt-6">
            <AskTab onSwitchToKb={() => setTab("kb")} />
          </TabsContent>
          <TabsContent value="kb" className="mt-6">
            <KbTab />
          </TabsContent>
        </Tabs>
      )}
    </main>
  );
}
