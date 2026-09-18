import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { toast } from "sonner";

import type { CsvRow } from "./csv";
import {
  getDvwFile,
  getGameCsv,
  getTree,
  getVolleyTree,
  listDvwFiles,
  listGames,
  saveTree,
  saveVolleyTree,
  type DvwFileRef,
  type GameRef,
} from "./data.functions";
import { parseDvw, teamCounts, type DvwMatch } from "./dvw";
import { buildDvwSeedTree } from "./dvwSeed";
import { supportsSets, type Grain, type Source, type SourceSchema } from "./source";
import { buildCsvSource } from "./source.csv";
import { buildDvwSource } from "./source.dvw";
import { buildSeedTree, cloneTree, coerceToGraph, isGraphTree, type GraphTree } from "./tree";

export type Workspace = "recruiting" | "player";

export const WORKSPACE_LABELS: Record<Workspace, string> = {
  recruiting: "Recruiting (match exports)",
  player: "Player analysis (play-by-play)",
};

export const WORKSPACE_CAPTIONS: Record<Workspace, string> = {
  recruiting: "Recruiting (match exports): one row per player per match, from Huddle CSV exports.",
  player:
    "Player analysis (play-by-play): one row per action, from DataVolley .dvw files. Supports per-set drill-down.",
};

export type WizardPrefill = {
  expr: string;
  description: string;
  label: string;
  aliases: string;
  branch: string;
} | null;

type StoreValue = {
  loading: boolean;
  bootError: string | null;
  /** Non-fatal problems from assembling the active workspace. */
  warnings: string[];

  workspace: Workspace;
  setWorkspace: (w: Workspace) => void;
  /** Changes whenever the workspace or team changes; tabs reset their state on it. */
  workspaceKey: string;
  team: string | null;
  setTeam: (t: string) => void;
  teamOptions: string[];

  source: Source | null;
  grain: Grain;
  schema: SourceSchema | null;
  gameLabels: string[];
  setLabels: string[];
  hasSetAxis: boolean;

  /** Recruiting CSV files — the Knowledge Base worked example reads one of these. */
  csvGames: GameRef[];
  loadCsv: (path: string) => Promise<CsvRow[]>;

  committed: GraphTree | null;
  staging: GraphTree | null;
  setCommitted: (t: GraphTree) => void;
  setStaging: (t: GraphTree) => void;
  persistCommitted: (t: GraphTree) => Promise<{ ok: boolean; error?: string }>;

  roster: string[];
  rosterLoading: boolean;
  rosterReady: () => Promise<string[]>;

  prefill: WizardPrefill;
  setPrefill: (p: WizardPrefill) => void;
  branchFocus: string | null;
  setBranchFocus: (b: string | null) => void;
};

const StoreContext = createContext<StoreValue | null>(null);

export function VolleyProvider({ children }: { children: React.ReactNode }) {
  const [workspace, setWorkspaceState] = useState<Workspace>("recruiting");
  const [loading, setLoading] = useState(true);
  const [bootError, setBootError] = useState<string | null>(null);

  /* recruiting workspace */
  const [csvGames, setCsvGames] = useState<GameRef[]>([]);
  const [csvTree, setCsvTree] = useState<GraphTree | null>(null);
  const [csvStaging, setCsvStaging] = useState<GraphTree | null>(null);
  const csvCache = useRef<Map<string, CsvRow[]>>(new Map());
  // De-dupes concurrent reads of the same file (StrictMode double effects, warm-up + tab mount).
  const csvInflight = useRef<Map<string, Promise<CsvRow[]>>>(new Map());
  const [csvLoadedCount, setCsvLoadedCount] = useState(0);
  const [rosterLoading, setRosterLoading] = useState(true);
  const warmupRef = useRef<Promise<void> | null>(null);

  /* player-analysis workspace */
  const [dvwMatches, setDvwMatches] = useState<DvwMatch[]>([]);
  const [dvwTree, setDvwTree] = useState<GraphTree | null>(null);
  const [dvwStaging, setDvwStaging] = useState<GraphTree | null>(null);
  const [team, setTeamState] = useState<string | null>(null);
  const [dvwLoading, setDvwLoading] = useState(false);
  const [dvwWarnings, setDvwWarnings] = useState<string[]>([]);
  const dvwStarted = useRef(false);
  const dvwReadyRef = useRef<Promise<void> | null>(null);
  const dvwTreeSeeded = useRef(false);

  const [prefill, setPrefill] = useState<WizardPrefill>(null);
  const [branchFocus, setBranchFocus] = useState<string | null>(null);
  const rosterRef = useRef<string[]>([]);

  /* ---------------- recruiting boot ---------------- */

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [treeRaw, gameList] = await Promise.all([
          getTree().catch(() => null),
          listGames().catch((e: unknown) => {
            if (!cancelled) {
              setBootError(
                `Couldn't list the match files: ${e instanceof Error ? e.message : String(e)}`,
              );
            }
            return [] as GameRef[];
          }),
        ]);
        if (cancelled) return;
        setCsvGames(gameList);

        let parsedTree: unknown = null;
        if (treeRaw?.json) {
          try {
            parsedTree = JSON.parse(treeRaw.json);
          } catch {
            parsedTree = null;
          }
        }

        // Accepts either the new graph shape or the legacy {recruiting:{...}} shape,
        // auto-converting the latter without losing any committed data.
        const graph = coerceToGraph(parsedTree);
        if (graph) {
          setCsvTree(graph);
          setCsvStaging(cloneTree(graph));
          if (!("nodes" in (parsedTree as Record<string, unknown>))) {
            toast.info("Converted the saved metrics tree to the new graph format.");
          }
        } else {
          const seed = buildSeedTree();
          setCsvTree(seed);
          setCsvStaging(cloneTree(seed));
          try {
            await saveTree({ data: { tree: seed } });
            toast.success("Seeded a fresh metrics tree and saved it to the data repo.");
          } catch (e) {
            toast.warning(
              `Seeded a fresh metrics tree, but saving it failed: ${
                e instanceof Error ? e.message : String(e)
              }`,
            );
          }
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const loadCsv = useCallback(async (path: string) => {
    const cached = csvCache.current.get(path);
    if (cached) return cached;
    const inflight = csvInflight.current.get(path);
    if (inflight) return inflight;
    const p = (async () => {
      const rows = await getGameCsv({ data: { path } });
      csvCache.current.set(path, rows);
      setCsvLoadedCount((n) => n + 1);
      return rows;
    })();
    csvInflight.current.set(path, p);
    try {
      return await p;
    } finally {
      csvInflight.current.delete(path);
    }
  }, []);

  // Eagerly warm every game's CSV (and therefore the full roster) in the background as
  // soon as the game list is known, so the router has the roster for the first question.
  useEffect(() => {
    if (loading) return;
    if (csvGames.length === 0) {
      warmupRef.current = Promise.resolve();
      setRosterLoading(false);
      return;
    }
    if (warmupRef.current) return;
    const run = (async () => {
      await Promise.all(
        csvGames.map((g) =>
          loadCsv(g.path).catch(() => {
            /* a single unreadable match shouldn't block the rest */
          }),
        ),
      );
    })();
    warmupRef.current = run;
    void run.finally(() => setRosterLoading(false));
  }, [loading, csvGames, loadCsv]);

  /* ---------------- player-analysis boot (lazy, on first switch) ---------------- */

  const startDvw = useCallback(() => {
    if (dvwStarted.current) return dvwReadyRef.current ?? Promise.resolve();
    dvwStarted.current = true;
    setDvwLoading(true);
    const run = (async () => {
      const warnings: string[] = [];
      let files: DvwFileRef[] = [];
      try {
        files = await listDvwFiles();
      } catch (e) {
        warnings.push(
          `Couldn't list the play-by-play files: ${e instanceof Error ? e.message : String(e)}`,
        );
      }
      if (files.length === 0) {
        warnings.push(
          "No .dvw match files found — upload them to the private data repo's dvw/ folder.",
        );
      }
      // Parsed on the client and cached by path, exactly like the CSV path.
      const parsed = await Promise.all(
        files.map(async (f) => {
          try {
            const { text } = await getDvwFile({ data: { path: f.path } });
            return parseDvw(text, f.path, f.filename);
          } catch (e) {
            warnings.push(
              `Couldn't read ${f.filename}: ${e instanceof Error ? e.message : String(e)}`,
            );
            return null;
          }
        }),
      );
      const matches = parsed.filter((m): m is DvwMatch => !!m);
      setDvwMatches(matches);
      setDvwWarnings(warnings);
      // Default to whichever team appears in the most files.
      setTeamState((prev) => prev ?? teamCounts(matches)[0]?.team ?? null);
    })();
    dvwReadyRef.current = run.finally(() => setDvwLoading(false));
    return dvwReadyRef.current;
  }, []);

  useEffect(() => {
    if (workspace === "player") void startDvw();
  }, [workspace, startDvw]);

  /* ---------------- active source ---------------- */

  const csvSource = useMemo(() => {
    void csvLoadedCount; // rebuild as files stream in
    return buildCsvSource(
      csvGames.map((g) => ({ opponent: g.opponent, rows: csvCache.current.get(g.path) ?? [] })),
    );
  }, [csvGames, csvLoadedCount]);

  const dvwBuild = useMemo(() => buildDvwSource(dvwMatches, team), [dvwMatches, team]);

  const dvwSource = dvwBuild.source;
  const source: Source | null = workspace === "player" ? dvwSource : csvSource;
  const schema = source?.schema ?? null;

  /* ---------------- play-by-play knowledge tree ---------------- */

  useEffect(() => {
    if (workspace !== "player" || dvwLoading || dvwTreeSeeded.current) return;
    // Never seed before the actions are loaded — against an empty schema every
    // primitive leaf would validate away and the tree would save as bare branches.
    if (!schema || dvwMatches.length === 0) return;
    if (!dvwSource || dvwSource.facts().length === 0) return;
    dvwTreeSeeded.current = true;
    (async () => {
      const raw = await getVolleyTree().catch(() => null);
      let parsed: unknown = null;
      if (raw?.json) {
        try {
          parsed = JSON.parse(raw.json);
        } catch {
          parsed = null;
        }
      }
      if (isGraphTree(parsed)) {
        setDvwTree(parsed);
        setDvwStaging(cloneTree(parsed));
        return;
      }
      const seed = buildDvwSeedTree(schema);
      setDvwTree(seed);
      setDvwStaging(cloneTree(seed));
      try {
        await saveVolleyTree({ data: { tree: seed } });
        toast.success("Seeded a fresh play-by-play metrics tree and saved it to the data repo.");
      } catch (e) {
        toast.warning(
          `Seeded a play-by-play metrics tree, but saving it failed: ${
            e instanceof Error ? e.message : String(e)
          }`,
        );
      }
    })().catch(() => {
      dvwTreeSeeded.current = false;
    });
  }, [workspace, dvwLoading, schema, dvwMatches, dvwSource]);

  /* ---------------- workspace-scoped switching ---------------- */

  const [workspaceKey, setWorkspaceKey] = useState("recruiting");

  const setWorkspace = useCallback((w: Workspace) => {
    setWorkspaceState(w);
    setPrefill(null);
    setBranchFocus(null);
    setWorkspaceKey(`${w}|${Date.now()}`);
  }, []);

  const setTeam = useCallback((t: string) => {
    setTeamState(t);
    setPrefill(null);
    setBranchFocus(null);
    // The roster and schema are re-derived for the new team, so nothing may carry over.
    dvwTreeSeeded.current = false;
    setWorkspaceKey(`player|${t}|${Date.now()}`);
  }, []);

  const roster = useMemo(() => source?.playerLabels() ?? [], [source]);
  useEffect(() => {
    rosterRef.current = roster;
  }, [roster]);

  const rosterReady = useCallback(async () => {
    if (workspace === "player") {
      await startDvw();
      await dvwReadyRef.current;
      return rosterRef.current;
    }
    // The warm-up may not have started yet (boot still resolving the game list).
    for (let i = 0; i < 200 && !warmupRef.current; i += 1) {
      await new Promise((r) => setTimeout(r, 50));
    }
    if (warmupRef.current) await warmupRef.current;
    return rosterRef.current;
  }, [workspace, startDvw]);

  const committed = workspace === "player" ? dvwTree : csvTree;
  const staging = workspace === "player" ? dvwStaging : csvStaging;
  const setCommitted = workspace === "player" ? setDvwTree : setCsvTree;
  const setStaging = workspace === "player" ? setDvwStaging : setCsvStaging;

  const persistCommitted = useCallback(
    async (t: GraphTree) => {
      try {
        if (workspace === "player") await saveVolleyTree({ data: { tree: t } });
        else await saveTree({ data: { tree: t } });
        return { ok: true };
      } catch (e) {
        return { ok: false, error: e instanceof Error ? e.message : String(e) };
      }
    },
    [workspace],
  );

  const warnings = workspace === "player" ? [...dvwWarnings, ...dvwBuild.warnings] : [];

  const value = useMemo<StoreValue>(
    () => ({
      loading: loading || (workspace === "player" && dvwLoading),
      bootError,
      warnings,
      workspace,
      setWorkspace,
      workspaceKey,
      team,
      setTeam,
      teamOptions: teamCounts(dvwMatches).map((t) => t.team),
      source,
      grain: source?.grain ?? "measure",
      schema,
      gameLabels: source?.gameLabels() ?? [],
      setLabels: source?.setLabels() ?? [],
      hasSetAxis: !!source && supportsSets(source),
      csvGames,
      loadCsv,
      committed,
      staging,
      setCommitted,
      setStaging,
      persistCommitted,
      roster,
      rosterLoading: workspace === "player" ? dvwLoading : rosterLoading,
      rosterReady,
      prefill,
      setPrefill,
      branchFocus,
      setBranchFocus,
    }),
    [
      loading,
      dvwLoading,
      bootError,
      warnings,
      workspace,
      setWorkspace,
      workspaceKey,
      team,
      setTeam,
      dvwMatches,
      source,
      schema,
      csvGames,
      loadCsv,
      committed,
      staging,
      setCommitted,
      setStaging,
      persistCommitted,
      roster,
      rosterLoading,
      rosterReady,
      prefill,
      branchFocus,
    ],
  );

  return <StoreContext.Provider value={value}>{children}</StoreContext.Provider>;
}

export function useStore(): StoreValue {
  const ctx = useContext(StoreContext);
  if (!ctx) throw new Error("useStore must be used inside VolleyProvider");
  return ctx;
}
