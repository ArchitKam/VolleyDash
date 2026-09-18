import { useCallback, useEffect, useMemo, useState } from "react";
import { toast } from "sonner";
import { ChevronDown, Loader2 } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import type { CsvRow } from "@/lib/csv";
import { computeWorkedExample, type WorkedExample } from "@/lib/evaluate";
import {
  authorEventMetric,
  authorMetric,
  RULE_EXAMPLE_PHRASINGS,
  type MetricProposal,
} from "@/lib/metricAuthor";
import { useStore } from "@/lib/store";
import {
  addEdge,
  applySpecToNode,
  childrenOf,
  cloneTree,
  DEPENDENTS_STUB,
  diffTrees,
  findNode,
  metricNodes,
  nodeDescription,
  nodeExpression,
  nodeFromSpec,
  nodeList,
  orphanNodes,
  parentsOf,
  removeEdge,
  removeNode,
  ROOT_LABEL,
  specFromParts,
  topLevelCategories,
  upsertNode,
  validateEdge,
  validateSpec,
  type GraphNode,
  type GraphTree,
  categoryLabels,
  specOfNode,
  type MetricSpec,
} from "@/lib/tree";

const AUTHOR = "justin";

function outline(graph: GraphTree): string {
  const lines: string[] = [];
  const seen = new Set<string>();
  const walk = (id: string, depth: number, viaParentId: string | null) => {
    const node = graph.nodes[id];
    if (!node) return;
    const pad = "  ".repeat(depth);
    const parents = parentsOf(graph, id)
      .filter((p) => p.node_id !== viaParentId)
      .map((p) => p.label);
    const repeat = seen.has(id);
    const tag = depth === 0 ? "[root]" : node.kind ? "[metric]" : "[category]";
    const also = parents.length > 0 ? ` (also under ${parents.join(", ")})` : "";
    const desc = nodeDescription(node) ? ` — ${nodeDescription(node)}` : "";
    lines.push(`${pad}${tag} ${node.label}${desc}${also}${repeat ? " [repeat appearance]" : ""}`);
    seen.add(id);
    for (const child of childrenOf(graph, id)) walk(child.node_id, depth + 1, id);
  };
  walk(graph.root_id, 0, null);
  return lines.join("\n");
}

export function KbTab() {
  const {
    committed,
    staging,
    setCommitted,
    setStaging,
    csvGames,
    loadCsv,
    grain,
    schema,
    workspaceKey,
    persistCommitted,
    prefill,
    setPrefill,
    branchFocus,
    setBranchFocus,
  } = useStore();

  const [path, setPath] = useState<string[]>([]);
  const [firstGameRows, setFirstGameRows] = useState<CsvRow[]>([]);
  const [editing, setEditing] = useState<{ expr: string; description: string } | null>(null);
  const [pendingDelete, setPendingDelete] = useState<{ nodeId: string; label: string } | null>(
    null,
  );
  const [newParent, setNewParent] = useState("");
  const [showAddParent, setShowAddParent] = useState(false);

  const [searchLabel, setSearchLabel] = useState("");
  const [searchResults, setSearchResults] = useState<GraphNode[]>([]);

  const [step, setStep] = useState<"describe" | "review">("describe");
  const [phrase, setPhrase] = useState("");
  const [interpreting, setInterpreting] = useState(false);
  const [authorError, setAuthorError] = useState<string | null>(null);
  const [triedRules, setTriedRules] = useState(false);
  const [draft, setDraft] = useState({
    expr: "",
    description: "",
    label: "",
    aliases: "",
    parents: [] as string[],
  });
  const [example, setExample] = useState<WorkedExample | null>(null);
  /** Set when the event-grain author produced a spec we must stage verbatim. */
  const [authoredSpec, setAuthoredSpec] = useState<MetricSpec | null>(null);
  const [merging, setMerging] = useState(false);

  useEffect(() => {
    if (grain !== "measure") return;
    const first = csvGames[0];
    if (!first) return;
    loadCsv(first.path)
      .then(setFirstGameRows)
      .catch(() => setFirstGameRows([]));
  }, [grain, csvGames, loadCsv]);

  useEffect(() => {
    if (!prefill || !committed) return;
    const target = prefill.branch ? findNode(committed, prefill.branch) : null;
    setDraft({
      expr: prefill.expr,
      description: prefill.description,
      label: prefill.label,
      aliases: prefill.aliases,
      parents: target ? [target.node_id] : [],
    });
    setStep("review");
    setPrefill(null);
  }, [prefill, committed, setPrefill]);

  const validationCtx = useMemo(
    () => ({ grain, schema: schema ?? undefined }),
    [grain, schema],
  );

  // Nothing from one workspace may carry over into the other.
  useEffect(() => {
    setPath([]);
    setEditing(null);
    setStep("describe");
    setPhrase("");
    setAuthorError(null);
    setAuthoredSpec(null);
    setDraft({ expr: "", description: "", label: "", aliases: "", parents: [] });
    setExample(null);
    setSearchLabel("");
    setSearchResults([]);
  }, [workspaceKey]);

  const recomputeExample = useCallback(
    (expr: string, description: string) => {
      if (!committed || grain !== "measure") return;
      setExample(computeWorkedExample(expr, firstGameRows, committed, description));
    },
    [committed, firstGameRows, grain],
  );

  useEffect(() => {
    if (step === "review" && draft.expr) recomputeExample(draft.expr, draft.description);
  }, [step, draft.expr, draft.description, recomputeExample]);

  const diff = useMemo(
    () => (committed && staging ? diffTrees(staging, committed) : []),
    [committed, staging],
  );
  const orphans = useMemo(() => (staging ? orphanNodes(staging) : []), [staging]);

  // Expensive per-render work, memoized on the graph/selection instead of every keystroke.
  const selectedNode = committed
    ? (committed.nodes[path[path.length - 1] ?? ""] ?? null)
    : null;
  const quickExample = useMemo(
    () =>
      committed && selectedNode && selectedNode.kind && grain === "measure"
        ? computeWorkedExample(
            nodeExpression(selectedNode),
            firstGameRows,
            committed,
            nodeDescription(selectedNode),
          )
        : null,
    [committed, selectedNode, firstGameRows, grain],
  );
  const committedOutline = useMemo(() => (committed ? outline(committed) : ""), [committed]);
  const stagedMetricCount = useMemo(() => (staging ? metricNodes(staging).length : 0), [staging]);

  if (!committed || !staging) return null;

  const selectedId = path[path.length - 1] ?? null;
  const selected = selectedId ? (committed.nodes[selectedId] ?? null) : null;
  const viaParentId =
    path.length >= 2 ? path[path.length - 2]! : path.length === 1 ? committed.root_id : null;
  const viaParent = viaParentId ? (committed.nodes[viaParentId] ?? null) : null;
  const breadcrumb = [ROOT_LABEL, ...path.map((id) => committed.nodes[id]?.label ?? id)];

  const stageEdit = (nodeId: string, expr: string, description: string, label: string) => {
    const existingNode = staging.nodes[nodeId];
    const existingSpec = existingNode ? specOfNode(existingNode) : null;
    const spec =
      grain === "event" && existingSpec && !expr.includes("[")
        ? { ...existingSpec, human_description: description }
        : specFromParts(expr, description);
    const err = validateSpec(spec, committed, validationCtx);
    if (err) {
      toast.error(err);
      return false;
    }
    const existing = staging.nodes[nodeId];
    if (!existing) return false;
    const updated: GraphNode = applySpecToNode(existing, spec, label, description);
    setStaging(upsertNode(staging, updated));
    toast.success("Edit staged. Merge below to finalize.");
    return true;
  };

  const stageDelete = (nodeId: string) => {
    const next = removeNode(staging, nodeId);
    setStaging(next);
    const newOrphans = orphanNodes(next).filter((n) =>
      orphanNodes(staging).every((o) => o.node_id !== n.node_id),
    );
    toast.success(
      newOrphans.length > 0
        ? `Deletion staged. Now orphaned: ${newOrphans.map((n) => n.label).join(", ")}.`
        : "Deletion staged. Merge below to finalize.",
    );
  };

  const stageUnlink = (parentId: string, childId: string) => {
    const next = removeEdge(staging, parentId, childId);
    setStaging(next);
    const orphaned = orphanNodes(next).some((n) => n.node_id === childId);
    toast.success(
      orphaned
        ? `Unlink staged — "${staging.nodes[childId]?.label}" now has no parent.`
        : "Unlink staged. Merge below to finalize.",
    );
  };

  const stageLink = (parentId: string, childId: string) => {
    const err = validateEdge(staging, parentId, childId);
    if (err) {
      toast.error(err);
      return;
    }
    setStaging(addEdge(staging, parentId, childId));
    toast.success("New parent staged. Merge below to finalize.");
  };

  const interpret = async () => {
    setInterpreting(true);
    setAuthorError(null);
    setTriedRules(false);
    try {
      if (grain === "event" && schema) {
        const ev = await authorEventMetric(phrase, committed, schema, categoryLabels(committed));
        if (!ev.matched) {
          setAuthorError(ev.message);
          setTriedRules(false);
          return;
        }
        const focusEv = branchFocus ? findNode(committed, branchFocus) : null;
        const suggestedEv = findNode(committed, ev.suggested_branch_group);
        const initialEv = focusEv ?? suggestedEv ?? topLevelCategories(committed)[0] ?? null;
        setAuthoredSpec(ev.spec);
        setDraft({
          expr: ev.display,
          description: ev.human_description,
          label: ev.suggested_label || phrase,
          aliases: ev.suggested_aliases.join(", "),
          parents: initialEv ? [initialEv.node_id] : [],
        });
        setStep("review");
        return;
      }
      const result = await authorMetric(phrase, committed);
      if (!result.matched) {
        setAuthorError(result.message);
        setTriedRules(result.triedRules);
        return;
      }
      setAuthoredSpec(null);
      const proposal = result as MetricProposal;
      const focus = branchFocus ? findNode(committed, branchFocus) : null;
      const suggested = findNode(committed, proposal.suggested_branch_group);
      const initial = focus ?? suggested ?? topLevelCategories(committed)[0] ?? null;
      setDraft({
        expr: proposal.expr,
        description: proposal.human_description,
        label: proposal.suggested_label || phrase,
        aliases: proposal.suggested_aliases.join(", "),
        parents: initial ? [initial.node_id] : [],
      });
      recomputeExample(proposal.expr, proposal.human_description);
      setStep("review");
    } finally {
      setInterpreting(false);
    }
  };

  const confirmStage = () => {
    const spec =
      authoredSpec && grain === "event"
        ? { ...authoredSpec, human_description: draft.description }
        : specFromParts(draft.expr, draft.description);
    const err = validateSpec(spec, committed, validationCtx);
    if (err) {
      toast.error(err);
      return;
    }
    if (!draft.label.trim()) {
      toast.error("Give the metric a label.");
      return;
    }
    if (draft.parents.length === 0) {
      toast.error("Pick at least one parent node.");
      return;
    }
    const node = nodeFromSpec(spec, {
      label: draft.label.trim(),
      aliases: draft.aliases
        .split(",")
        .map((a) => a.trim())
        .filter(Boolean),
      authored_by: AUTHOR,
      description: draft.description,
    });
    let next = upsertNode(cloneTree(staging), node);
    for (const parentId of draft.parents) {
      const edgeErr = validateEdge(next, parentId, node.node_id);
      if (edgeErr) {
        toast.error(edgeErr);
        return;
      }
      next = addEdge(next, parentId, node.node_id);
    }
    setStaging(next);
    toast.success(`Staged "${node.label}". Merge below to finalize.`);
    setStep("describe");
    setPhrase("");
    setDraft({ expr: "", description: "", label: "", aliases: "", parents: [] });
    setAuthoredSpec(null);
    setBranchFocus(null);
  };

  const mergeAll = async () => {
    setMerging(true);
    const next = cloneTree(staging);
    setCommitted(next);
    const res = await persistCommitted(next);
    setMerging(false);
    if (res.ok) toast.success(`Merged ${diff.length} change(s) into committed tree.`);
    else
      toast.warning(
        `Merged for this session only — saving to the data repo failed and this won't survive a reload. ${res.error ?? ""}`,
      );
  };

  const currentLevelId = selectedId ?? committed.root_id;
  const currentChildren = childrenOf(committed, currentLevelId);
  const selectedParents = selected ? parentsOf(committed, selected.node_id) : [];
  const childrenOfSelected = selected ? childrenOf(committed, selected.node_id) : [];
  const allNodesSorted = nodeList(committed).sort((a, b) => a.label.localeCompare(b.label));

  return (
    <div className="space-y-6">
      <section className="space-y-1">
        <h3 className="section-heading text-lg">Current Metrics Graph (committed)</h3>
        <p className="text-xs text-muted-foreground">
          Persisted to the private VolleyData repo — updates on merge. A node can sit under more
          than one parent.
        </p>
      </section>

      <section className="space-y-3 rounded-lg border border-border bg-card p-4">
        <div
          className="mx-auto w-fit bg-primary px-6 py-3 text-sm font-bold text-primary-foreground"
          style={{
            clipPath: "polygon(25% 0%, 75% 0%, 100% 50%, 75% 100%, 25% 100%, 0% 50%)",
          }}
        >
          {ROOT_LABEL}
        </div>

        <div className="text-xs text-muted-foreground">{breadcrumb.join(" → ")}</div>
        {path.length > 0 && (
          <div className="flex gap-2">
            <Button size="sm" variant="ghost" onClick={() => setPath(path.slice(0, -1))}>
              ← Back
            </Button>
            <Button size="sm" variant="ghost" onClick={() => setPath([])}>
              Back to root
            </Button>
          </div>
        )}

        {currentChildren.length > 0 && (
          <div className="flex flex-wrap justify-center gap-2">
            {currentChildren.map((child) => (
              <Button
                key={child.node_id}
                size="sm"
                variant={child.kind ? "secondary" : "outline"}
                onClick={() => {
                  setPath([...path, child.node_id]);
                  setEditing(null);
                  setShowAddParent(false);
                }}
              >
                {child.label}
                {childrenOf(committed, child.node_id).length > 0 ? " ›" : ""}
              </Button>
            ))}
          </div>
        )}

        {selected && (
          <div className="space-y-2 rounded border border-gold/50 p-3 text-sm">
            <div className="text-base font-semibold text-gold">{selected.label}</div>
            <div className="text-xs text-muted-foreground">
              Path taken: {breadcrumb.join(" → ")}
              {selectedParents.length > 1 ? " (other paths also lead here)" : ""}
            </div>
            <div className="text-xs text-muted-foreground">authored_by: {selected.authored_by}</div>

            <div className="space-y-1">
              <div className="text-xs text-muted-foreground">Parents</div>
              {selectedParents.length === 0 && (
                <div className="text-xs text-primary">
                  No parents — this node is unreachable from the root.
                </div>
              )}
              {selectedParents.map((p) => (
                <div key={p.node_id} className="flex items-center justify-between gap-2">
                  <span>{p.label}</span>
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => stageUnlink(p.node_id, selected.node_id)}
                  >
                    Unlink from here
                  </Button>
                </div>
              ))}
              {showAddParent ? (
                <div className="flex gap-2">
                  <Select value={newParent} onValueChange={setNewParent}>
                    <SelectTrigger>
                      <SelectValue placeholder="Pick a node" />
                    </SelectTrigger>
                    <SelectContent>
                      {allNodesSorted
                        .filter((n) => n.node_id !== selected.node_id)
                        .map((n) => (
                          <SelectItem key={n.node_id} value={n.node_id}>
                            {n.label}
                          </SelectItem>
                        ))}
                    </SelectContent>
                  </Select>
                  <Button
                    size="sm"
                    onClick={() => {
                      if (!newParent) return;
                      stageLink(newParent, selected.node_id);
                      setNewParent("");
                      setShowAddParent(false);
                    }}
                  >
                    Add
                  </Button>
                  <Button size="sm" variant="ghost" onClick={() => setShowAddParent(false)}>
                    Cancel
                  </Button>
                </div>
              ) : (
                <Button size="sm" variant="outline" onClick={() => setShowAddParent(true)}>
                  + Add another parent
                </Button>
              )}
            </div>

            {selected.kind && (
              <>
                <div>{nodeDescription(selected)}</div>
                {grain === "measure" && (
                <div className="text-xs">
                  Quick example ({csvGames[0]?.opponent ?? "no game data"}):{" "}
                  {quickExample
                    ? quickExample.error
                      ? quickExample.error
                      : `${quickExample.player} → ${quickExample.substituted} = ${
                          quickExample.value === null ? "—" : quickExample.value.toFixed(3)
                        }`
                    : "no player rows available"}
                  {quickExample?.note ? ` (${quickExample.note})` : ""}
                </div>
                )}
                <pre className="overflow-x-auto rounded bg-black/60 p-2 text-xs text-primary">
                  {nodeExpression(selected)}
                </pre>
              </>
            )}
            {!selected.kind && (
              <div className="text-xs text-muted-foreground">
                Category node — no formula of its own.
              </div>
            )}

            {editing ? (
              <div className="space-y-2">
                <Input
                  value={editing.expr}
                  onChange={(e) => setEditing({ ...editing, expr: e.target.value })}
                  placeholder="Formula or column reference"
                />
                <Textarea
                  value={editing.description}
                  onChange={(e) => setEditing({ ...editing, description: e.target.value })}
                  placeholder="Description"
                />
                <div className="flex gap-2">
                  <Button
                    size="sm"
                    onClick={() => {
                      const ok = stageEdit(
                        selected.node_id,
                        editing.expr,
                        editing.description,
                        selected.label,
                      );
                      if (ok) setEditing(null);
                    }}
                  >
                    Save
                  </Button>
                  <Button size="sm" variant="ghost" onClick={() => setEditing(null)}>
                    Cancel
                  </Button>
                </div>
              </div>
            ) : (
              <div className="flex flex-wrap gap-2">
                {selected.kind && (
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() =>
                      setEditing({
                        expr: nodeExpression(selected),
                        description: nodeDescription(selected),
                      })
                    }
                  >
                    Edit
                  </Button>
                )}
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => {
                    setBranchFocus(selected.label);
                    setStep("describe");
                    document
                      .getElementById("add-metric-wizard")
                      ?.scrollIntoView({ behavior: "smooth" });
                    toast.info(`New metrics will be filed under ${selected.label}.`);
                  }}
                >
                  + Add a metric under {selected.label}
                </Button>
                {viaParent && (
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => stageUnlink(viaParent.node_id, selected.node_id)}
                  >
                    Unlink from {viaParent.label}
                  </Button>
                )}
                <Button
                  size="sm"
                  variant="destructive"
                  onClick={() =>
                    setPendingDelete({ nodeId: selected.node_id, label: selected.label })
                  }
                >
                  Delete entirely
                </Button>
              </div>
            )}
          </div>
        )}

        {pendingDelete && (
          <div className="space-y-2 rounded border border-primary bg-black/40 p-3 text-sm">
            <div>
              Delete {pendingDelete.label} entirely? This removes the node and every link to it —
              children are NOT deleted.
            </div>
            {(() => {
              const kids = childrenOf(staging, pendingDelete.nodeId).filter(
                (k) => parentsOf(staging, k.node_id).length <= 1,
              );
              return kids.length > 0 ? (
                <div className="text-xs text-gold">
                  These children would be left orphaned: {kids.map((k) => k.label).join(", ")}
                </div>
              ) : null;
            })()}
            <div className="text-xs text-muted-foreground">
              This only stages the removal — Merge All Staged Changes below to finalize.
            </div>
            <div className="flex gap-2">
              <Button
                size="sm"
                variant="destructive"
                onClick={() => {
                  stageDelete(pendingDelete.nodeId);
                  setPendingDelete(null);
                  setPath((p) => p.filter((id) => id !== pendingDelete.nodeId));
                }}
              >
                Yes
              </Button>
              <Button size="sm" variant="ghost" onClick={() => setPendingDelete(null)}>
                Cancel
              </Button>
            </div>
          </div>
        )}

        {childrenOfSelected.length > 0 && (
          <p className="text-center text-xs text-muted-foreground">
            {childrenOfSelected.length} node(s) filed under {selected?.label} — shown above.
          </p>
        )}
      </section>

      {orphans.length > 0 && (
        <section className="space-y-2 rounded-lg border border-primary bg-card p-4 text-sm">
          <div className="font-semibold text-primary">Orphaned / unreachable from root</div>
          <p className="text-xs text-muted-foreground">
            These staged nodes have no parent left. Give them a new parent or delete them.
          </p>
          {orphans.map((n) => (
            <div key={n.node_id} className="flex items-center justify-between gap-2">
              <span>
                {n.label} — <span className="text-muted-foreground">{nodeDescription(n)}</span>
              </span>
              <div className="flex gap-2">
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => {
                    const root = staging.root_id;
                    stageLink(root, n.node_id);
                  }}
                >
                  Re-home under root
                </Button>
                <Button
                  size="sm"
                  variant="destructive"
                  onClick={() => setPendingDelete({ nodeId: n.node_id, label: n.label })}
                >
                  Delete entirely
                </Button>
              </div>
            </div>
          ))}
        </section>
      )}

      <Collapsible>
        <CollapsibleTrigger asChild>
          <Button variant="outline" size="sm">
            <ChevronDown className="mr-1 size-4" /> Delete a metric by name
          </Button>
        </CollapsibleTrigger>
        <CollapsibleContent className="mt-2 space-y-2 rounded-lg border border-border bg-card p-3">
          <div className="flex gap-2">
            <Input
              value={searchLabel}
              onChange={(e) => setSearchLabel(e.target.value)}
              placeholder="Exact metric label"
            />
            <Button
              size="sm"
              onClick={() =>
                setSearchResults(nodeList(committed).filter((n) => n.label === searchLabel.trim()))
              }
            >
              Find
            </Button>
          </div>
          {searchResults.map((n) => (
            <div key={n.node_id} className="space-y-1 border-l-2 border-primary pl-2 text-sm">
              <div>
                {n.label} —{" "}
                <span className="text-muted-foreground">
                  parents:{" "}
                  {parentsOf(committed, n.node_id)
                    .map((p) => p.label)
                    .join(", ") || "none"}
                </span>
              </div>
              <div className="flex flex-wrap gap-2">
                {parentsOf(committed, n.node_id).map((p) => (
                  <Button
                    key={p.node_id}
                    size="sm"
                    variant="outline"
                    onClick={() => stageUnlink(p.node_id, n.node_id)}
                  >
                    Unlink from {p.label}
                  </Button>
                ))}
                <Button
                  size="sm"
                  variant="destructive"
                  onClick={() => setPendingDelete({ nodeId: n.node_id, label: n.label })}
                >
                  Delete entirely
                </Button>
              </div>
            </div>
          ))}
        </CollapsibleContent>
      </Collapsible>

      <Collapsible>
        <CollapsibleTrigger asChild>
          <Button variant="outline" size="sm">
            <ChevronDown className="mr-1 size-4" /> Browse as a plain list
          </Button>
        </CollapsibleTrigger>
        <CollapsibleContent className="mt-2 space-y-3 rounded-lg border border-border bg-card p-3 text-sm">
          {nodeList(committed)
            .filter((n) => n.node_id !== committed.root_id)
            .sort((a, b) => a.label.localeCompare(b.label))
            .map((n) => (
              <div key={n.node_id} className="space-y-1 border-b border-border/40 pb-2">
                <div>
                  {n.label} — <span className="text-muted-foreground">{nodeDescription(n)}</span>
                </div>
                <div className="text-xs text-muted-foreground">
                  parents:{" "}
                  {parentsOf(committed, n.node_id)
                    .map((p) => p.label)
                    .join(", ") || "none"}
                </div>
                <div className="flex flex-wrap gap-2">
                  {parentsOf(committed, n.node_id).map((p) => (
                    <Button
                      key={p.node_id}
                      size="sm"
                      variant="outline"
                      onClick={() => stageUnlink(p.node_id, n.node_id)}
                    >
                      Unlink from {p.label}
                    </Button>
                  ))}
                  <Button
                    size="sm"
                    variant="destructive"
                    onClick={() => setPendingDelete({ nodeId: n.node_id, label: n.label })}
                  >
                    Delete entirely
                  </Button>
                </div>
              </div>
            ))}
        </CollapsibleContent>
      </Collapsible>

      <Collapsible>
        <CollapsibleTrigger asChild>
          <Button variant="outline" size="sm">
            <ChevronDown className="mr-1 size-4" /> Raw text view
          </Button>
        </CollapsibleTrigger>
        <CollapsibleContent className="mt-2 rounded-lg border border-border bg-card p-3">
          <pre className="overflow-x-auto text-xs">{committedOutline}</pre>
        </CollapsibleContent>
      </Collapsible>

      <section id="add-metric-wizard" className="space-y-3">
        <h3 className="section-heading text-lg">Add a New Metric</h3>
        {step === "describe" ? (
          <div className="space-y-2 rounded-lg border border-border bg-card p-4">
            <Input
              value={phrase}
              onChange={(e) => setPhrase(e.target.value)}
              placeholder="Describe the stat in plain language, e.g. net kills per set"
            />
            <Button
              size="sm"
              onClick={() => void interpret()}
              disabled={interpreting || !phrase.trim()}
            >
              {interpreting ? <Loader2 className="mr-1 size-4 animate-spin" /> : null}
              Interpret
            </Button>
            {branchFocus && (
              <p className="text-xs text-gold">Parent node pre-selected: {branchFocus}</p>
            )}
            {authorError && (
              <div className="space-y-1 text-sm text-primary">
                <div>{authorError}</div>
                {triedRules && (
                  <div className="text-xs text-muted-foreground">
                    Phrasings the built-in fallback understands: {RULE_EXAMPLE_PHRASINGS.join("; ")}
                  </div>
                )}
              </div>
            )}
          </div>
        ) : (
          <div className="space-y-3 rounded-lg border border-gold/60 bg-card p-4">
            <div className="grid gap-2 md:grid-cols-2">
              <div>
                <div className="mb-1 text-xs text-muted-foreground">Formula</div>
                <Input
                  value={draft.expr}
                  onChange={(e) => setDraft({ ...draft, expr: e.target.value })}
                />
              </div>
              <div>
                <div className="mb-1 text-xs text-muted-foreground">Metric label</div>
                <Input
                  value={draft.label}
                  onChange={(e) => setDraft({ ...draft, label: e.target.value })}
                />
              </div>
              <div className="md:col-span-2">
                <div className="mb-1 text-xs text-muted-foreground">Description</div>
                <Textarea
                  value={draft.description}
                  onChange={(e) => setDraft({ ...draft, description: e.target.value })}
                />
              </div>
              <div>
                <div className="mb-1 text-xs text-muted-foreground">Aliases (comma-separated)</div>
                <Input
                  value={draft.aliases}
                  onChange={(e) => setDraft({ ...draft, aliases: e.target.value })}
                />
              </div>
              <div className="md:col-span-2">
                <div className="mb-1 text-xs text-muted-foreground">
                  Parent node(s) — pick at least one
                </div>
                <div className="flex max-h-48 flex-wrap gap-2 overflow-y-auto rounded border border-border/60 p-2">
                  {allNodesSorted.map((n) => {
                    const on = draft.parents.includes(n.node_id);
                    return (
                      <Button
                        key={n.node_id}
                        size="sm"
                        variant={on ? "default" : "outline"}
                        onClick={() =>
                          setDraft({
                            ...draft,
                            parents: on
                              ? draft.parents.filter((p) => p !== n.node_id)
                              : [...draft.parents, n.node_id],
                          })
                        }
                      >
                        {on ? "✓ " : ""}
                        {n.label}
                      </Button>
                    );
                  })}
                </div>
              </div>
            </div>

            <div className="rounded border border-border/60 p-2 text-sm">
              <div className="text-xs text-muted-foreground">Worked example</div>
              {example ? (
                example.error ? (
                  <div className="text-primary">{example.error}</div>
                ) : (
                  <div>
                    {example.player} → {example.substituted} ={" "}
                    {example.value === null ? "—" : example.value.toFixed(3)}
                    {example.fallbackZeroed.length > 0 && (
                      <div className="text-xs text-muted-foreground">
                        Fallback-zeroed columns: {example.fallbackZeroed.join(", ")}
                      </div>
                    )}
                    {example.note && (
                      <div className="text-xs text-muted-foreground">{example.note}</div>
                    )}
                  </div>
                )
              ) : (
                <div className="text-muted-foreground">No player rows available yet.</div>
              )}
              <Button
                size="sm"
                variant="outline"
                className="mt-2"
                onClick={() => recomputeExample(draft.expr, draft.description)}
              >
                Recompute
              </Button>
            </div>

            <div className="flex gap-2">
              <Button size="sm" onClick={confirmStage}>
                Confirm &amp; Stage
              </Button>
              <Button
                size="sm"
                variant="ghost"
                onClick={() => {
                  setStep("describe");
                  setDraft({ expr: "", description: "", label: "", aliases: "", parents: [] });
                  setExample(null);
                }}
              >
                Cancel and start over
              </Button>
            </div>
          </div>
        )}
      </section>

      <section className="space-y-3">
        <h3 className="section-heading text-lg">Review Staged Changes (Diff)</h3>
        <div className="space-y-3 rounded-lg border border-border bg-card p-4 text-sm">
          {diff.length === 0 && <p className="text-muted-foreground">Nothing staged.</p>}
          {(["Added", "Edited", "Linked", "Unlinked", "Deleted"] as const).map((type) => {
            const entries = diff.filter((d) => d.type === type);
            if (entries.length === 0) return null;
            return (
              <div key={type} className="space-y-1">
                <div className="font-semibold text-gold">{type}</div>
                {entries.map((e, i) => (
                  <div key={`${e.label}-${i}`} className="border-l-2 border-primary pl-2">
                    <div>
                      {e.label} <span className="text-muted-foreground">({e.branch})</span>
                    </div>
                    {e.detail && <div className="text-xs text-muted-foreground">{e.detail}</div>}
                    {e.before && (
                      <div className="text-xs text-muted-foreground">before: {e.before}</div>
                    )}
                    {e.after && (
                      <div className="text-xs text-muted-foreground">after: {e.after}</div>
                    )}
                    {type !== "Added" && (
                      <div className="text-xs text-muted-foreground italic">{DEPENDENTS_STUB}</div>
                    )}
                  </div>
                ))}
              </div>
            );
          })}
          {orphans.length > 0 && (
            <div className="space-y-1">
              <div className="font-semibold text-primary">Orphaned / unreachable from root</div>
              {orphans.map((n) => (
                <div key={n.node_id} className="border-l-2 border-primary pl-2 text-xs">
                  {n.label} — no parent left
                </div>
              ))}
            </div>
          )}
          <p className="text-xs text-muted-foreground">
            {stagedMetricCount} queryable metric(s) staged in total.
          </p>
          <Button size="sm" disabled={diff.length === 0 || merging} onClick={() => void mergeAll()}>
            {merging ? <Loader2 className="mr-1 size-4 animate-spin" /> : null}
            Merge All Staged Changes
          </Button>
        </div>
      </section>
    </div>
  );
}
