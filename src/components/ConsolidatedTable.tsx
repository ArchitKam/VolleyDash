import { useMemo, useState } from "react";
import { ChevronDown } from "lucide-react";

import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Button } from "@/components/ui/button";
import type { ActionResult } from "@/lib/compute";
import { formatKindFor, formatValue } from "@/lib/format";
import { findMetric, nodeDescription, nodeExpression, type GraphTree } from "@/lib/tree";

type Props = {
  results: ActionResult[];
  committed: GraphTree;
  unrecognized: string[];
};

type WideRow = { Player: string; Game: string; values: Record<string, number | null> };

export function ConsolidatedTable({ results, committed, unrecognized }: Props) {
  const [sortKey, setSortKey] = useState<string>("Player");
  const [asc, setAsc] = useState(true);

  const { columns, rows } = useMemo(() => {
    const cols: string[] = [];
    const map = new Map<string, WideRow>();
    for (const r of results) {
      for (const row of r.frame) {
        if (!cols.includes(row.Metric)) cols.push(row.Metric);
        const key = `${row.Player}||${row.Game}`;
        const existing = map.get(key) ?? { Player: row.Player, Game: row.Game, values: {} };
        existing.values[row.Metric] = row.Value;
        map.set(key, existing);
      }
    }
    return { columns: cols, rows: [...map.values()] };
  }, [results]);

  const sorted = useMemo(() => {
    const copy = [...rows];
    copy.sort((a, b) => {
      let av: string | number | null;
      let bv: string | number | null;
      if (sortKey === "Player" || sortKey === "Game") {
        av = a[sortKey];
        bv = b[sortKey];
      } else {
        av = a.values[sortKey] ?? null;
        bv = b.values[sortKey] ?? null;
      }
      if (av === null) return 1;
      if (bv === null) return -1;
      const cmp = typeof av === "string" ? av.localeCompare(String(bv)) : Number(av) - Number(bv);
      return asc ? cmp : -cmp;
    });
    return copy;
  }, [rows, sortKey, asc]);

  const toggleSort = (key: string) => {
    if (key === sortKey) setAsc((v) => !v);
    else {
      setSortKey(key);
      setAsc(true);
    }
  };

  if (rows.length === 0) {
    return (
      <div className="rounded-lg border border-border bg-card p-4 text-sm text-muted-foreground">
        {unrecognized.length > 0
          ? `I didn't recognize '${unrecognized.join("', '")}' as a stat or metric — did you mean something else?`
          : "No computable values returned for this combination."}
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <div className="overflow-x-auto rounded-lg border border-border bg-card">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border">
              {["Player", "Game", ...columns].map((c) => (
                <th
                  key={c}
                  onClick={() => toggleSort(c)}
                  className="cursor-pointer px-3 py-2 text-left font-semibold whitespace-nowrap text-gold hover:text-primary"
                >
                  {c}
                  {sortKey === c ? (asc ? " ▲" : " ▼") : ""}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sorted.map((r) => (
              <tr key={`${r.Player}-${r.Game}`} className="border-b border-border/50">
                <td className="px-3 py-2 whitespace-nowrap">{r.Player}</td>
                <td className="px-3 py-2 whitespace-nowrap">{r.Game}</td>
                {columns.map((c) => (
                  <td key={c} className="px-3 py-2 whitespace-nowrap tabular-nums">
                    {formatValue(
                      r.values[c] ?? null,
                      formatKindFor(c, findMetric(committed, c) ?? undefined),
                    )}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <Collapsible>
        <CollapsibleTrigger asChild>
          <Button variant="outline" size="sm">
            <ChevronDown className="mr-1 size-4" /> Table Provenance &amp; Metric Contracts
          </Button>
        </CollapsibleTrigger>
        <CollapsibleContent className="mt-2 space-y-2 rounded-lg border border-border bg-card p-3 text-sm">
          {columns.map((c) => {
            const leaf = findMetric(committed, c) ?? undefined;
            return (
              <div key={c} className="border-b border-border/40 pb-2 last:border-0">
                <div className="font-semibold text-gold">{c}</div>
                <div className="text-muted-foreground">
                  {leaf ? nodeDescription(leaf) : "Derived column"}
                </div>
                <code className="text-xs text-primary">{leaf ? nodeExpression(leaf) : c}</code>
                <div className="text-xs text-muted-foreground">
                  authored_by: {leaf?.authored_by ?? "unknown"}
                </div>
              </div>
            );
          })}
        </CollapsibleContent>
      </Collapsible>
    </div>
  );
}
