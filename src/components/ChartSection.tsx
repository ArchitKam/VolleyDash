import { useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type { ActionResult } from "@/lib/compute";
import { buildPlayerColors } from "@/lib/format";
import { describePipeline, type Axis, type Row } from "@/lib/pipeline";

const NONE = "(none)";
type Slot = "group" | "color" | "panel";

function distinct(frame: Row[], axis: Axis): string[] {
  return [...new Set(frame.map((r) => String(r[axis])))];
}

export function ChartSection({ result }: { result: ActionResult }) {
  const frame = useMemo(() => result.frame.filter((r) => r.Value !== null), [result.frame]);
  const varying = useMemo(
    () => (["Player", "Game", "Metric"] as Axis[]).filter((a) => distinct(frame, a).length > 1),
    [frame],
  );

  const [assign, setAssign] = useState<Record<Slot, string>>({
    group: NONE,
    color: NONE,
    panel: NONE,
  });
  const [order, setOrder] = useState<"value" | "original">("original");

  useEffect(() => {
    setAssign({
      group: varying.includes("Game") ? "Game" : (varying[0] ?? NONE),
      color: varying.includes("Player") ? "Player" : NONE,
      panel: varying.includes("Metric") ? "Metric" : NONE,
    });
  }, [varying.join("|")]); // eslint-disable-line react-hooks/exhaustive-deps

  const setSlot = (slot: Slot, value: string) => {
    setAssign((prev) => {
      const next = { ...prev };
      if (value !== NONE) {
        for (const s of ["group", "color", "panel"] as Slot[]) {
          if (s !== slot && next[s] === value) next[s] = prev[slot];
        }
      }
      next[slot] = value;
      return next;
    });
  };

  const playerColors = useMemo(
    () => buildPlayerColors(result.frame.map((r) => r.Player)),
    [result.frame],
  );

  if (frame.length === 0) return null;

  const groupAxis = assign.group === NONE ? null : (assign.group as Axis);
  const colorAxis = assign.color === NONE ? null : (assign.color as Axis);
  const panelAxis = assign.panel === NONE ? null : (assign.panel as Axis);

  const panels = panelAxis ? distinct(frame, panelAxis) : ["All"];
  const series = colorAxis ? distinct(frame, colorAxis) : ["Value"];

  const buildData = (panel: string) => {
    const rows = panelAxis ? frame.filter((r) => String(r[panelAxis]) === panel) : frame;
    const groups = groupAxis ? distinct(rows, groupAxis) : ["All"];
    const data = groups.map((g) => {
      const point: Record<string, string | number | null> = { name: g };
      const inGroup = groupAxis ? rows.filter((r) => String(r[groupAxis]) === g) : rows;
      for (const s of series) {
        const matching = colorAxis ? inGroup.filter((r) => String(r[colorAxis]) === s) : inGroup;
        const vals = matching.map((r) => r.Value!).filter((v) => v !== null);
        point[s] = vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
      }
      return point;
    });
    if (groupAxis && order === "value") {
      data.sort((a, b) => {
        const sum = (p: Record<string, string | number | null>) =>
          series.reduce((acc, s) => acc + (Number(p[s]) || 0), 0);
        return sum(b) - sum(a);
      });
    }
    return data;
  };

  const colorFor = (s: string, idx: number) => {
    if (colorAxis === "Player") return playerColors[s] ?? "#E21833";
    const palette = ["#E21833", "#B8860B", "#8B0000", "#DAA520", "#FFFFFF", "#A9A9A9"];
    return palette[idx % palette.length]!;
  };

  return (
    <div className="space-y-3 rounded-lg border border-border bg-card p-4">
      <div>
        <h4 className="font-semibold text-gold">{result.title}</h4>
        {result.action.pipeline.length > 0 && (
          <p className="text-xs text-muted-foreground">
            Pipeline: {describePipeline(result.action.pipeline)}
          </p>
        )}
        {result.warnings.map((w) => (
          <p key={w} className="text-xs text-primary">
            {w}
          </p>
        ))}
      </div>

      <div className="flex flex-wrap gap-2">
        {(
          [
            ["group", "Group by"],
            ["color", "Color by"],
            ["panel", "Split into panels by"],
          ] as Array<[Slot, string]>
        ).map(([slot, label]) => (
          <div key={slot} className="min-w-[9rem]">
            <div className="mb-1 text-xs text-muted-foreground">{label}</div>
            <Select value={assign[slot]} onValueChange={(v) => setSlot(slot, v)}>
              <SelectTrigger className="h-8">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value={NONE}>{NONE}</SelectItem>
                {varying.map((a) => (
                  <SelectItem key={a} value={a}>
                    {a}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        ))}
        {groupAxis && (
          <div>
            <div className="mb-1 text-xs text-muted-foreground">Order</div>
            <Button
              size="sm"
              variant="outline"
              onClick={() => setOrder((o) => (o === "value" ? "original" : "value"))}
            >
              {order === "value" ? "By value" : "Original order"}
            </Button>
          </div>
        )}
      </div>

      <div className={`grid gap-4 ${panels.length > 1 ? "md:grid-cols-2" : "grid-cols-1"}`}>
        {panels.map((panel) => (
          <div key={panel} className="rounded border border-border/60 p-2">
            <div className="mb-1 text-xs text-gold">{panel}</div>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={buildData(panel)}>
                <CartesianGrid stroke="#2a2a2a" vertical={false} />
                <XAxis dataKey="name" stroke="#A9A9A9" fontSize={11} />
                <YAxis stroke="#A9A9A9" fontSize={11} />
                <Tooltip
                  contentStyle={{
                    background: "#111",
                    border: "1px solid #333",
                    color: "#fff",
                  }}
                />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                {series.map((s, i) => (
                  <Bar key={s} dataKey={s} fill={colorFor(s, i)} />
                ))}
              </BarChart>
            </ResponsiveContainer>
          </div>
        ))}
      </div>
    </div>
  );
}
