/**
 * DataVolley .dvw parser.
 *
 * The grammar below is reverse-engineered from real match files. Three fixed
 * behaviours are baked in deliberately:
 *  1. [3SET] fields are parsed INDEPENDENTLY, so a partially scored (15-point)
 *     deciding set can't corrupt the row width.
 *  2. Jersey number 0 is legal. "the digits didn't parse" is a separate null
 *     case and is never conflated with "the number zero".
 *  3. The per-set "played" flag is unreliable (reads True for unplayed sets);
 *     whether a set was played is inferred from its populated score pair.
 */

export type DvwSide = "home" | "visiting";

export type DvwPlayer = {
  side: DvwSide;
  team_id: string;
  jersey: number | null;
  player_id: string;
  lastname: string;
  firstname: string;
  nickname: string;
  role: string;
  /** set1..set5 participation markers; blank = did not play that set. */
  participation: string[];
  player_name: string;
  /** "#7 Sloan Miller" — jersey included, since one name can wear two numbers. */
  player_label: string;
};

export type DvwSetRow = {
  set_number: number;
  /** Partial scores at each break plus the final pair; [null,null] when blank. */
  scores: Array<[number | null, number | null]>;
  /** Field 5. This is the set's TARGET score (25, or 15 for a fifth set). */
  target_score: number | null;
  played: boolean;
};

export type DvwAction = {
  code: string;
  team_side: DvwSide;
  jersey: number | null;
  skill: string | null;
  evaluation_code: string | null;
  attack_code: string | null;
  set_code: string | null;
  set_type: string | null;
  start_zone: string | null;
  end_zone: string | null;
  end_subzone: string | null;
  num_players_numeric: string | null;
  custom_code: string | null;
  time: string;
  set_number: number | null;
  home_setter_position: string;
  visiting_setter_position: string;
  home_lineup: string[];
  visiting_lineup: string[];
  /* derived */
  rally_number: number | null;
  team: string;
  point_won_by: string | null;
  serving_team: string | null;
  receiving_team: string | null;
  point_phase: string | null;
  attack_phase: string | null;
  possession_number: number | null;
  home_team_score: number | null;
  visiting_team_score: number | null;
  player_name: string;
  player_label: string;
};

export type DvwMatch = {
  path: string;
  filename: string;
  date_raw: string;
  date: { year: number; month: number; day: number } | null;
  time: string;
  season: string;
  championship: string;
  home_team: string;
  visiting_team: string;
  home_team_id: string;
  visiting_team_id: string;
  home_sets_won: number;
  visiting_sets_won: number;
  players: DvwPlayer[];
  sets: DvwSetRow[];
  /** Skill-bearing actions only; score markers / subs / timeouts are dropped. */
  actions: DvwAction[];
  warnings: string[];
};

const SKILL_BY_LETTER: Record<string, string> = {
  S: "Serve",
  R: "Reception",
  E: "Set",
  A: "Attack",
  D: "Dig",
  B: "Block",
  F: "Freeball",
};

const EVALUATION_CODES = new Set(["#", "+", "!", "-", "/", "="]);

function sectionIndex(lines: string[], header: string): number {
  return lines.findIndex((l) => l.trim() === header);
}

function isSectionHeader(line: string): boolean {
  return line.trim().startsWith("[");
}

function intOrNull(text: string): number | null {
  const t = text.trim();
  if (!t) return null;
  const n = Number(t);
  return Number.isFinite(n) ? Math.trunc(n) : null;
}

function nz(value: string | undefined): string | null {
  const v = (value ?? "").trim();
  if (!v || v === "~" || v === "~~") return null;
  return v;
}

/** MM/DD/YYYY -> parts. */
export function parseDvwDate(raw: string): { year: number; month: number; day: number } | null {
  const m = /^(\d{1,2})\/(\d{1,2})\/(\d{2,4})$/.exec(raw.trim());
  if (!m) return null;
  const month = Number(m[1]);
  const day = Number(m[2]);
  let year = Number(m[3]);
  if (year < 100) year += 2000;
  if (!month || !day || !year) return null;
  return { year, month, day };
}

const MONTHS = [
  "Jan",
  "Feb",
  "Mar",
  "Apr",
  "May",
  "Jun",
  "Jul",
  "Aug",
  "Sep",
  "Oct",
  "Nov",
  "Dec",
];

/** "11/08/2025" -> "Nov 8" (no leading zero). Unparseable -> "". */
export function formatDvwDate(raw: string): string {
  const parts = parseDvwDate(raw);
  if (!parts) return "";
  const month = MONTHS[parts.month - 1];
  if (!month) return "";
  return `${month} ${parts.day}`;
}

const TEAM_NAME_OVERRIDES: Record<string, string> = {
  "university of california, los angeles": "UCLA",
  "university of california los angeles": "UCLA",
  "university of southern california": "USC",
  "university of illinois urbana-champaign": "Illinois",
  "university of wisconsin-madison": "Wisconsin",
  "pennsylvania state university": "Penn State",
};

/** Cosmetic only — never used as a key. */
export function shortenTeamName(raw: string): string {
  const trimmed = (raw ?? "").trim();
  const override = TEAM_NAME_OVERRIDES[trimmed.toLowerCase()];
  if (override) return override;
  let out = trimmed.replace(/,.*$/, "").trim();
  out = out.replace(/^University of\s+/i, "");
  out = out.replace(/\sState University$/i, " State");
  out = out.replace(/\sUniversity$/i, "");
  return out.trim() || trimmed;
}

/* ---------------- code decoding ---------------- */

export type DecodedCode = {
  team_side: DvwSide;
  is_marker: boolean;
  jersey: number | null;
  skill: string | null;
  evaluation_code: string | null;
  attack_code: string | null;
  set_code: string | null;
  set_type: string | null;
  start_zone: string | null;
  end_zone: string | null;
  end_subzone: string | null;
  num_players_numeric: string | null;
  custom_code: string | null;
};

export function decodeCode(code: string): DecodedCode {
  const side: DvwSide = code[0] === "a" ? "visiting" : "home";
  const lastTilde = code.lastIndexOf("~");
  const custom = lastTilde >= 0 ? nz(code.slice(lastTilde + 1)) : null;

  // code[1] === 'p' marks a rally-outcome/score row ("*p01:03") — no player, no skill letter.
  if (code[1] === "p") {
    return {
      team_side: side,
      is_marker: true,
      jersey: null,
      skill: "Point",
      evaluation_code: null,
      attack_code: null,
      set_code: null,
      set_type: null,
      start_zone: null,
      end_zone: null,
      end_subzone: null,
      num_players_numeric: null,
      custom_code: custom,
    };
  }

  // Jersey 0 is a real number. "not digits here" is a separate null case.
  const jerseyText = code.slice(1, 3);
  const jersey = /^\d\d$/.test(jerseyText) ? Number.parseInt(jerseyText, 10) : null;

  const skill = SKILL_BY_LETTER[code[3] ?? ""] ?? null;
  const evalChar = code[5] ?? "";
  const evaluation_code = EVALUATION_CODES.has(evalChar) ? evalChar : null;
  const pair = nz(code.slice(6, 8));

  return {
    team_side: side,
    is_marker: false,
    jersey,
    skill,
    evaluation_code,
    attack_code: skill === "Attack" ? pair : null,
    set_code: skill === "Set" ? pair : null,
    set_type: skill === "Set" ? nz(code[8]) : null,
    start_zone: nz(code[9]),
    end_zone: nz(code[10]),
    end_subzone: nz(code[11]),
    num_players_numeric: skill === "Attack" ? nz(code[13]) : null,
    custom_code: custom,
  };
}

/** "*p01:03" -> [1, 3]. */
function markerScore(code: string): [number | null, number | null] {
  const m = /p\s*(\d+)\s*:\s*(\d+)/.exec(code);
  if (!m) return [null, null];
  return [Number(m[1]), Number(m[2])];
}

/* ---------------- sections ---------------- */

function parsePlayers(lines: string[], header: string, side: DvwSide): DvwPlayer[] {
  const start = sectionIndex(lines, header);
  if (start < 0) return [];
  const out: DvwPlayer[] = [];
  for (let i = start + 1; i < lines.length; i++) {
    const line = lines[i] ?? "";
    if (isSectionHeader(line)) break;
    if (!line.trim()) continue;
    const f = line.split(";").slice(0, 13);
    const jerseyText = (f[1] ?? "").trim();
    const jersey = /^\d+$/.test(jerseyText) ? Number.parseInt(jerseyText, 10) : null;
    const lastname = (f[9] ?? "").trim();
    const firstname = (f[10] ?? "").trim();
    const player_name = `${firstname} ${lastname}`.trim();
    out.push({
      side,
      team_id: (f[0] ?? "").trim(),
      jersey,
      player_id: (f[8] ?? "").trim(),
      lastname,
      firstname,
      nickname: (f[11] ?? "").trim(),
      role: (f[12] ?? "").trim(),
      participation: [3, 4, 5, 6, 7].map((idx) => (f[idx] ?? "").trim()),
      player_name,
      player_label: jersey === null ? "" : `#${jersey} ${player_name}`.trim(),
    });
  }
  return out;
}

function parseSets(lines: string[]): DvwSetRow[] {
  const start = sectionIndex(lines, "[3SET]");
  if (start < 0) return [];
  const out: DvwSetRow[] = [];
  for (let i = start + 1; i < lines.length && out.length < 5; i++) {
    const line = lines[i] ?? "";
    if (isSectionHeader(line)) break; // don't assume 5 rows are always present
    if (!line.trim()) continue;
    const f = line.split(";");
    // Each field parsed independently — one malformed pair must not shift the rest.
    const scores: Array<[number | null, number | null]> = [];
    for (let k = 1; k <= 4; k++) {
      const raw = (f[k] ?? "").trim();
      const m = /^(\d+)-(\d+)$/.exec(raw);
      scores.push(m ? [Number(m[1]), Number(m[2])] : [null, null]);
    }
    out.push({
      set_number: out.length + 1,
      scores,
      target_score: intOrNull(f[5] ?? ""),
      // The flag in field 0 lies; a populated score pair is the real signal.
      played: scores.some(([h, v]) => h !== null && v !== null),
    });
  }
  return out;
}

/* ---------------- main entry ---------------- */

export function parseDvw(text: string, path: string, filename: string): DvwMatch {
  const warnings: string[] = [];
  const lines = text.split(/\r?\n/);

  const matchIdx = sectionIndex(lines, "[3MATCH]");
  const matchFields = (matchIdx >= 0 ? (lines[matchIdx + 1] ?? "") : "").split(";");
  const date_raw = (matchFields[0] ?? "").trim();
  if (matchIdx < 0) warnings.push(`${filename}: no [3MATCH] section found.`);

  const teamsIdx = sectionIndex(lines, "[3TEAMS]");
  const homeFields = (teamsIdx >= 0 ? (lines[teamsIdx + 1] ?? "") : "").split(";");
  const visitFields = (teamsIdx >= 0 ? (lines[teamsIdx + 2] ?? "") : "").split(";");
  const home_team = (homeFields[1] ?? "").trim();
  const visiting_team = (visitFields[1] ?? "").trim();
  if (!home_team || !visiting_team) warnings.push(`${filename}: team names are missing.`);

  const players = [
    ...parsePlayers(lines, "[3PLAYERS-H]", "home"),
    ...parsePlayers(lines, "[3PLAYERS-V]", "visiting"),
  ];
  const sets = parseSets(lines);

  const byJersey = new Map<string, DvwPlayer>();
  for (const p of players) {
    if (p.jersey !== null) byJersey.set(`${p.side}|${p.jersey}`, p);
  }

  /* ---- [3SCOUT] ---- */
  const scoutIdx = sectionIndex(lines, "[3SCOUT]");
  const raw: DvwAction[] = [];
  if (scoutIdx < 0) {
    warnings.push(`${filename}: no [3SCOUT] section found, so it has no actions.`);
  } else {
    for (let i = scoutIdx + 1; i < lines.length; i++) {
      const line = lines[i] ?? "";
      if (isSectionHeader(line)) break;
      if (!line.trim()) continue;
      const f = line.split(";");
      const code = (f[0] ?? "").trim();
      if (!code) continue;
      const d = decodeCode(code);
      const player = d.jersey === null ? undefined : byJersey.get(`${d.team_side}|${d.jersey}`);
      raw.push({
        code,
        team_side: d.team_side,
        jersey: d.jersey,
        skill: d.skill,
        evaluation_code: d.evaluation_code,
        attack_code: d.attack_code,
        set_code: d.set_code,
        set_type: d.set_type,
        start_zone: d.start_zone,
        end_zone: d.end_zone,
        end_subzone: d.end_subzone,
        num_players_numeric: d.num_players_numeric,
        custom_code: d.custom_code,
        time: (f[7] ?? "").trim(),
        set_number: intOrNull(f[8] ?? ""),
        home_setter_position: (f[9] ?? "").trim(),
        visiting_setter_position: (f[10] ?? "").trim(),
        home_lineup: [14, 15, 16, 17, 18, 19].map((k) => (f[k] ?? "").trim()),
        visiting_lineup: [20, 21, 22, 23, 24, 25].map((k) => (f[k] ?? "").trim()),
        rally_number: null,
        team: d.team_side === "home" ? home_team : visiting_team,
        point_won_by: null,
        serving_team: null,
        receiving_team: null,
        point_phase: null,
        attack_phase: null,
        possession_number: null,
        home_team_score: null,
        visiting_team_score: null,
        player_name: player?.player_name ?? "",
        player_label: d.is_marker ? "" : (player?.player_label ?? ""),
      });
    }
  }

  /* ---- derived fields ---- */

  // rally_number: within each set, count of Serve rows so far.
  const rallyCount = new Map<number, number>();
  for (const row of raw) {
    const set = row.set_number ?? 0;
    if (row.skill === "Serve") rallyCount.set(set, (rallyCount.get(set) ?? 0) + 1);
    row.rally_number = rallyCount.get(set) ?? 0;
  }

  // point_won_by: lookahead to the score marker that ends the rally, backfilled per set.
  {
    const pending = new Map<number, string | null>();
    for (let i = raw.length - 1; i >= 0; i--) {
      const row = raw[i]!;
      const set = row.set_number ?? 0;
      if (/^[*a]p/.test(row.code)) {
        const winner = row.code[0] === "a" ? visiting_team : home_team;
        pending.set(set, winner);
        row.point_won_by = winner;
        continue;
      }
      row.point_won_by = pending.get(set) ?? null;
    }
  }

  // serving_team / receiving_team forward-filled within (set, rally); scores backfilled likewise.
  {
    const serving = new Map<string, string>();
    for (const row of raw) {
      const key = `${row.set_number ?? 0}|${row.rally_number ?? 0}`;
      if (row.skill === "Serve") {
        serving.set(key, row.team_side === "home" ? home_team : visiting_team);
      }
      const s = serving.get(key) ?? null;
      row.serving_team = s;
      row.receiving_team = s === null ? null : s === home_team ? visiting_team : home_team;
      row.point_phase = s === null ? null : row.team === s ? "Serve" : "Reception";
    }

    const scoreByKey = new Map<string, [number | null, number | null]>();
    for (let i = raw.length - 1; i >= 0; i--) {
      const row = raw[i]!;
      const key = `${row.set_number ?? 0}|${row.rally_number ?? 0}`;
      if (/^[*a]p/.test(row.code)) scoreByKey.set(key, markerScore(row.code));
      const pair = scoreByKey.get(key);
      row.home_team_score = pair?.[0] ?? null;
      row.visiting_team_score = pair?.[1] ?? null;
    }
  }

  // attack_phase (straightforward version) + possession_number.
  {
    const possession = new Map<string, number>();
    for (let i = 0; i < raw.length; i++) {
      const row = raw[i]!;
      const key = `${row.set_number ?? 0}|${row.rally_number ?? 0}`;
      if (row.skill === "Attack") {
        possession.set(key, (possession.get(key) ?? 0) + 1);
        row.possession_number = possession.get(key) ?? 1;

        const sameTeamBefore: DvwAction[] = [];
        for (let k = i - 1; k >= 0 && sameTeamBefore.length < 2; k--) {
          const prev = raw[k]!;
          if (prev.set_number !== row.set_number || prev.rally_number !== row.rally_number) break;
          if (prev.team_side !== row.team_side) continue;
          if (!prev.skill || prev.skill === "Point") continue;
          sameTeamBefore.push(prev);
        }
        const afterReception =
          sameTeamBefore[0]?.skill === "Set" && sameTeamBefore[1]?.skill === "Reception";
        if (afterReception) row.attack_phase = "Reception";
        else if (row.serving_team && row.serving_team !== row.team)
          row.attack_phase = "SO-Transition";
        else row.attack_phase = "BP-Transition";
      } else {
        row.possession_number = (possession.get(key) ?? 0) + 1;
      }
    }
  }

  // Drop every row with no resolvable skill (markers, subs, timeouts, rotations) — keeping
  // them would create a phantom no-name player, and it also drops the trailing extra set.
  const actions = raw.filter(
    (r) => !!r.skill && r.skill !== "Point" && !!r.player_label && r.set_number !== null,
  );
  if (scoutIdx >= 0 && actions.length === 0) {
    warnings.push(`${filename}: no player actions could be decoded from [3SCOUT].`);
  }

  return {
    path,
    filename,
    date_raw,
    date: parseDvwDate(date_raw),
    time: (matchFields[1] ?? "").trim(),
    season: (matchFields[2] ?? "").trim(),
    championship: (matchFields[3] ?? "").trim(),
    home_team,
    visiting_team,
    home_team_id: (homeFields[0] ?? "").trim(),
    visiting_team_id: (visitFields[0] ?? "").trim(),
    home_sets_won: intOrNull(homeFields[2] ?? "") ?? 0,
    visiting_sets_won: intOrNull(visitFields[2] ?? "") ?? 0,
    players,
    sets,
    actions,
    warnings,
  };
}

export function setLabel(setNumber: number): string {
  return `Set ${setNumber}`;
}

/** Game label: opponent plus date, because one opponent can appear twice a season. */
export function gameLabelFor(match: DvwMatch, teamOfInterest: string | null): string {
  const opponent =
    teamOfInterest && match.home_team === teamOfInterest
      ? match.visiting_team
      : teamOfInterest && match.visiting_team === teamOfInterest
        ? match.home_team
        : match.visiting_team;
  const date = formatDvwDate(match.date_raw);
  return date ? `${shortenTeamName(opponent)} (${date})` : shortenTeamName(opponent);
}

/** Chronological, from the parsed date — never by label text or filename. */
export function compareMatchesByDate(a: DvwMatch, b: DvwMatch): number {
  const av = a.date ? a.date.year * 10000 + a.date.month * 100 + a.date.day : Number.MAX_SAFE_INTEGER;
  const bv = b.date ? b.date.year * 10000 + b.date.month * 100 + b.date.day : Number.MAX_SAFE_INTEGER;
  if (av !== bv) return av - bv;
  return a.filename.localeCompare(b.filename);
}

export function teamCounts(matches: DvwMatch[]): Array<{ team: string; count: number }> {
  const counts = new Map<string, number>();
  for (const m of matches) {
    for (const t of [m.home_team, m.visiting_team]) {
      if (!t) continue;
      counts.set(t, (counts.get(t) ?? 0) + 1);
    }
  }
  return [...counts.entries()]
    .map(([team, count]) => ({ team, count }))
    .sort((a, b) => b.count - a.count || a.team.localeCompare(b.team));
}
