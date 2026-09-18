import { resolveFuzzy } from "./fuzzy";

/** Exact (case-insensitive) match first; fuzzy matcher only as a backstop. */
export function resolveCandidate(text: string, candidates: string[]): string | null {
  const t = text.trim().toLowerCase();
  const exact = candidates.find((c) => c.trim().toLowerCase() === t);
  if (exact) return exact;
  return resolveFuzzy(text, candidates);
}

/**
 * Player labels in the play-by-play workspace carry the jersey number ("#7 Sloan
 * Miller"), so "#7" or a bare "7" is a legitimate way for a coach to name a player.
 */
export function resolvePlayerHint(text: string, candidates: string[]): string | null {
  const raw = text.trim();
  const jersey = /^#?(\d{1,2})$/.exec(raw);
  if (jersey) {
    const hit = candidates.find((c) => c.startsWith(`#${Number(jersey[1])} `));
    if (hit) return hit;
  }
  return resolveCandidate(raw, candidates);
}

/**
 * Game hints can legitimately mean SEVERAL matches — the same opponent twice in a
 * season, with different dates — so every plausible match is returned.
 */
export function resolveGameHints(text: string, candidates: string[]): string[] {
  const t = text.trim().toLowerCase();
  if (!t) return [];
  const exact = candidates.filter((c) => c.trim().toLowerCase() === t);
  if (exact.length > 0) return exact;
  // "Illinois" should hit both "Illinois (Oct 3)" and "Illinois (Nov 8)".
  const prefix = candidates.filter((c) => c.toLowerCase().startsWith(t));
  if (prefix.length > 0) return prefix;
  const contains = candidates.filter((c) => c.toLowerCase().includes(t));
  if (contains.length > 0) return contains;
  const fuzzy = resolveCandidate(text, candidates);
  return fuzzy ? [fuzzy] : [];
}

/** "set 3", "third set", "Set 3" -> the matching set label. */
export function resolveSetHint(text: string, candidates: string[]): string[] {
  const t = text.trim().toLowerCase();
  if (!t) return [];
  const words: Record<string, number> = {
    first: 1,
    second: 2,
    third: 3,
    fourth: 4,
    fifth: 5,
    deciding: 5,
  };
  const num = /(\d)/.exec(t)?.[1];
  const n = num ? Number(num) : Object.entries(words).find(([w]) => t.includes(w))?.[1];
  if (n) {
    const hit = candidates.find((c) => c.toLowerCase() === `set ${n}`);
    if (hit) return [hit];
  }
  const exact = candidates.find((c) => c.toLowerCase() === t);
  return exact ? [exact] : [];
}
