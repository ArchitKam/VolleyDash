function norm(s: string): string {
  return s
    .toLowerCase()
    .replace(/[^a-z0-9 ]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function bigrams(s: string): string[] {
  const out: string[] = [];
  for (let i = 0; i < s.length - 1; i++) out.push(s.slice(i, i + 2));
  return out;
}

/** Dice coefficient string similarity in [0,1]. */
export function similarity(a: string, b: string): number {
  const x = norm(a);
  const y = norm(b);
  if (!x || !y) return 0;
  if (x === y) return 1;
  const ax = bigrams(x);
  const by = bigrams(y);
  if (ax.length === 0 || by.length === 0) return 0;
  const pool = [...by];
  let hits = 0;
  for (const g of ax) {
    const idx = pool.indexOf(g);
    if (idx >= 0) {
      hits++;
      pool.splice(idx, 1);
    }
  }
  return (2 * hits) / (ax.length + by.length);
}

/**
 * Exact -> substring / word-prefix -> fuzzy (>= 0.75) resolution.
 * Returns the matching candidate or null.
 */
export function resolveFuzzy(query: string, candidates: string[]): string | null {
  const q = norm(query);
  if (!q) return null;

  for (const c of candidates) if (norm(c) === q) return c;

  for (const c of candidates) {
    const n = norm(c);
    if (n.includes(q) || q.includes(n)) return c;
  }

  for (const c of candidates) {
    const words = norm(c).split(" ");
    const qWords = q.split(" ");
    if (
      qWords.some(
        (qw) => qw.length >= 2 && words.some((w) => w.startsWith(qw) || qw.startsWith(w)),
      )
    )
      return c;

  }

  let best: { c: string; s: number } | null = null;
  for (const c of candidates) {
    const s = similarity(q, c);
    if (!best || s > best.s) best = { c, s };
  }
  return best && best.s >= 0.75 ? best.c : null;
}
