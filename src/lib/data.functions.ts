import { createServerFn } from "@tanstack/react-start";

import { parseCsv, type CsvRow } from "./csv";

const GH_API = "https://api.github.com";
const TREE_PATH = "recruiting_kb_data.json";
const CSV_DIR = "recruiting_csvs";

type GhEnv = { repo: string; token: string };

function ghEnv(): GhEnv {
  const repo = process.env["GITHUB_DATA_REPO"];
  const token = process.env["GITHUB_DATA_TOKEN"];
  if (!repo || !token) {
    throw new Error(
      "Missing GITHUB_DATA_REPO / GITHUB_DATA_TOKEN. Add them in Project Settings → Secrets.",
    );
  }
  return { repo, token };
}

function ghHeaders(token: string): Record<string, string> {
  return {
    Authorization: `Bearer ${token}`,
    Accept: "application/vnd.github+json",
    "User-Agent": "volleydash",
  };
}

function decodeBase64(b64: string): string {
  const raw = atob(b64.replace(/\s/g, ""));
  const bytes = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
  return new TextDecoder().decode(bytes);
}

function encodeBase64(text: string): string {
  const bytes = new TextEncoder().encode(text);
  let binary = "";
  bytes.forEach((b) => {
    binary += String.fromCharCode(b);
  });
  return btoa(binary);
}

async function ghGet(path: string, env: GhEnv): Promise<Response> {
  return fetch(`${GH_API}/repos/${env.repo}/contents/${path}`, {
    headers: ghHeaders(env.token),
  });
}

/** Reads the committed knowledge tree. Returns null on ANY failure. */
export const getTree = createServerFn({ method: "GET" }).handler(
  async (): Promise<{ json: string } | null> => {
    try {
      const env = ghEnv();
      const res = await ghGet(TREE_PATH, env);
      if (!res.ok) return null;
      const payload = (await res.json()) as { content?: string };
      if (!payload.content) return null;
      const text = decodeBase64(payload.content);
      JSON.parse(text); // reject malformed JSON
      return { json: text };
    } catch {
      return null;
    }
  },
);

/** Two-step GET-sha then PUT. Loud failure at either step. */
export const saveTree = createServerFn({ method: "POST" })
  .inputValidator((input: { tree: unknown }) => input)
  .handler(async ({ data }) => {
    const env = ghEnv();
    let sha: string | undefined;
    const getRes = await ghGet(TREE_PATH, env);
    if (getRes.ok) {
      const payload = (await getRes.json()) as { sha?: string };
      sha = payload.sha;
    } else if (getRes.status !== 404) {
      throw new Error(
        `Could not read ${TREE_PATH} before saving (GitHub ${getRes.status}). Nothing was pushed.`,
      );
    }

    const body = {
      message: "Update recruiting_kb_data.json via dashboard",
      content: encodeBase64(JSON.stringify(data.tree, null, 2)),
      ...(sha ? { sha } : {}),
    };
    const putRes = await fetch(`${GH_API}/repos/${env.repo}/contents/${TREE_PATH}`, {
      method: "PUT",
      headers: { ...ghHeaders(env.token), "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!putRes.ok) {
      const text = await putRes.text();
      throw new Error(`GitHub rejected the push (${putRes.status}): ${text.slice(0, 300)}`);
    }
    return { ok: true as const };
  });

/* ---------------- player-analysis workspace (.dvw) ---------------- */

const VOLLEY_TREE_PATH = "volley_kb_data.json";
const DVW_DIR = "dvw";

/**
 * Reads the play-by-play knowledge tree. Its top-level key must be "volley";
 * a file carrying the recruiting shape is rejected so the caller seeds fresh
 * rather than silently mixing two grains.
 */
export const getVolleyTree = createServerFn({ method: "GET" }).handler(
  async (): Promise<{ json: string } | null> => {
    try {
      const env = ghEnv();
      const res = await ghGet(VOLLEY_TREE_PATH, env);
      if (!res.ok) return null;
      const payload = (await res.json()) as { content?: string };
      if (!payload.content) return null;
      const text = decodeBase64(payload.content);
      const parsed = JSON.parse(text) as Record<string, unknown>;
      if (!parsed || typeof parsed !== "object" || !("volley" in parsed)) return null;
      return { json: JSON.stringify(parsed["volley"]) };
    } catch {
      return null;
    }
  },
);

export const saveVolleyTree = createServerFn({ method: "POST" })
  .inputValidator((input: { tree: unknown }) => input)
  .handler(async ({ data }) => {
    const env = ghEnv();
    let sha: string | undefined;
    const getRes = await ghGet(VOLLEY_TREE_PATH, env);
    if (getRes.ok) {
      const payload = (await getRes.json()) as { sha?: string };
      sha = payload.sha;
    } else if (getRes.status !== 404) {
      throw new Error(
        `Could not read ${VOLLEY_TREE_PATH} before saving (GitHub ${getRes.status}). Nothing was pushed.`,
      );
    }
    const body = {
      message: "Update volley_kb_data.json via dashboard",
      content: encodeBase64(JSON.stringify({ volley: data.tree }, null, 2)),
      ...(sha ? { sha } : {}),
    };
    const putRes = await fetch(`${GH_API}/repos/${env.repo}/contents/${VOLLEY_TREE_PATH}`, {
      method: "PUT",
      headers: { ...ghHeaders(env.token), "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!putRes.ok) {
      const text = await putRes.text();
      throw new Error(`GitHub rejected the push (${putRes.status}): ${text.slice(0, 300)}`);
    }
    return { ok: true as const };
  });

export type DvwFileRef = { path: string; filename: string };

/** No filename parsing: opponent and date come out of the file's own sections. */
export const listDvwFiles = createServerFn({ method: "GET" }).handler(
  async (): Promise<DvwFileRef[]> => {
    const env = ghEnv();
    const res = await ghGet(DVW_DIR, env);
    if (res.status === 404) return [];
    if (!res.ok) throw new Error(`Could not list ${DVW_DIR}/ (GitHub ${res.status}).`);
    const items = (await res.json()) as Array<{ name: string; path: string; type: string }>;
    return items
      .filter((i) => i.type === "file" && /\.dvw$/i.test(i.name))
      .map((i) => ({ path: i.path, filename: i.name }))
      .sort((a, b) => a.filename.localeCompare(b.filename));
  },
);

/** Raw text — parsing happens on the client, like the CSV path. */
export const getDvwFile = createServerFn({ method: "POST" })
  .inputValidator((input: { path: string }) => input)
  .handler(async ({ data }): Promise<{ text: string }> => {
    const env = ghEnv();
    const res = await ghGet(data.path, env);
    if (!res.ok) throw new Error(`Could not read ${data.path} (GitHub ${res.status}).`);
    const payload = (await res.json()) as { content?: string };
    if (!payload.content) throw new Error(`${data.path} returned no content.`);
    return { text: decodeBase64(payload.content) };
  });

export type GameRef = { path: string; filename: string; opponent: string };

function cleanTeamName(part: string): string {
  return part
    .replace(/-\s*stats\s*$/i, "")
    .replace(/^[\s\-_]+|[\s\-_]+$/g, "")
    .replace(/[_]+/g, " ")
    .replace(/\s{2,}/g, " ")
    .trim();
}

const OWN_TEAM = /^nat(ionals?)?\b/i;

/** Keep only the team name(s) that are not our own team ("Nat"). */
function opponentFromFilename(filename: string): string {
  const base = filename.replace(/\.csv$/i, "");
  const parts = base
    .split(/\s*\bvs\.?\b\s*/i)
    .map(cleanTeamName)
    .filter(Boolean);
  const others = parts.filter((p) => !OWN_TEAM.test(p));
  return (others[0] ?? parts[parts.length - 1] ?? cleanTeamName(base)) || base.trim();
}

export const listGames = createServerFn({ method: "GET" }).handler(async (): Promise<GameRef[]> => {
  const env = ghEnv();
  const res = await ghGet(CSV_DIR, env);
  if (!res.ok) {
    throw new Error(`Could not list ${CSV_DIR}/ (GitHub ${res.status}).`);
  }
  const items = (await res.json()) as Array<{ name: string; path: string; type: string }>;
  return items
    .filter((i) => i.type === "file" && /\.csv$/i.test(i.name))
    .map((i) => ({ path: i.path, filename: i.name, opponent: opponentFromFilename(i.name) }))
    .sort((a, b) => a.filename.localeCompare(b.filename));
});

export const getGameCsv = createServerFn({ method: "POST" })
  .inputValidator((input: { path: string }) => input)
  .handler(async ({ data }): Promise<CsvRow[]> => {
    const env = ghEnv();
    const res = await ghGet(data.path, env);
    if (!res.ok) throw new Error(`Could not read ${data.path} (GitHub ${res.status}).`);
    const payload = (await res.json()) as { content?: string };
    if (!payload.content) throw new Error(`${data.path} returned no content.`);
    return parseCsv(decodeBase64(payload.content));
  });

function stripFences(text: string): string {
  let t = text.trim();
  if (t.startsWith("```")) {
    t = t.replace(/^```[a-zA-Z]*\s*/, "");
    if (t.endsWith("```")) t = t.slice(0, -3);
  }
  return t.trim();
}

export const callLlm = createServerFn({ method: "POST" })
  .inputValidator(
    (input: {
      systemPrompt: string;
      userMessage: string;
      maxTokens?: number;
      temperature?: number;
    }) => input,
  )
  .handler(async ({ data }): Promise<{ text: string; model: string }> => {
    const endpoint = process.env["AZURE_OPENAI_ENDPOINT"];
    const deployment = process.env["AZURE_OPENAI_DEPLOYMENT"];
    const key = process.env["AZURE_OPENAI_API_KEY"];
    if (!endpoint || !deployment || !key) {
      throw new Error(
        "LLM unreachable: AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_DEPLOYMENT / AZURE_OPENAI_API_KEY isn't configured. Add them in Project Settings → Secrets.",
      );
    }

    const url = `${endpoint.replace(/\/+$/, "")}/chat/completions`;
    const messages = [
      { role: "system", content: data.systemPrompt },
      { role: "user", content: data.userMessage },
    ];

    // Newer models reject max_tokens and/or temperature. Retry without the
    // offending parameter when the API reports it as unsupported.
    const body: Record<string, unknown> = {
      model: deployment,
      messages,
      max_completion_tokens: data.maxTokens ?? 2048,
      temperature: data.temperature ?? 0.1,
    };

    let res: Response;
    let text = "";
    for (let attempt = 0; attempt < 3; attempt++) {
      try {
        res = await fetch(url, {
          method: "POST",
          headers: { Authorization: `Bearer ${key}`, "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
      } catch (e) {
        throw new Error(`LLM unreachable: ${e instanceof Error ? e.message : String(e)}`);
      }

      if (res.ok) {
        const payload = (await res.json()) as {
          choices?: Array<{ message?: { content?: string } }>;
        };
        const content = payload.choices?.[0]?.message?.content ?? "";
        return { text: stripFences(content), model: deployment };
      }

      text = await res.text();
      let badParam: string | null = null;
      try {
        const err = JSON.parse(text) as { error?: { code?: string; param?: string } };
        if (err.error?.code === "unsupported_parameter" && err.error.param) {
          badParam = err.error.param;
        } else if (err.error?.code === "unsupported_value" && err.error.param) {
          badParam = err.error.param;
        }
      } catch {
        badParam = null;
      }
      if (badParam && badParam in body) {
        delete body[badParam];
        if (badParam === "max_tokens") body["max_completion_tokens"] = data.maxTokens ?? 2048;
        continue;
      }
      break;
    }

    throw new Error(`LLM unreachable: Azure OpenAI responded ${res!.status} — ${text.slice(0, 300)}`);
  });
