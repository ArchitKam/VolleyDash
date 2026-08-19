#omsairam omsairam omsairam 
"""
recruiting_llm.py
==================
All NLP/LLM logic for the recruiting KB: the shared Groq client, the
LLM-backed metric author (turns a plain-language metric description
into a formula), the LLM-backed query router (turns a coach's question
into one or more resolvable actions), and a rule-based fallback parser
used when the LLM is unreachable.

Uses Groq's free-tier, OpenAI-compatible chat completions API -- no
GPU, no self-hosted server, and no cost for this workload's volume.
Deliberately kept Streamlit-agnostic (reads its API key straight from
os.environ) so this module stays directly importable/testable without
a Streamlit runtime, exactly as test_integration.py already does;
app.py is responsible for bridging st.secrets["GROQ_API_KEY"] into the
environment before anything here is called.

Trust boundary: the LLM only ever proposes a CANDIDATE expression or a
CANDIDATE query decomposition. Every proposal still runs through the
real MetricSpec.validate() (recruiting_tree.py) or the deterministic
Python repair step below before it's trusted -- the LLM never gets to
invent a column, metric, or category that doesn't really exist.
"""

import difflib
import json
import os
import re
from typing import Any, Dict, List, Optional

from openai import OpenAI
from openai import APIError, NotFoundError, RateLimitError

from recruiting_tree import ALL_SKILL_GROUPS, KnowledgeTree, NodeKind, RECRUITING_COLUMN_SCHEMA
from recruiting_operations import VALID_AGGS, VALID_COMPARISONS, pipeline_from_dicts, pipeline_to_dicts

LLM_BASE_URL = "https://api.groq.com/openai/v1"
LLM_MODEL = "openai/gpt-oss-20b"
# Tried in order if LLM_MODEL itself is rate-limited or unavailable (404,
# e.g. deprecated) -- Groq tracks rate limits PER MODEL, so a model that's
# hit its own daily/per-minute cap doesn't mean every model has. Same
# family/size tier as LLM_MODEL, so JSON-output behavior should be similar.
LLM_FALLBACK_MODELS = ["openai/gpt-oss-120b"]


class LLMUnavailableError(Exception):
    """Raised when Groq can't be reached at all (connection refused/
    timeout/no API key) -- distinct from a reachable-but-badly-formatted
    response, so callers can fall back (metric authoring) or show a
    specific "LLM unreachable" message (Q&A) rather than treating every
    failure the same way."""


def call_llm(system_prompt: str, user_message: str, max_tokens: int = 1536,
             temperature: float = 0.1) -> str:
    """
    One blocking, non-streaming chat completion against Groq's API,
    returning the raw response text with markdown code-fences stripped
    (models routinely wrap JSON in ```json ... ``` even when asked not
    to). Parsing that text into JSON is left to each caller, since the
    two features want slightly different schemas.

    Tries LLM_MODEL first, falling back to each of LLM_FALLBACK_MODELS in
    order ONLY on RateLimitError/NotFoundError (this specific model is
    rate-limited or unavailable -- a different model has its own,
    separate quota and may well still work). Any other APIError
    (connection/timeout, auth, a transient 5xx) fails immediately without
    trying the fallbacks, since those affect Groq/the account as a whole,
    not just one model -- retrying a different model wouldn't help.

    Raises LLMUnavailableError on a missing API key, or once every model
    in the chain has been tried and failed. Callers already treat
    LLMUnavailableError uniformly (fall back to the rule-based parser, or
    show a clear "unreachable" message) -- a Groq-side hiccup should
    degrade the SAME way regardless of which specific failure caused it,
    not crash the app with an uncaught SDK exception. Only a genuinely
    unexpected failure (e.g. a malformed response object once a call
    actually succeeded) propagates as-is.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise LLMUnavailableError(
            "GROQ_API_KEY isn't set -- get a free key at https://console.groq.com/keys "
            "and set it as an environment variable (or a Streamlit secret)."
        )

    client = OpenAI(base_url=LLM_BASE_URL, api_key=api_key)
    models_to_try = [LLM_MODEL] + [m for m in LLM_FALLBACK_MODELS if m != LLM_MODEL]

    response = None
    last_error: Optional[Exception] = None
    for model in models_to_try:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            break
        except (RateLimitError, NotFoundError) as e:
            last_error = e
            continue
        except APIError as e:
            raise LLMUnavailableError(
                f"Groq request failed ({type(e).__name__}): {e}"
            ) from e

    if response is None:
        raise LLMUnavailableError(
            f"Groq request failed on every configured model ({', '.join(models_to_try)}): {last_error}"
        )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
    if raw.endswith("```"):
        raw = raw[:-3]
    if raw.startswith("json"):
        raw = raw[4:]
    return raw.strip()


# ──────────────────────────────────────────────────────────────
# RULE-BASED FALLBACK PARSER -- used when the LLM is unreachable.
# Matching strategy: AND-of-OR keyword groups over a normalized phrase,
# NOT plain substring matching. Substring matching overmatches -- e.g. a
# bare "kills" substring check would also fire on "how many kills did
# she have, nothing about errors or rate", silently producing a
# plausible-looking but wrong formula. Requiring every keyword GROUP in
# a rule to have at least one hit makes accidental cross-rule matches
# much less likely, while still being entirely hardcoded/rule-based (no
# ML, no fuzzy scoring). Every rule's formula is hand-verified against
# RECRUITING_COLUMN_SCHEMA at authoring time, so a matched rule can
# never reference a nonexistent column.
# ──────────────────────────────────────────────────────────────

from dataclasses import dataclass, field
from typing import Callable, Set

_FILLER_WORDS = {"her", "his", "their", "the", "is", "whats", "what", "a", "an", "of"}


def _normalize(phrase: str) -> str:
    lowered = phrase.lower()
    stripped = re.sub(r"[^a-z0-9\s]", " ", lowered)
    words = [w for w in stripped.split() if w not in _FILLER_WORDS]
    return " ".join(words)


@dataclass
class ParseResult:
    matched: bool
    spec_kind: Optional[str] = None          # "formula" or "column"
    formula_expr: Optional[str] = None
    column_ref: Optional[str] = None
    human_description: Optional[str] = None
    suggested_label: Optional[str] = None
    suggested_branch_group: Optional[str] = None   # one of ALL_SKILL_GROUPS
    matched_rule_name: Optional[str] = None
    message: Optional[str] = None            # shown to the user either way
    # Populated by the LLM-backed author (parse_phrase_to_formula_llm) so
    # a freshly-staged metric is actually findable by the Q&A router's
    # alias-substring retrieval later; the rule-based rules below don't set
    # this (defaults to []) since their 6 phrasings are already the aliases.
    suggested_aliases: List[str] = field(default_factory=list)


@dataclass
class _Rule:
    name: str
    required_keyword_groups: List[Set[str]]   # AND across groups, OR within a group
    example_phrase: str
    build: Callable[[], ParseResult] = field(repr=False)

    def matches(self, normalized_phrase: str) -> bool:
        words = set(normalized_phrase.split())
        return all(any(kw in words or kw in normalized_phrase for kw in group)
                    for group in self.required_keyword_groups)


def _rule_net_kills_per_set() -> ParseResult:
    return ParseResult(
        matched=True, spec_kind="formula",
        formula_expr="([Attack K] - [Attack E]) / [Sets Sets Played]",
        human_description="Net kills (kills minus errors) per set played.",
        suggested_label="Net Kills Per Set", suggested_branch_group="Attack",
        matched_rule_name="net_kills_per_set",
    )


def _rule_raw_net_kills() -> ParseResult:
    return ParseResult(
        matched=True, spec_kind="formula",
        formula_expr="[Attack K] - [Attack E]",
        human_description="Kills minus attack errors (raw net kills, not rate-adjusted).",
        suggested_label="Net Kills (Raw)", suggested_branch_group="Attack",
        matched_rule_name="raw_net_kills",
    )


def _rule_aces_per_set() -> ParseResult:
    return ParseResult(
        matched=True, spec_kind="formula",
        formula_expr="[Serve SA] / [Sets Sets Played]",
        human_description="Service aces per set played.",
        suggested_label="Aces Per Set", suggested_branch_group="Serve",
        matched_rule_name="aces_per_set",
    )


def _rule_hitting_efficiency() -> ParseResult:
    return ParseResult(
        matched=True, spec_kind="column", column_ref="Attack Atk%",
        human_description="Existing hitting-efficiency column, surfaced directly (no new formula needed).",
        suggested_label="Hitting Efficiency", suggested_branch_group="Attack",
        matched_rule_name="hitting_efficiency",
    )


def _rule_blocks_per_set() -> ParseResult:
    return ParseResult(
        matched=True, spec_kind="formula",
        formula_expr="([Block BS] + [Block BA]) / [Sets Sets Played]",
        human_description="Combined solo+assisted blocks per set.",
        suggested_label="Blocks Per Set (Combined)", suggested_branch_group="Block",
        matched_rule_name="blocks_per_set",
    )


def _rule_dig_efficiency() -> ParseResult:
    return ParseResult(
        matched=True, spec_kind="formula",
        formula_expr="[Dig DS] / ([Dig DS] + [Dig DE])",
        human_description="Successful digs as a share of total dig attempts.",
        suggested_label="Dig Success Rate", suggested_branch_group="Dig",
        matched_rule_name="dig_efficiency",
    )


# Order matters: first full match wins. More specific rules (e.g. "per set"
# rate variants) are listed before their plainer counterparts so a phrase
# like "net kills per set" doesn't get short-circuited by a broader rule.
#
# Deliberately NOT included: a "passing minus errors" rule. The schema has
# no single "passing count" column (only quality buckets Receive 3/2/1/0
# and the already-a-ratio Receive Pass%), so there's no unit-consistent
# literal mapping available -- rather than guess at a mismatched formula,
# a phrase like "her passing minus her errors" is left to fall through to
# the no-match path below, same as any other unrecognized phrase.
_RULES: List[_Rule] = [
    _Rule("net_kills_per_set",
          [{"net"}, {"kill", "kills"}, {"per set", "per-set", "rate"}],
          "net kills per set", _rule_net_kills_per_set),
    _Rule("raw_net_kills",
          [{"kill", "kills"}, {"minus", "less"}, {"error", "errors"}],
          "kills minus errors", _rule_raw_net_kills),
    _Rule("aces_per_set",
          [{"ace", "aces"}, {"per set", "per-set", "rate"}],
          "aces per set", _rule_aces_per_set),
    _Rule("hitting_efficiency",
          [{"hitting", "attack", "attacking"}, {"efficiency", "atk", "hit percentage", "percentage"}],
          "hitting efficiency", _rule_hitting_efficiency),
    _Rule("blocks_per_set",
          [{"block", "blocks"}, {"per set", "per-set", "rate"}],
          "blocks per set", _rule_blocks_per_set),
    _Rule("dig_efficiency",
          [{"dig", "digs"}, {"efficiency", "rate", "percentage"}],
          "dig efficiency", _rule_dig_efficiency),
]

EXAMPLE_PHRASES: List[str] = [r.example_phrase for r in _RULES]


def parse_phrase_to_formula(phrase: str) -> ParseResult:
    """
    Matches a plain-language phrase against a small fixed set of hardcoded
    rules. First full match wins. No match -> ParseResult(matched=False,
    message=...) -- callers MUST show EXAMPLE_PHRASES as guidance in that
    case rather than guessing or partially applying a rule.
    """
    if not phrase or not phrase.strip():
        return ParseResult(matched=False, message="Type a description first.")

    normalized = _normalize(phrase)
    for rule in _RULES:
        if rule.matches(normalized):
            return rule.build()

    return ParseResult(
        matched=False,
        message=(
            f"Couldn't match \"{phrase}\" to a known phrasing. "
            "This is a rule-based stub (no LLM yet), so it only understands a "
            "small fixed set of phrasings -- try one of the examples below, or "
            "switch to editing a formula directly."
        ),
    )


# ──────────────────────────────────────────────────────────────
# LLM-BACKED METRIC AUTHOR -- replaces parse_phrase_to_formula above with
# a real LLM call, same ParseResult contract so app.py's wizard barely
# has to change between the two. Not a trusted source of truth: the
# LLM's proposed expression still runs through app.py's
# build_spec_from_expr()/MetricSpec.validate() path exactly like a
# rule-based or hand-typed formula would.
# ──────────────────────────────────────────────────────────────

_SINGLE_COL_RE = re.compile(r"^\[([^\[\]]+)\]$")


def _collect_existing_metric_labels(tree: KnowledgeTree) -> List[str]:
    return sorted(n.label for n in tree.committed.values() if n.kind == NodeKind.LEAF)


def _build_metric_author_system_prompt(tree: KnowledgeTree) -> str:
    columns_desc = "\n".join(
        f"- [{name}] ({spec.skill_group}, {spec.type.value}): {spec.description}"
        for name, spec in sorted(RECRUITING_COLUMN_SCHEMA.items())
    )
    existing_metrics = _collect_existing_metric_labels(tree)
    existing_desc = "\n".join(f"- [{label}]" for label in existing_metrics) or "(none yet)"
    groups = ", ".join(ALL_SKILL_GROUPS)

    return f"""You are the metric-authoring assistant for a volleyball recruiting knowledge base.
CRITICAL GROUNDING RULES:
1. ONLY route to a metric or category if the user's prompt explicitly names or closely describes it.
2. DO NOT guess or infer intent from unrecognized words, random characters, or nonsensical terms. 
3. If a word in the query cannot be confidently matched to a volleyball stat or player (e.g., gibberish or unrelated slang), DO NOT invent a typo correction. Return an empty "actions": [] array and explain in "reasoning" that the term was unrecognized.
4. Never assume a random word is a typo for a metric unless it has an obvious 1-2 character spelling edit.

A coach will describe a new statistic in plain language. Turn that description into an expression built ONLY from the bracketed names below -- either a single bracketed name (a direct reference to one existing column or metric) or an arithmetic expression combining bracketed names with +, -, *, /, and parentheses.

Never invent a bracketed name that isn't in one of the two lists below. If the request can't be mapped onto real columns/metrics, set "matched": false instead of guessing at something unit-mismatched.

RAW COLUMNS (bracket exactly as shown):
{columns_desc}

EXISTING METRICS already committed to the knowledge base (formulas may build on top of these too, e.g. "[Kills Per Set] * 2"):
{existing_desc}

Skill groups a new metric can be filed under: {groups}

Respond with ONLY this JSON shape, no other text:
{{
  "matched": true,
  "expr": "([Attack K] - [Attack E]) / [Sets Sets Played]",
  "human_description": "Net kills (kills minus errors) per set played.",
  "suggested_label": "Net Kills Per Set",
  "suggested_branch_group": "Attack",
  "suggested_aliases": ["net kills per set", "adjusted kill rate"]
}}

If the request is really just one existing column or metric with no arithmetic, "expr" should be that single bracketed name, e.g. "expr": "[Attack Atk%]".

If you truly cannot map the request onto real columns/metrics, respond with exactly:
{{"matched": false, "message": "<brief reason>"}}
"""


def parse_phrase_to_formula_llm(phrase: str, tree: KnowledgeTree) -> ParseResult:
    if not phrase or not phrase.strip():
        return ParseResult(matched=False, message="Type a description first.")

    raw = call_llm(_build_metric_author_system_prompt(tree), phrase)  # raises LLMUnavailableError on connection failure

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return ParseResult(
            matched=False,
            message=f"LLM returned something that wasn't valid JSON: {raw[:200]!r}",
        )

    if not isinstance(data, dict) or not data.get("matched"):
        message = data.get("message") if isinstance(data, dict) else None
        return ParseResult(matched=False, message=message or "The LLM couldn't map this to a real column or metric.")

    expr = data.get("expr")
    if not isinstance(expr, str) or not expr.strip():
        return ParseResult(matched=False, message="The LLM's response was missing a usable expression.")

    single_col_match = _SINGLE_COL_RE.match(expr.strip())
    suggested_aliases = [a for a in (data.get("suggested_aliases") or []) if isinstance(a, str)]

    return ParseResult(
        matched=True,
        spec_kind="column" if single_col_match else "formula",
        column_ref=single_col_match.group(1) if single_col_match else None,
        formula_expr=None if single_col_match else expr.strip(),
        human_description=(data.get("human_description") or phrase.strip()),
        suggested_label=(data.get("suggested_label") or phrase.strip().title()),
        suggested_branch_group=data.get("suggested_branch_group"),
        matched_rule_name="llm",
        suggested_aliases=suggested_aliases,
    )


# ──────────────────────────────────────────────────────────────
# LLM-BACKED QUERY ROUTER -- query-decomposition for the "Ask a
# Question" tab. Two-stage shape: (1) cheap non-embedding alias
# retrieval narrows what the prompt shows, (2) one LLM call produces
# JSON, (3) deterministic Python-side repair. Supports full multi-action
# decomposition: one question can name several (metric, player, game)
# triples at once, each becoming its own action.
#
# Nothing here resolves a player or game to a real record -- player/
# game_hint come back as whatever raw text the LLM extracted.
# Resolution against the real roster/game files happens downstream in
# app.py (resolve_player_name / resolve_game_hint) -- "the LLM
# extracts, Python resolves".
# ──────────────────────────────────────────────────────────────

_TOP_K_CANDIDATES = 8

# Small hardcoded synonym list per skill-group branch, same "cheap, hand-
# verified, no ML" philosophy as the rule-based parser above -- lets
# "serving"/"passing" resolve to the real branch labels ("Serve"/
# "Receive") even though those words never appear as a branch label
# substring themselves.
_SKILL_GROUP_SYNONYMS: Dict[str, List[str]] = {
    "Attack": ["attack", "attacking", "hitting", "hit"],
    "Serve": ["serve", "serving", "serves"],
    "Receive": ["receive", "receiving", "passing", "pass"],
    "Set": ["set", "setting"],
    "Dig": ["dig", "digging", "digs"],
    "Block": ["block", "blocking", "blocks"],
    "Points": ["point", "points", "scoring"],
    "Sets": ["sets played"],
}


def _resolve_skill_group(hint: Optional[str], valid_branches: List[str]) -> Optional[str]:
    """Tiered match of a raw category guess (e.g. 'serving') against the
    tree's REAL current branch labels: exact (case-insensitive) -> hardcoded
    synonym list -> substring -> difflib ratio >= 0.75. Returns None rather
    than guessing on ambiguity, same as resolve_game_hint/resolve_player_name
    (recruiting_executor logic, merged into app.py)."""
    if not hint or not hint.strip():
        return None
    needle = hint.strip().lower()

    exact = [b for b in valid_branches if b.lower() == needle]
    if exact:
        return exact[0]

    synonym_hits = [b for b in valid_branches if needle in _SKILL_GROUP_SYNONYMS.get(b, [])]
    if synonym_hits:
        return synonym_hits[0]

    substring = [b for b in valid_branches if needle in b.lower() or b.lower() in needle]
    if substring:
        return substring[0]

    fuzzy = [b for b in valid_branches if difflib.SequenceMatcher(None, needle, b.lower()).ratio() >= 0.75]
    return fuzzy[0] if fuzzy else None


def _retrieve_candidate_metrics(query: str, tree: KnowledgeTree) -> Dict[str, Any]:
    """
    Stage 1 -- pure Python, no LLM. For every committed LEAF, treats its
    own label as an implicit alias on top of its real `aliases` list (so a
    metric with no aliases yet can still be found via its label words),
    substring-matches each against the lowercased query, and ranks by
    number of hits.

    Falls back to EVERY committed leaf label if nothing matches at all
    (fallback=True) -- preserve recall for unanticipated phrasing rather
    than showing the model an empty candidate list.
    """
    q = query.lower()
    all_leaves = {n.label: [n.label.lower()] + [a.lower() for a in n.aliases]
                  for n in tree.committed.values() if n.kind == NodeKind.LEAF}

    hits: Dict[str, List[str]] = {}
    for label, alias_variants in all_leaves.items():
        matched = [a for a in alias_variants if a in q]
        if matched:
            hits[label] = matched

    if not hits:
        return {"candidates": sorted(all_leaves), "matched_aliases": {}, "fallback": True}

    ranked = sorted(hits.items(), key=lambda kv: -len(kv[1]))
    candidates = [label for label, _ in ranked[:_TOP_K_CANDIDATES]]
    return {"candidates": candidates, "matched_aliases": hits, "fallback": False}


def _build_router_system_prompt(candidates: List[str], tree: KnowledgeTree, known_games: List[str]) -> str:
    descriptions = []
    for label in candidates:
        node = next((n for n in tree.committed.values()
                     if n.kind == NodeKind.LEAF and n.label == label), None)
        desc = node.spec.description if (node and node.spec) else ""
        descriptions.append(f'- "{label}": {desc}')
    metrics_block = "\n".join(descriptions) if descriptions else "(no metrics available)"
    games_block = ", ".join(known_games) if known_games else "(no games loaded)"

    branch_lines = []
    for node in tree.committed.values():
        if node.kind == NodeKind.BRANCH and node.node_id != tree.root_id:
            member_labels = [tree.committed[c].label for c in node.children if c in tree.committed]
            branch_lines.append(f'- "{node.label}": {", ".join(member_labels) or "(no metrics yet)"}')
    branches_block = "\n".join(sorted(branch_lines)) if branch_lines else "(no categories available)"

    return f"""You are the query-routing engine for a volleyball recruiting analytics tool.

A coach will ask a question about player statistics. Split it into one action per distinct ask -- a compound question like "compare Sloan's kills per set and blocks per set in the Vegas Aces game" becomes TWO actions (same player/game, different metric); "Sloan's and Azana's kills per set" becomes TWO actions (same metric, different player).

Each action is EITHER about one specific metric OR about a whole skill category:
- If the coach names one specific stat, set "metric_of_interest" to exactly one of the metric names below (verbatim, never invented or paraphrased) and leave "skill_group" null.
- If the coach asks about a whole skill category in general WITHOUT naming one specific stat (e.g. "how was Sloan's serving", "how did she do on defense"), set "metric_of_interest" to null and "skill_group" to exactly one of the category names below (verbatim) -- every metric in that category will be returned together.
- Never set both metric_of_interest and skill_group non-null in the same action.
- If the coach describes a stat that sounds plausible but isn't in the metric list below, still set "metric_of_interest" to your best paraphrase of what they asked for (e.g. "Net Kills Per Set") -- this lets the coach author it on the spot. Do NOT do this for a word/term that has no plausible volleyball-stat meaning at all (gibberish, a typo with no obvious correction, unrelated slang) -- for THOSE, leave metric_of_interest null and list the term in "unrecognized_terms" instead (see below). The difference matters: one path offers to draft a new metric, the other is an honest "I don't understand this" -- never blur them by guessing a typo correction.

Metrics (use for metric_of_interest):
{metrics_block}

Categories (use for skill_group):
{branches_block}

Games currently loaded: {games_block}

Rules for each action:
- "player": the player's name/jersey exactly as the coach said it (e.g. "Sloan", "#7", "Sloan T."). Extract it as plain text ONLY -- do NOT try to resolve it to a real roster entry yourself, that happens downstream. Use null if no specific player was named (meaning "every player").
- "game_hint": the opponent name exactly as mentioned (e.g. "Vegas Aces", "the Mavs game" -> "Mavs"), as plain text ONLY -- resolution against the real loaded games happens downstream. Use null if no specific game was named (meaning "every loaded game", e.g. "throughout all games").
- "title": a short human-readable title for this action's result.
- "pipeline" (optional, omit or use [] for a plain lookup): an ORDERED list of operations to apply to the result before showing it. Order matters -- each operation runs on the output of the previous one. Every operation is one of:
    {{"op": "slice", "axis": "Player"|"Game"|"Metric", "keep": [<values to keep>]}}
        -- restrict an axis to specific values, e.g. {{"op": "slice", "axis": "Player", "keep": ["Sloan"]}}
    {{"op": "slice", "axis": "Player"|"Game"|"Metric", "predicate": {{"metric": "<a metric name>", "op": ">"|">="|"<"|"<="|"=="|"!=" , "threshold": <number>, "decided_by_player": <player name or null>}}}}
        -- restrict an axis to only the rows where ANOTHER metric's value satisfies a comparison, e.g. "games where she had 10+ kills" -> {{"op": "slice", "axis": "Game", "predicate": {{"metric": "Kills", "op": ">=", "threshold": 10, "decided_by_player": null}}}}. Set "decided_by_player" to a specific player's name when ONE player's value should gate the axis for everyone (e.g. "games where SLOAN had 10+ kills" even when showing other players' data); leave it null when each player is judged by their OWN value.
    {{"op": "reduce", "axis": "Player"|"Game"|"Metric", "how": "mean"|"sum"|"min"|"max"|"count"|"median"}}
        -- collapse an axis into one aggregated value, e.g. "her average X across all games" -> {{"op": "reduce", "axis": "Game", "how": "mean"}}
    {{"op": "rank", "axis": "Player"|"Game"|"Metric", "descending": true|false, "limit": <integer or null>}}
        -- order by value along an axis, e.g. "who has the highest X" -> {{"op": "rank", "axis": "Player", "descending": true, "limit": 1}}
    {{"op": "compare", "axis": "Player"|"Game"|"Metric", "how": "mean"|"sum"|"min"|"max"|"count"|"median", "mode": "difference"|"ratio"}}
        -- express each value relative to a baseline (e.g. team average): "is she above team average" -> {{"op": "compare", "axis": "Player", "how": "mean", "mode": "difference"}}
  A predicate's "metric" can name a DIFFERENT metric than the action's own metric_of_interest/skill_group (e.g. slicing Passing by a Kills threshold) -- that's expected and handled downstream, just spell the metric name exactly as it appears in the metrics list above.
- "unrecognized_terms": a list of any word/phrase from the question that has NO plausible mapping to a real metric, category, player, or game (gibberish, unrelated slang, a typo with no obvious correction). Leave it [] if everything mapped to something. This is STRUCTURED -- do not also try to explain it in "reasoning" prose, since free text there risks breaking JSON.

Respond with ONLY this JSON shape, no other text:
{{
  "intent_summary": "one sentence describing what the coach is asking for",
  "actions": [
    {{"metric_of_interest": "Kills Per Set", "skill_group": null, "player": "Sloan", "game_hint": "Vegas Aces", "title": "Sloan's Kills Per Set vs Vegas Aces", "pipeline": []}},
    {{"metric_of_interest": null, "skill_group": "Serve", "player": "Sloan", "game_hint": null, "title": "Sloan's Serving, all games", "pipeline": []}},
    {{"metric_of_interest": "Passing quality", "skill_group": null, "player": "Sloan", "game_hint": null, "title": "Sloan's Passing in games with 10+ Kills", "pipeline": [
        {{"op": "slice", "axis": "Game", "predicate": {{"metric": "Kills", "op": ">=", "threshold": 10, "decided_by_player": null}}}}
    ]}}
  ],
  "reasoning": "brief explanation of how you split/interpreted the question",
  "limitations": "",
  "unrecognized_terms": []
}}
"""

def _repair_pipeline(raw_pipeline: Any, notes: List[str], action_label: str) -> List[Any]:
    """Deterministic repair for one action's pipeline: absent/not-a-list ->
    empty (no-op, backwards compatible with every pre-pipeline query);
    present but containing an invalid op/axis/agg/comparison -> the WHOLE
    pipeline is dropped for that action (not partially applied), same
    all-or-nothing mechanical validation as a MetricSpec's own formula --
    a half-trusted pipeline could silently compute something other than
    what was asked."""
    if not raw_pipeline:
        return []
    if not isinstance(raw_pipeline, list):
        notes.append(f"Dropped pipeline for {action_label!r}: expected a list, got {type(raw_pipeline).__name__}.")
        return []
    try:
        return pipeline_from_dicts(raw_pipeline)
    except ValueError as e:
        notes.append(f"Dropped invalid pipeline for {action_label!r}: {e}")
        return []


def _repair_unrecognized_terms(raw_terms: Any) -> List[str]:
    if not isinstance(raw_terms, list):
        return []
    return [t.strip() for t in raw_terms if isinstance(t, str) and t.strip()]


def _action_shape_key(action: Dict[str, Any]) -> tuple:
    """Everything about an action EXCEPT player/title -- two actions with
    the same key are "the same ask, different player" candidates for
    merge_same_shape_actions. Pipeline is compared by VALUE (via
    pipeline_to_dicts + a sorted-keys JSON dump), not by identity/hash --
    Slice carries a `keep: List[str]` field, which makes the Operation
    dataclasses unhashable, so they can't be used as/in a dict key
    directly."""
    return (
        action.get("metric_of_interest"),
        action.get("skill_group"),
        action.get("game_hint"),
        action.get("is_committed"),
        json.dumps(pipeline_to_dicts(action.get("pipeline") or []), sort_keys=True),
    )


def merge_same_shape_actions(actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Runs after per-action repair, before anything executes. The router is
    explicitly instructed (see _build_router_system_prompt) to split a
    same-metric-different-player question into one action per player --
    intentional, left alone here. This undoes the visible EFFECT of that
    split after the fact (one action, multiple players) rather than
    relying on the model not doing it in the first place.

    Groups actions by everything EXCEPT player/title (see
    _action_shape_key). Within a group, if 2+ actions each have a
    distinct, non-null, single (string) player -- and none already has a
    null or multi-value player -- merges them into ONE action: player
    becomes the list of those raw hints (still unresolved text; resolving
    a hint against the real roster needs known_player_pool, which only
    app.py has -- see its execution loop for where that list gets
    resolved and turned into a Slice(axis="Player", keep=[...]) pipeline
    step), title is regenerated, everything else copied from the first
    action in the group.

    An action with no same-shaped sibling, or one that differs in metric/
    skill_group/game_hint/pipeline, is left completely alone -- a false
    merge (combining two actions that aren't really "the same ask,
    different player") would be worse than the one-chart-per-player bug
    this exists to fix.
    """
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    order: List[tuple] = []
    for action in actions:
        key = _action_shape_key(action)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(action)

    merged: List[Dict[str, Any]] = []
    for key in order:
        group = groups[key]
        players = [a.get("player") for a in group]
        is_mergeable = (
            len(group) >= 2
            and all(isinstance(p, str) and p.strip() for p in players)
            and len({p.strip().lower() for p in players}) == len(players)
        )
        if not is_mergeable:
            merged.extend(group)
            continue

        merged_action = dict(group[0])
        merged_action["player"] = players
        label = merged_action.get("metric_of_interest") or merged_action.get("skill_group") or ""
        vs_players = " vs ".join(players)
        merged_action["title"] = f"{vs_players} -- {label}" if label else vs_players
        merged.append(merged_action)

    return merged


def _validate_and_repair(result: Dict[str, Any], tree: KnowledgeTree) -> Dict[str, Any]:
    """Stage 2 -- deterministic repair. Preserves raw metric names even if they
    are not yet committed in the tree, allowing downstream UIs to trigger
    in-situ knowledge acquisition."""
    valid_metrics = {n.label for n in tree.committed.values() if n.kind == NodeKind.LEAF}
    valid_branches = [n.label for n in tree.committed.values()
                       if n.kind == NodeKind.BRANCH and n.node_id != tree.root_id]
    notes: List[str] = []
    repaired_actions = []

    for action in result.get("actions", []):
        if not isinstance(action, dict):
            notes.append("Dropped a malformed action (not a JSON object).")
            continue

        raw_metric = action.get("metric_of_interest")
        metric = raw_metric if isinstance(raw_metric, str) and raw_metric.strip() else None

        skill_group = None
        if metric is None:
            raw_group = action.get("skill_group")
            if isinstance(raw_group, str) and raw_group.strip():
                skill_group = _resolve_skill_group(raw_group, valid_branches)
                if skill_group is None:
                    notes.append(f"Couldn't identify skill category {raw_group!r}.")

        if metric is None and skill_group is None:
            notes.append("Dropped an action with no metric or category specified.")
            continue

        player = action.get("player")
        if not isinstance(player, str) or not player.strip():
            player = None

        game_hint = action.get("game_hint")
        if not isinstance(game_hint, str) or not game_hint.strip():
            game_hint = None

        title = action.get("title")
        if not isinstance(title, str) or not title.strip():
            title = metric or f"{skill_group} (category)"

        pipeline = _repair_pipeline(action.get("pipeline"), notes, title)

        repaired_actions.append({
            "metric_of_interest": metric,
            "skill_group": skill_group,
            "player": player,
            "game_hint": game_hint,
            "title": title,
            "is_committed": metric in valid_metrics if metric else True,
            "pipeline": pipeline,
        })

    result["actions"] = merge_same_shape_actions(repaired_actions)
    result["unrecognized_terms"] = _repair_unrecognized_terms(result.get("unrecognized_terms"))
    if notes:
        existing = result.get("limitations") or ""
        result["limitations"] = (existing + " " + " ".join(notes)).strip()
    return result

_RETRY_SUFFIX = (
    "\n\nIMPORTANT: Your previous response was not valid JSON and may have been cut off "
    "(e.g. stuck in a repetition loop). Respond with ONLY the JSON object, nothing else, "
    "and keep it concise so it finishes within the token budget."
)


def decompose_recruiting_query(query: str, tree: KnowledgeTree, known_games: List[str]) -> Dict[str, Any]:
    """
    Full pipeline: retrieve candidate metrics -> build a prompt scoped to
    them -> call the LLM -> parse -> mechanically repair -> attach
    retrieved_evidence -> flag (not reject) any retrieval/generation
    mismatch. Raises LLMUnavailableError (propagated straight through
    call_llm) if Groq can't be reached -- callers must catch that
    specifically and show a clear error rather than attempting to
    interpret a result that doesn't exist.

    On malformed/truncated JSON (e.g. the model enters a repetition loop
    and gets cut off at max_tokens before the object closes), retries
    ONCE with a stricter follow-up instruction before falling back to the
    error path -- a truncated response is often just a one-off generation
    hiccup, and a single retry recovers a real answer far more often than
    giving up immediately does.
    """
    retrieval = _retrieve_candidate_metrics(query, tree)
    system_prompt = _build_router_system_prompt(retrieval["candidates"], tree, known_games)

    raw = call_llm(system_prompt, query)
    result = None
    for attempt_prompt in (None, system_prompt + _RETRY_SUFFIX):
        if attempt_prompt is not None:
            raw = call_llm(attempt_prompt, query)
        try:
            result = json.loads(raw)
            if not isinstance(result, dict):
                raise ValueError("top-level JSON was not an object")
            break
        except (json.JSONDecodeError, ValueError):
            result = None
            continue

    if result is None:
        result = {
            "error": True,
            "intent_summary": "", "actions": [], "reasoning": "",
            "limitations": f"Couldn't parse the LLM's response as JSON, even after a retry: {raw[:200]!r}",
            "unrecognized_terms": [],
        }
        result["retrieved_evidence"] = {
            "matched_aliases": retrieval["matched_aliases"],
            "candidate_metrics": retrieval["candidates"],
            "fallback_to_full_taxonomy": retrieval["fallback"],
        }
        return result

    result = _validate_and_repair(result, tree)
    result["retrieved_evidence"] = {
        "matched_aliases": retrieval["matched_aliases"],
        "candidate_metrics": retrieval["candidates"],
        "fallback_to_full_taxonomy": retrieval["fallback"],
    }

    if not retrieval["fallback"]:
        # Category actions (skill_group set, metric_of_interest None) aren't
        # part of metric retrieval at all -- only single-metric actions can
        # meaningfully mismatch against the retrieved metric candidates.
        mismatches = sorted({
            a["metric_of_interest"] for a in result.get("actions", [])
            if a.get("metric_of_interest") and a["metric_of_interest"] not in retrieval["candidates"]
        })
        if mismatches:
            note = (
                f"Retrieval/generation mismatch: model selected {mismatches}, not among "
                f"retrieved candidates {retrieval['candidates']}."
            )
            result["limitations"] = (result.get("limitations", "") + " " + note).strip()

    return result
