"""
volley_llm.py
==============
LLM-backed metric AUTHORING for the event grain. Query ROUTING is not
re-implemented here: recruiting_llm.decompose_recruiting_query is driven
entirely by the tree's committed labels and the list of game names, both
of which this app supplies, so it routes event-grain questions without
modification and is imported as-is.

What could not be reused is the metric author. Its prompt
(_build_metric_author_system_prompt) is built from
RECRUITING_COLUMN_SCHEMA -- Huddle CSV column names -- which mean
nothing to a .dvw file. An event-grain author has to propose a FILTER
over (skill, evaluation_code) instead of arithmetic over columns, so it
needs its own prompt and its own JSON shape.

Same trust boundary as the CSV app: the LLM only ever proposes a
CANDIDATE. Whatever comes back is run through the real
MetricSpec.validate() from volley_event_spec.py -- which checks the
fields exist, the values occur in these matches, and the evaluation
code is legal FOR THAT SKILL -- before anything is staged. The model
cannot invent a skill, a code, or a metric that is not really there.
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import _parent_path  # noqa: F401
from recruiting_llm import LLMUnavailableError, call_llm
from recruiting_tree import KnowledgeTree, MetricSpec, NodeKind

from volley_event_spec import (
    COUNT, VALID_AGGREGATES, make_event_spec, make_metric_formula_spec,
)
from volley_source import SourceSchema

SKILL_FIELD = "skill"
EVALUATION_FIELD = "evaluation_code"


@dataclass
class EventParseResult:
    """
    Mirrors recruiting_llm.ParseResult so the app's authoring wizard
    reads the same either way, with the payload widened to cover an
    event spec's filter.
    """

    matched: bool
    spec_kind: Optional[str] = None            # "event" or "formula"
    where: Optional[Dict[str, object]] = None
    aggregate: str = COUNT
    formula_expr: Optional[str] = None
    human_description: Optional[str] = None
    suggested_label: Optional[str] = None
    suggested_branch_group: Optional[str] = None
    suggested_aliases: List[str] = field(default_factory=list)
    message: Optional[str] = None


def _existing_metric_labels(tree: KnowledgeTree) -> List[str]:
    return sorted(
        node.label for node in tree.committed.values() if node.kind == NodeKind.LEAF
    )


def _branch_labels(tree: KnowledgeTree) -> List[str]:
    return sorted(
        node.label for node in tree.committed.values()
        if node.kind == NodeKind.BRANCH and node.node_id != tree.root_id
    )


def build_author_system_prompt(schema: SourceSchema, tree: KnowledgeTree) -> str:
    """
    The vocabulary shown to the model is DERIVED FROM THE LOADED MATCHES
    (schema), not from a DataVolley reference table. The per-skill code
    listing matters most: presenting a flat list of six codes invites a
    proposal like "a Set evaluated /", which never occurs and would
    count zero forever.
    """
    vocabulary = schema.dependent_values.get((SKILL_FIELD, EVALUATION_FIELD), {})
    skill_lines = []
    for skill in sorted(vocabulary):
        codes = ", ".join(sorted(vocabulary[skill]))
        skill_lines.append(f'- "{skill}": evaluation codes {codes}')
    skills_block = "\n".join(skill_lines) or "(no skills available)"

    other_fields = [
        name for name in schema.field_names()
        if name not in (SKILL_FIELD, EVALUATION_FIELD) and schema.field_values(name)
    ]
    field_lines = []
    for name in other_fields:
        values = sorted(schema.field_values(name))
        shown = ", ".join(values[:12]) + (" ..." if len(values) > 12 else "")
        field_lines.append(f'- "{name}": {shown}')
    fields_block = "\n".join(field_lines) or "(no additional filterable fields)"

    metrics_block = "\n".join(f"- [{label}]" for label in _existing_metric_labels(tree)) or "(none yet)"
    groups = ", ".join(_branch_labels(tree))

    return f"""You are the metric-authoring assistant for a volleyball recruiting knowledge base built on DataVolley play-by-play data.

Each row of the underlying data is ONE ACTION by one player. A metric is therefore either:
  (a) an EVENT metric -- a filter over action fields, counted; or
  (b) a FORMULA -- arithmetic over metrics that already exist.

CRITICAL GROUNDING RULES:
1. ONLY use skills, evaluation codes and field values listed below. They are the values that actually occur in the loaded matches.
2. An evaluation code means a DIFFERENT thing depending on its skill, and is not legal for every skill. "#" is an ace on a Serve, a perfect pass on a Reception and a kill on an Attack. Only pair a code with a skill it is listed under.
3. DO NOT invent a skill, a code, a field, or a metric name. If the request cannot be expressed with what is listed, return matched=false.
4. Do not guess at gibberish or unrelated slang. Return matched=false instead.

SKILLS and the evaluation codes each one actually takes:
{skills_block}

OTHER FILTERABLE FIELDS:
{fields_block}

EXISTING METRICS (a formula may reference these, in [brackets]):
{metrics_block}

Skill groups a new metric can be filed under: {groups}

For an EVENT metric respond with ONLY this JSON:
{{
  "matched": true,
  "kind": "event",
  "where": {{"skill": "Attack", "evaluation_code": "#"}},
  "aggregate": "count",
  "human_description": "Attacks that ended the rally in the attacker's favour.",
  "suggested_label": "Kills",
  "suggested_branch_group": "Attack",
  "suggested_aliases": ["kill", "kills"]
}}

A "where" value may be a list to mean OR, e.g. {{"evaluation_code": ["#", "+"]}}.

For a FORMULA metric respond with ONLY this JSON:
{{
  "matched": true,
  "kind": "formula",
  "expr": "[Kills] / [Sets Played]",
  "human_description": "Kills per set played.",
  "suggested_label": "Kills Per Set",
  "suggested_branch_group": "Attack",
  "suggested_aliases": ["kills per set"]
}}

If you cannot express the request with the listed values, respond with exactly:
{{"matched": false, "message": "<brief reason>"}}
"""


def parse_phrase_to_event_spec(phrase: str, schema: SourceSchema,
                                tree: KnowledgeTree) -> EventParseResult:
    """
    Ask the model for a candidate. Raises LLMUnavailableError straight
    through from call_llm when Groq cannot be reached, so the caller can
    say so plainly rather than showing an invented metric.

    Nothing here is trusted: build_spec() below re-validates whatever
    comes back against the real schema.
    """
    if not phrase or not phrase.strip():
        return EventParseResult(matched=False, message="Type a description first.")

    raw = call_llm(build_author_system_prompt(schema, tree), phrase)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return EventParseResult(
            matched=False, message=f"The LLM returned something that wasn't valid JSON: {raw[:200]!r}",
        )

    if not isinstance(data, dict) or not data.get("matched"):
        message = data.get("message") if isinstance(data, dict) else None
        return EventParseResult(
            matched=False,
            message=message or "The LLM couldn't express this with the available fields.",
        )

    kind = data.get("kind")
    aliases = [a for a in (data.get("suggested_aliases") or []) if isinstance(a, str)]
    common = {
        "human_description": data.get("human_description") or phrase.strip(),
        "suggested_label": data.get("suggested_label") or phrase.strip().title(),
        "suggested_branch_group": data.get("suggested_branch_group"),
        "suggested_aliases": aliases,
    }

    if kind == "formula":
        expression = data.get("expr")
        if not isinstance(expression, str) or not expression.strip():
            return EventParseResult(matched=False, message="The LLM's formula was empty.")
        return EventParseResult(matched=True, spec_kind="formula",
                                 formula_expr=expression.strip(), **common)

    if kind == "event":
        where = data.get("where")
        if not isinstance(where, dict) or not where:
            return EventParseResult(
                matched=False, message="The LLM's event metric had no 'where' filter.",
            )
        aggregate = data.get("aggregate", COUNT)
        if aggregate not in VALID_AGGREGATES:
            aggregate = COUNT
        return EventParseResult(matched=True, spec_kind="event", where=where,
                                 aggregate=aggregate, **common)

    return EventParseResult(
        matched=False, message=f"The LLM returned an unknown metric kind {kind!r}.",
    )


def build_spec(result: EventParseResult, schema: SourceSchema,
                tree: KnowledgeTree) -> MetricSpec:
    """
    Turn a proposal into a real MetricSpec -- still UNVALIDATED here.
    The caller runs .validate() (or lets tree.add_node do it) so a bad
    proposal is rejected on exactly the same path a hand-typed one is.
    """
    if result.spec_kind == "formula":
        return make_metric_formula_spec(
            result.formula_expr, result.human_description or "", _existing_metric_labels(tree),
        )
    return make_event_spec(
        result.where or {}, result.human_description or "", schema, aggregate=result.aggregate,
    )
