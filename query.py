"""
query.py
================
Everything between "the router produced a decomposition" and "here is a
frame to draw": name resolution, action execution, the pipeline hand-off,
and the tidy/consolidate step.

Kept OUT of app.py on purpose. The CSV app interleaves this logic with
widget calls, which is why its own test suite has to re-implement the
execution loop by hand to test it (see _run_repaired_actions in
test_integration.py, whose docstring says exactly that). Here the whole
decision path is importable without a Streamlit runtime, so the tests
exercise the real code.

The cube contract is unchanged from the CSV app -- Player x Game x
Metric -> Value, in a long frame with columns Game/Player/Value/Note --
so recruiting_operations' Slice/Reduce/Rank/Compare and
recruiting_encoding's chart encoding are imported and used as-is rather
than reimplemented for the event grain.
"""

import difflib
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from recruiting_operations import Operation, Rank, Slice, run_pipeline
from recruiting_tree import KnowledgeTree, Node, NodeKind

from evaluate import evaluate_category, evaluate_metric
from source import Source

PLAYER_AXIS = "Player"
GAME_AXIS = "Game"


# ──────────────────────────────────────────────────────────────
# NAME RESOLUTION -- the LLM extracts raw text, Python resolves it
# against what really exists. Same tiered strategy the CSV app uses
# (exact -> substring -> fuzzy), so a hint behaves identically here.
# ──────────────────────────────────────────────────────────────

def _strip_jersey(label: str) -> str:
    """"#24 Ally Williams" -> "ally williams". Player labels carry the
    jersey number because names are not unique on a roster, but a coach
    asking about "Ally" should still match."""
    return re.sub(r"^#\S*\s*", "", label).strip().lower()


def resolve_player_name(hint: Optional[str], known_players: List[str]) -> Optional[str]:
    if not hint or not hint.strip() or not known_players:
        return None
    needle = hint.strip().lower()

    exact = [p for p in known_players if p.strip().lower() == needle]
    if exact:
        return exact[0]

    # Jersey-only hints ("#7", "7") are common from a coach.
    bare = needle.lstrip("#")
    if bare.isdigit():
        numbered = [p for p in known_players if p.startswith(f"#{bare} ")]
        if numbered:
            return numbered[0]

    substring = [p for p in known_players
                 if needle in _strip_jersey(p) or _strip_jersey(p) in needle]
    if substring:
        return substring[0]

    fuzzy = [p for p in known_players
             if difflib.SequenceMatcher(None, needle, _strip_jersey(p)).ratio() >= 0.75]
    return fuzzy[0] if fuzzy else None


def resolve_game_hint(hint: Optional[str], known_games: List[str]) -> List[str]:
    """Returns EVERY match the hint could mean, not one: this corpus has
    three opponents played twice, so "the Penn State game" legitimately
    resolves to two matches and the caller shows both rather than
    silently picking one."""
    if not hint or not hint.strip() or not known_games:
        return []
    needle = hint.strip().lower()

    exact = [g for g in known_games if g.lower() == needle]
    if exact:
        return exact

    substring = [g for g in known_games if needle in g.lower()]
    if substring:
        return substring

    fuzzy = [g for g in known_games
             if difflib.SequenceMatcher(None, needle, g.lower()).ratio() >= 0.7]
    return fuzzy


def get_branches(tree: KnowledgeTree) -> Dict[str, str]:
    root = tree.committed.get(tree.root_id)
    if not root:
        return {}
    return {
        node.label: child_id
        for child_id in root.children
        for node in [tree.committed.get(child_id)]
        if node and node.kind == NodeKind.BRANCH
    }


def find_leaf_by_exact_label(tree: KnowledgeTree, label: Optional[str]) -> Optional[Node]:
    if not label:
        return None
    return next(
        (n for n in tree.committed.values() if n.kind == NodeKind.LEAF and n.label == label),
        None,
    )


def collect_leaf_labels(tree: KnowledgeTree) -> Set[str]:
    return {n.label for n in tree.committed.values() if n.kind == NodeKind.LEAF}


def known_players(source: Source) -> List[str]:
    facts = source.facts()
    if facts.empty:
        return []
    player_column = source.identity_fields()[PLAYER_AXIS]
    return sorted(facts[player_column].dropna().unique())


def known_games(source: Source) -> List[str]:
    return source.game_labels()


# ──────────────────────────────────────────────────────────────
# PIPELINE SUPPORT
# ──────────────────────────────────────────────────────────────

def _pipeline_referenced_metrics(pipeline: List[Operation]) -> Set[str]:
    """Metric names a pipeline's Slice predicates reference. Only a
    Slice predicate can name a metric OTHER than the one being computed
    (e.g. slicing passing by a kills threshold), so only those need
    fetching alongside."""
    return {
        op.predicate.metric
        for op in pipeline
        if isinstance(op, Slice) and op.predicate is not None
    }


def prepare_pipeline_frame(result_df: pd.DataFrame, pipeline: List[Operation],
                            primary_metric_label: Optional[str], tree: KnowledgeTree,
                            source: Source) -> Tuple[pd.DataFrame, List[str]]:
    """
    Guarantees a Metric column, then fetches any EXTRA metric a
    cross-metric Slice predicate references so run_pipeline sees one
    consistent long frame. Only called when a pipeline is non-empty, so
    a plain lookup's frame is untouched.
    """
    notes: List[str] = []
    if "Metric" not in result_df.columns:
        result_df = result_df.copy()
        result_df.insert(0, "Metric", primary_metric_label)

    already_have = set(result_df["Metric"].dropna().unique())
    frames = [result_df]
    for extra_label in sorted(_pipeline_referenced_metrics(pipeline) - already_have):
        leaf = find_leaf_by_exact_label(tree, extra_label)
        if leaf is None or leaf.spec is None:
            notes.append(f"Pipeline referenced unknown metric '{extra_label}' -- skipped.")
            continue
        extra = evaluate_metric(leaf.spec, source, tree)
        extra.insert(0, "Metric", extra_label)
        frames.append(extra)

    combined = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    return combined, notes


def execute_query_actions(decomposition: Dict[str, Any], tree: KnowledgeTree, source: Source,
                           selected_games: Optional[List[str]] = None,
                           player_filter: Optional[List[str]] = None) -> List[dict]:
    """
    Run every action in an already-decomposed query. No LLM call.

    Mirrors the CSV app's execute_query_actions decision-for-decision --
    unresolved metric becomes a missing_metric result the UI can offer to
    author, player narrowing is applied as a trailing Slice so any
    Rank/Reduce/Compare still sees the full roster first, and
    `player_filter` overrides whatever player the action itself named so
    a Player Key click re-runs the same question for someone else.
    """
    unrecognized = {t.lower() for t in decomposition.get("unrecognized_terms", [])}
    roster = known_players(source)
    all_games = known_games(source)
    selected_games = selected_games if selected_games is not None else all_games

    results: List[dict] = []
    for action in decomposition.get("actions", []):
        is_category = bool(action.get("skill_group"))
        pipeline = list(action.get("pipeline") or [])

        if is_category:
            branch_id = get_branches(tree).get(action["skill_group"])
            if branch_id is None:
                results.append({"action": action, "error": "Category no longer exists in committed tree."})
                continue
            leaf = None
        else:
            leaf = find_leaf_by_exact_label(tree, action.get("metric_of_interest"))
            if leaf is None or leaf.spec is None:
                raw_name = action.get("metric_of_interest") or "Unknown Metric"
                results.append({
                    "action": action,
                    "status": "missing_metric",
                    "raw_metric_name": raw_name,
                    "is_gibberish": raw_name.strip().lower() in unrecognized,
                })
                continue

        if player_filter:
            resolved_player = player_filter[0] if len(player_filter) == 1 else None
            multi_players = player_filter if len(player_filter) >= 2 else None
        else:
            raw_player = action.get("player")
            if isinstance(raw_player, list):
                resolved, seen = [], set()
                for hint in raw_player:
                    candidate = resolve_player_name(hint, roster)
                    if candidate and candidate not in seen:
                        seen.add(candidate)
                        resolved.append(candidate)
                multi_players = resolved if len(resolved) >= 2 else None
                resolved_player = resolved[0] if len(resolved) == 1 else None
            else:
                resolved_player = resolve_player_name(raw_player, roster)
                multi_players = None

        game_note = None
        if action.get("game_hint"):
            hits = resolve_game_hint(action["game_hint"], all_games)
            if hits:
                action_games = hits
            else:
                action_games = selected_games
                game_note = (
                    f"Couldn't identify game '{action['game_hint']}' -- "
                    "showing all selected matches instead."
                )
        else:
            action_games = selected_games

        if is_category:
            frame = evaluate_category(tree, branch_id, source)
        else:
            frame = evaluate_metric(leaf.spec, source, tree)

        # Game scoping is a frame filter here rather than a fetch-time
        # choice: the evaluator computes the whole cube in one vectorized
        # pass, so narrowing afterwards costs nothing and keeps one code
        # path for "all matches" and "one match".
        if action_games is not None:
            frame = frame[frame[GAME_AXIS].isin(action_games)].reset_index(drop=True)

        pipeline_notes: List[str] = []
        if pipeline or multi_players or player_filter:
            if player_filter:
                pipeline = [op for op in pipeline
                            if not (isinstance(op, Slice) and op.axis == PLAYER_AXIS)]
                pipeline.append(Slice(axis=PLAYER_AXIS, keep=player_filter))
            else:
                has_player_slice = any(
                    isinstance(op, Slice) and op.axis == PLAYER_AXIS for op in pipeline
                )
                if not has_player_slice:
                    if multi_players:
                        pipeline.append(Slice(axis=PLAYER_AXIS, keep=multi_players))
                    elif resolved_player is not None:
                        pipeline.append(Slice(axis=PLAYER_AXIS, keep=[resolved_player]))

            has_metric_slice = any(
                isinstance(op, Slice) and op.axis == "Metric" for op in pipeline
            )
            primary_metric = action.get("metric_of_interest")
            if not is_category and primary_metric and not has_metric_slice:
                pipeline.append(Slice(axis="Metric", keep=[primary_metric]))

            combined, fetch_notes = prepare_pipeline_frame(
                frame, pipeline, action.get("metric_of_interest"), tree, source,
            )
            frame, run_notes = run_pipeline(combined, pipeline)
            pipeline_notes = fetch_notes + run_notes
        elif resolved_player is not None:
            frame = frame[frame[PLAYER_AXIS] == resolved_player].reset_index(drop=True)

        results.append({
            "action": action, "is_category": is_category, "resolved_player": resolved_player,
            "action_games": action_games, "game_note": game_note, "result_df": frame,
            "pipeline": pipeline, "pipeline_notes": pipeline_notes,
        })

    return results


# ──────────────────────────────────────────────────────────────
# TIDY / CONSOLIDATE / FORMAT
# ──────────────────────────────────────────────────────────────

def tidy_data(df: pd.DataFrame, metric_label: Optional[str] = None) -> pd.DataFrame:
    """Long observations -> a wide (Player, Game) x Metric matrix."""
    tidy = df.copy()
    if "Metric" not in tidy.columns:
        tidy.insert(0, "Metric", metric_label)
    tidy["Value"] = pd.to_numeric(tidy["Value"], errors="coerce")
    tidy = tidy.drop_duplicates(subset=[PLAYER_AXIS, GAME_AXIS, "Metric"], keep="first")

    wide = tidy.pivot(index=[PLAYER_AXIS, GAME_AXIS], columns="Metric", values="Value")
    wide = wide.reset_index()
    wide.columns.name = None
    return wide.sort_values([PLAYER_AXIS, GAME_AXIS], kind="stable").reset_index(drop=True)


def consolidate_action_results(action_results: List[dict]) -> pd.DataFrame:
    """Merge every action's output into ONE wide table keyed by
    (Player, Game), so a multi-part question reads as a single matrix."""
    wide_frames = []
    for item in action_results:
        if "error" in item or item.get("status") == "missing_metric":
            continue
        frame = item.get("result_df")
        if frame is None or frame.empty:
            continue
        wide = tidy_data(frame, metric_label=item["action"].get("metric_of_interest"))
        if not wide.empty:
            wide_frames.append(wide)

    if not wide_frames:
        return pd.DataFrame()

    consolidated = wide_frames[0]
    for nxt in wide_frames[1:]:
        metric_cols = [c for c in nxt.columns if c not in (PLAYER_AXIS, GAME_AXIS)]
        shared = [c for c in metric_cols if c in consolidated.columns]
        new = [c for c in metric_cols if c not in consolidated.columns]
        consolidated = pd.merge(
            consolidated, nxt[[PLAYER_AXIS, GAME_AXIS] + shared + new],
            on=[PLAYER_AXIS, GAME_AXIS], how="outer", suffixes=("", "_dup"),
        )
        for column in shared:
            duplicate = f"{column}_dup"
            if duplicate in consolidated.columns:
                consolidated[column] = consolidated[column].combine_first(consolidated[duplicate])
                consolidated = consolidated.drop(columns=[duplicate])

    return consolidated.sort_values([PLAYER_AXIS, GAME_AXIS], kind="stable").reset_index(drop=True)


def metric_format_pattern(tree: KnowledgeTree, metric_label: str) -> str:
    """
    Formatting follows the SPEC KIND, which at event grain is exact:
    an event count and a sets-played measure are whole numbers, and a
    formula is not. Falls back to the label only for the rate/efficiency
    distinction, which the payload genuinely does not encode.
    """
    leaf = find_leaf_by_exact_label(tree, metric_label)
    kind = leaf.spec.payload.get("kind") if (leaf and leaf.spec) else None

    if kind in ("event", "measure"):
        return "{:.0f}"

    lowered = metric_label.lower()
    if "rate" in lowered or "efficiency" in lowered or "%" in lowered or "percentage" in lowered:
        return "{:.1%}"
    return "{:.2f}"


def format_table(df: pd.DataFrame, tree: KnowledgeTree):
    """Styler with per-metric formatting and an em dash for missing
    values -- a blank cell and a zero mean different things here (did
    not play vs did not record one), so they must not look alike."""
    formats = {
        column: metric_format_pattern(tree, str(column))
        for column in df.columns if column not in (PLAYER_AXIS, GAME_AXIS)
    }
    return df.style.format(formats, na_rep="—")
