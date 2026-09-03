"""
volley_seed.py
===============
Seeds the EVENT-grain knowledge tree: one branch per skill, one leaf per
PRIMITIVE (a (skill, evaluation_code) count), then the derived rate and
efficiency metrics composed on top of those primitives.

This is the event-grain twin of seed_recruiting_tree() in
recruiting_tree.py, and deliberately mirrors its shape: branches are
skills because the DataVolley skill field already groups actions that
way, exactly as the Huddle CSV's column prefixes already grouped its
columns. Neither taxonomy is invented.

────────────────────────────────────────────────────────────────
THE PRIMITIVE NAMES ARE EVIDENCE-BASED, NOT COPIED FROM A SPEC
────────────────────────────────────────────────────────────────
Every "this code means X" claim below was checked against rally
outcomes across 8 real matches (~5,600 evaluated actions) by asking:
what share of rallies did the ACTING player's team win?

    Attack #   100.0%  (674)   -> terminal, in the attacker's favour  = KILL
    Attack =     0.0%  (128)   -> terminal, against                   = ERROR
    Attack /     0.0%  (115)   -> terminal, against                   = BLOCKED
    Serve  #   100.0%   (86)   -> terminal, in favour                 = ACE
    Serve  =     0.0%  (121)   -> terminal, against                   = ERROR
    Block  #   100.0%  (116)   -> terminal, in favour                 = STUFF BLOCK
    Block  =     0.0%  (220)   -> terminal, against                   = ERROR
    Reception =  0.0%   (86)   -> terminal, against                   = ERROR
    Reception /  18.0%  (50)   -> mostly against                      = OVERPASS
    Dig    =     0.0%  (278)   -> terminal, against                   = ERROR
    Set    =     0.0%    (4)   -> terminal, against                   = ERROR

The non-terminal codes (+, -, !) all sit in the 30-70% band, i.e. the
rally continued, so they are named neutrally ("Positive ...") rather
than being given outcome words the data does not support. Block ! is
0% across 53 events -- terminal AGAINST the blocking team -- so it is
deliberately left unnamed rather than guessed at.

BRANCH NAMES TRACK THE ROUTER, NOT THE FILE FORMAT.
The DataVolley skill value is "Reception", but the branch is called
"Receive" so that recruiting_llm's _SKILL_GROUP_SYNONYMS ("passing",
"pass", "receiving" -> "Receive") keeps resolving category questions
here exactly as it does for the CSV app. The skill VALUE used in the
spec is still the real "Reception"; only the human-facing branch label
is aligned.
"""

from typing import Dict, List, Tuple

import _parent_path  # noqa: F401
from recruiting_tree import KnowledgeTree, NodeKind

from volley_event_spec import SETS_PLAYED, make_event_spec, make_measure_spec, make_metric_formula_spec
from volley_source import SourceSchema

AUTHOR = "system:dvw_import"

# branch label -> list of (leaf label, where-clause, description, aliases)
PRIMITIVES: Dict[str, List[Tuple[str, dict, str, List[str]]]] = {
    "Attack": [
        ("Kills", {"skill": "Attack", "evaluation_code": "#"},
         "Attacks that ended the rally in the attacker's favour.", ["kill", "kills"]),
        ("Attack Errors", {"skill": "Attack", "evaluation_code": "="},
         "Attacks that ended the rally against the attacker.", ["attack error", "hitting error"]),
        ("Blocked Attacks", {"skill": "Attack", "evaluation_code": "/"},
         "Attacks stuffed by the opposing block.", ["blocked", "stuffed"]),
        ("Positive Attacks", {"skill": "Attack", "evaluation_code": "+"},
         "Attacks the scout graded positive; the rally continued.", ["positive attack"]),
        ("Attack Attempts", {"skill": "Attack"},
         "Every attack, regardless of outcome.", ["attack attempts", "total attacks", "swings"]),
    ],
    "Serve": [
        ("Aces", {"skill": "Serve", "evaluation_code": "#"},
         "Serves that ended the rally immediately in the server's favour.", ["ace", "aces"]),
        ("Service Errors", {"skill": "Serve", "evaluation_code": "="},
         "Serves that ended the rally against the server.", ["service error", "serve error"]),
        ("Strong Serves", {"skill": "Serve", "evaluation_code": "/"},
         "Serves graded one below an ace; the receiving team won only 18% of these rallies.",
         ["strong serve"]),
        ("Serve Attempts", {"skill": "Serve"},
         "Every serve, regardless of outcome.", ["serve attempts", "total serves"]),
    ],
    "Receive": [
        ("Perfect Passes", {"skill": "Reception", "evaluation_code": "#"},
         "Receptions graded perfect.", ["perfect pass", "perfect passes"]),
        ("Positive Passes", {"skill": "Reception", "evaluation_code": ["#", "+"]},
         "Receptions graded perfect or positive.", ["positive pass", "good pass"]),
        ("Reception Errors", {"skill": "Reception", "evaluation_code": "="},
         "Receptions that ended the rally against the receiver.", ["reception error", "passing error"]),
        ("Overpasses", {"skill": "Reception", "evaluation_code": "/"},
         "Receptions sent over the net; the receiving team won only 18% of these rallies.",
         ["overpass", "overpasses"]),
        ("Reception Attempts", {"skill": "Reception"},
         "Every reception, regardless of outcome.", ["reception attempts", "passes", "total receptions"]),
    ],
    "Set": [
        ("Perfect Sets", {"skill": "Set", "evaluation_code": "#"},
         "Sets graded perfect.", ["perfect set", "perfect sets"]),
        ("Setting Errors", {"skill": "Set", "evaluation_code": "="},
         "Sets that ended the rally against the setter.", ["setting error", "set error"]),
        ("Set Attempts", {"skill": "Set"},
         "Every set, regardless of outcome.", ["set attempts", "total sets", "assists"]),
    ],
    "Dig": [
        ("Digs", {"skill": "Dig"},
         "Every dig, regardless of outcome.", ["dig", "digs"]),
        ("Perfect Digs", {"skill": "Dig", "evaluation_code": "#"},
         "Digs graded perfect.", ["perfect dig"]),
        ("Dig Errors", {"skill": "Dig", "evaluation_code": "="},
         "Digs that ended the rally against the digger.", ["dig error"]),
    ],
    "Block": [
        ("Kill Blocks", {"skill": "Block", "evaluation_code": "#"},
         "Blocks that ended the rally in the blocker's favour.", ["kill block", "stuff", "stuff block"]),
        ("Block Errors", {"skill": "Block", "evaluation_code": "="},
         "Blocks that ended the rally against the blocker.", ["block error"]),
        ("Block Touches", {"skill": "Block"},
         "Every block touch, regardless of outcome.", ["block touches", "total blocks"]),
    ],
    "Freeball": [
        ("Freeballs", {"skill": "Freeball"},
         "Every freeball, regardless of outcome.", ["freeball", "freeballs"]),
    ],
}

# Derived metrics, composed over the primitives above. These are the
# event-grain equivalents of the CSV world's rate/efficiency columns --
# the difference is that there they were read off the export, and here
# they are constructed, which is what makes them auditable.
DERIVED: List[Tuple[str, str, str, str, List[str]]] = [
    ("Attack", "Kills Per Set", "[Kills] / [Sets Played]",
     "Kills per set played.", ["kills per set", "kps"]),
    ("Attack", "Hitting Efficiency", "([Kills] - [Attack Errors]) / [Attack Attempts]",
     "(kills - errors) / total attacks, the standard hitting efficiency.",
     ["hitting efficiency", "hitting percentage", "attack efficiency"]),
    ("Attack", "Kill Rate", "[Kills] / [Attack Attempts]",
     "Share of attacks that were kills.", ["kill rate", "kill percentage"]),
    ("Serve", "Aces Per Set", "[Aces] / [Sets Played]",
     "Service aces per set played.", ["aces per set"]),
    ("Serve", "Service Error Rate", "[Service Errors] / [Serve Attempts]",
     "Share of serves that were errors.", ["service error rate", "serve error rate"]),
    ("Receive", "Positive Pass Rate", "[Positive Passes] / [Reception Attempts]",
     "Share of receptions graded perfect or positive.", ["positive pass rate", "passing percentage"]),
    ("Receive", "Reception Error Rate", "[Reception Errors] / [Reception Attempts]",
     "Share of receptions that were errors.", ["reception error rate", "passing error rate"]),
    ("Dig", "Digs Per Set", "[Digs] / [Sets Played]",
     "Digs per set played.", ["digs per set"]),
    ("Block", "Blocks Per Set", "[Kill Blocks] / [Sets Played]",
     "Kill blocks per set played.", ["blocks per set"]),
    ("Sets", "Points Per Set", "([Kills] + [Aces] + [Kill Blocks]) / [Sets Played]",
     "Terminal points (kills + aces + kill blocks) per set played.",
     ["points per set", "scoring"]),
]


def seed_volley_tree(schema: SourceSchema) -> Tuple[KnowledgeTree, Dict[str, str]]:
    """
    Build the committed event-grain tree.

    `schema` comes from the live Source, so every primitive is validated
    against the values those matches actually contain -- a primitive
    naming a (skill, code) pair absent from the data is rejected at seed
    time instead of silently returning zero forever.

    Seeds straight into `committed` (to="committed" is for initial
    taxonomy only, same convention as seed_recruiting_tree), then mirrors
    committed into staging so the caller can start staging edits.

    Returns (tree, branch_ids).
    """
    tree = KnowledgeTree()
    tree.add_root()
    branch_ids: Dict[str, str] = {}

    for branch_label, primitives in PRIMITIVES.items():
        branch_ids[branch_label] = tree.add_node(
            label=branch_label, kind=NodeKind.BRANCH, parent_id=tree.root_id, to="committed",
        )
        for leaf_label, where, description, aliases in primitives:
            spec = make_event_spec(where, description, schema)
            errors = spec.validate()
            if errors:
                # The data cannot express this primitive -- skip it
                # rather than commit a metric that can only ever be 0.
                continue
            tree.add_node(
                label=leaf_label, kind=NodeKind.LEAF, parent_id=branch_ids[branch_label],
                to="committed", spec=spec, aliases=aliases, authored_by=AUTHOR,
            )

    # "Sets" holds the one per-(player, match) measure that is read from
    # roster metadata rather than counted from actions -- every per-set
    # rate below divides by it.
    branch_ids["Sets"] = tree.add_node(
        label="Sets", kind=NodeKind.BRANCH, parent_id=tree.root_id, to="committed",
    )
    tree.add_node(
        label="Sets Played", kind=NodeKind.LEAF, parent_id=branch_ids["Sets"], to="committed",
        spec=make_measure_spec(SETS_PLAYED, "Sets this player was on court for, per match."),
        aliases=["sets played", "sets"], authored_by=AUTHOR,
    )

    # Derived metrics last: their validator checks tokens against the
    # labels committed above, so ordering is load-bearing.
    committed_labels = {
        node.label for node in tree.committed.values() if node.kind == NodeKind.LEAF
    }
    for branch_label, leaf_label, expression, description, aliases in DERIVED:
        branch_id = branch_ids.get(branch_label)
        if branch_id is None:
            continue
        spec = make_metric_formula_spec(expression, description, committed_labels)
        if spec.validate():
            continue
        tree.add_node(
            label=leaf_label, kind=NodeKind.LEAF, parent_id=branch_id, to="committed",
            spec=spec, aliases=aliases, authored_by=AUTHOR,
        )
        committed_labels.add(leaf_label)

    tree.seed_from_committed()
    return tree, branch_ids
