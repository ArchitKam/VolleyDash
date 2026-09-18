import type { SourceSchema } from "./source";
import {
  ROOT_LABEL,
  newId,
  validateSpec,
  type EventWhere,
  type GraphNode,
  type GraphTree,
} from "./tree";

const AUTHOR = "system:dvw_import";

/** The DataVolley skill VALUE is "Reception"; only the human-facing branch is "Receive". */
export const DVW_BRANCHES = [
  "Attack",
  "Serve",
  "Receive",
  "Set",
  "Dig",
  "Block",
  "Freeball",
  "Sets",
] as const;

type Primitive = {
  label: string;
  where: EventWhere;
  description: string;
  aliases: string[];
};

const PRIMITIVES: Record<string, Primitive[]> = {
  Attack: [
    { label: "Kills", where: { skill: "Attack", evaluation_code: "#" }, description: "Attacks that ended the rally as a point.", aliases: ["kill", "kills", "attack kills"] },
    { label: "Attack Errors", where: { skill: "Attack", evaluation_code: "=" }, description: "Attacks that gave the opponent the point.", aliases: ["attack error", "attack errors", "hitting errors"] },
    { label: "Blocked Attacks", where: { skill: "Attack", evaluation_code: "/" }, description: "Attacks stuffed by the opposing block.", aliases: ["blocked", "blocked attacks", "stuffed"] },
    { label: "Positive Attacks", where: { skill: "Attack", evaluation_code: "+" }, description: "Attacks that put the opponent out of system.", aliases: ["positive attacks", "good attacks"] },
    { label: "Attack Attempts", where: { skill: "Attack" }, description: "Every attack swing.", aliases: ["attempts", "attack attempts", "swings", "total attacks"] },
  ],
  Serve: [
    { label: "Aces", where: { skill: "Serve", evaluation_code: "#" }, description: "Serves that scored directly.", aliases: ["ace", "aces", "service aces"] },
    { label: "Service Errors", where: { skill: "Serve", evaluation_code: "=" }, description: "Serves that went out or into the net.", aliases: ["service error", "service errors", "missed serves"] },
    { label: "Strong Serves", where: { skill: "Serve", evaluation_code: "/" }, description: "Serves that forced an overpass or broke the opponent's offence.", aliases: ["strong serves", "tough serves"] },
    { label: "Serve Attempts", where: { skill: "Serve" }, description: "Every serve.", aliases: ["serves", "serve attempts", "total serves"] },
  ],
  Receive: [
    { label: "Perfect Passes", where: { skill: "Reception", evaluation_code: "#" }, description: "Receptions that left every option open.", aliases: ["perfect pass", "perfect passes", "threes"] },
    { label: "Positive Passes", where: { skill: "Reception", evaluation_code: ["#", "+"] }, description: "Receptions that kept the team in system.", aliases: ["positive passes", "good passes"] },
    { label: "Reception Errors", where: { skill: "Reception", evaluation_code: "=" }, description: "Receptions that gave up the point.", aliases: ["reception error", "passing errors", "shanks"] },
    { label: "Overpasses", where: { skill: "Reception", evaluation_code: "/" }, description: "Receptions that crossed the net.", aliases: ["overpass", "overpasses"] },
    { label: "Reception Attempts", where: { skill: "Reception" }, description: "Every serve received.", aliases: ["receptions", "passes", "reception attempts"] },
  ],
  Set: [
    { label: "Perfect Sets", where: { skill: "Set", evaluation_code: "#" }, description: "Sets that gave the hitter a clean swing.", aliases: ["perfect sets", "good sets"] },
    { label: "Setting Errors", where: { skill: "Set", evaluation_code: "=" }, description: "Setting mistakes that cost the point.", aliases: ["setting errors", "ball handling errors"] },
    { label: "Set Attempts", where: { skill: "Set" }, description: "Every set.", aliases: ["sets attempted", "set attempts"] },
  ],
  Dig: [
    { label: "Digs", where: { skill: "Dig" }, description: "Every dig of an opponent attack.", aliases: ["dig", "digs"] },
    { label: "Perfect Digs", where: { skill: "Dig", evaluation_code: "#" }, description: "Digs that came up perfectly playable.", aliases: ["perfect digs", "clean digs"] },
    { label: "Dig Errors", where: { skill: "Dig", evaluation_code: "=" }, description: "Digs that could not be played.", aliases: ["dig errors"] },
  ],
  Block: [
    { label: "Kill Blocks", where: { skill: "Block", evaluation_code: "#" }, description: "Blocks that scored the point.", aliases: ["kill block", "kill blocks", "stuff blocks", "blocks"] },
    { label: "Block Errors", where: { skill: "Block", evaluation_code: "=" }, description: "Blocking mistakes that gave up the point.", aliases: ["block errors"] },
    { label: "Block Touches", where: { skill: "Block" }, description: "Every block touch.", aliases: ["block touches", "touches"] },
  ],
  Freeball: [
    { label: "Freeballs", where: { skill: "Freeball" }, description: "Freeballs played.", aliases: ["freeball", "freeballs"] },
  ],
};

type Derived = { branch: string; label: string; expr: string; description: string; aliases: string[] };

const DERIVED: Derived[] = [
  { branch: "Attack", label: "Kills Per Set", expr: "[Kills]/[Sets Played]", description: "Kills divided by sets played.", aliases: ["kills per set", "k/s"] },
  { branch: "Attack", label: "Hitting Efficiency", expr: "([Kills]-[Attack Errors])/[Attack Attempts]", description: "Kills minus errors over total attempts.", aliases: ["hitting efficiency", "hitting percentage", "attack efficiency"] },
  { branch: "Attack", label: "Kill Rate", expr: "[Kills]/[Attack Attempts]", description: "Share of swings that were kills.", aliases: ["kill rate", "kill percentage"] },
  { branch: "Serve", label: "Aces Per Set", expr: "[Aces]/[Sets Played]", description: "Aces divided by sets played.", aliases: ["aces per set", "a/s"] },
  { branch: "Serve", label: "Service Error Rate", expr: "[Service Errors]/[Serve Attempts]", description: "Share of serves missed.", aliases: ["service error rate", "serve error percentage"] },
  { branch: "Receive", label: "Positive Pass Rate", expr: "[Positive Passes]/[Reception Attempts]", description: "Share of receptions that kept the team in system.", aliases: ["positive pass rate", "passing percentage"] },
  { branch: "Receive", label: "Reception Error Rate", expr: "[Reception Errors]/[Reception Attempts]", description: "Share of receptions that gave up the point.", aliases: ["reception error rate"] },
  { branch: "Dig", label: "Digs Per Set", expr: "[Digs]/[Sets Played]", description: "Digs divided by sets played.", aliases: ["digs per set", "d/s"] },
  { branch: "Block", label: "Blocks Per Set", expr: "[Kill Blocks]/[Sets Played]", description: "Kill blocks divided by sets played.", aliases: ["blocks per set", "b/s"] },
  { branch: "Sets", label: "Points Per Set", expr: "([Kills]+[Aces]+[Kill Blocks])/[Sets Played]", description: "Kills, aces and kill blocks per set played.", aliases: ["points per set", "pts/s"] },
];

/**
 * Seeds the play-by-play knowledge tree, validating every leaf against the LIVE schema
 * so a primitive these files never recorded is skipped instead of shipped broken.
 */
export function buildDvwSeedTree(schema: SourceSchema): GraphTree {
  const now = new Date().toISOString();
  const rootId = newId("root");
  const graph: GraphTree = {
    nodes: {
      [rootId]: {
        node_id: rootId,
        label: ROOT_LABEL,
        aliases: [],
        authored_by: "system:root",
        created_at: now,
      },
    },
    edges: [],
    root_id: rootId,
  };

  const branchIds = new Map<string, string>();
  for (const branch of DVW_BRANCHES) {
    const id = newId("branch");
    branchIds.set(branch, id);
    graph.nodes[id] = {
      node_id: id,
      label: branch,
      aliases: [branch.toLowerCase()],
      authored_by: AUTHOR,
      created_at: now,
    };
    graph.edges.push({ parent: rootId, child: id });
  }

  const add = (branch: string, node: GraphNode) => {
    graph.nodes[node.node_id] = node;
    graph.edges.push({ parent: branchIds.get(branch)!, child: node.node_id });
  };

  for (const [branch, primitives] of Object.entries(PRIMITIVES)) {
    for (const p of primitives) {
      const spec = { kind: "event", where: p.where, aggregate: "count" } as const;
      if (validateSpec(spec, graph, { grain: "event", schema })) continue; // not in these files
      add(branch, {
        node_id: newId("leaf"),
        label: p.label,
        kind: "event",
        where: p.where,
        aggregate: "count",
        human_description: p.description,
        aliases: p.aliases,
        authored_by: AUTHOR,
        created_at: now,
      });
    }
  }

  // Sets played is a measure read from roster participation, never counted from actions.
  add("Sets", {
    node_id: newId("leaf"),
    label: "Sets Played",
    kind: "measure",
    measure: "sets_played",
    human_description: "Sets the player participated in, from the match roster.",
    aliases: ["sets", "sets played"],
    authored_by: AUTHOR,
    created_at: now,
  });

  // Derived formulas come last, so their tokens validate against leaves that now exist.
  for (const d of DERIVED) {
    const spec = {
      kind: "formula",
      formula_expr: d.expr,
      human_description: d.description,
    } as const;
    if (validateSpec(spec, graph, { grain: "event", schema })) continue;
    add(d.branch, {
      node_id: newId("leaf"),
      label: d.label,
      kind: "formula",
      formula_expr: d.expr,
      human_description: d.description,
      aliases: d.aliases,
      authored_by: AUTHOR,
      created_at: now,
    });
  }

  return graph;
}
