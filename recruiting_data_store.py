"""
recruiting_data_store.py
=========================
The ONE module that talks to the private data store -- a separate
private GitHub repo (VolleyData) holding the real match CSVs and the
committed knowledge tree's JSON. The public code repo (VolleyDash,
what Streamlit Community Cloud actually deploys) never contains this
data itself; it only reaches it over the network, authenticated with a
token scoped to VolleyData alone.

Everything here goes through GitHub's Contents API
(GET/PUT /repos/{repo}/contents/{path}) against two secrets:
  GITHUB_DATA_REPO  -- "owner/repo", e.g. "ArchitKam/VolleyData"
  GITHUB_DATA_TOKEN -- a PAT scoped only to that repo's Contents

That boundary is deliberate: if this repo is ever swapped for a real
database (e.g. Supabase), only THIS file changes -- app.py just calls
save_committed_tree()/load_committed_tree()/get_games()/load_game_df()
and never knows or cares where the bytes actually live.

Contract (unchanged from the app's existing session-init logic):
  save_committed_tree(tree) -- raises RuntimeError on ANY failure
    (missing secret, network error, stale sha, bad response). Loud on
    purpose: a silent failure here means an edit LOOKS merged in the
    UI but was never durably saved.
  load_committed_tree() -- returns None on ANY failure (missing
    secret, network error, malformed JSON, file not found). Callers
    treat None as "no committed tree yet, seed a fresh one" -- never
    a crash.
"""

import base64
import copy
import io
import json
import os
import re
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd
import requests
import streamlit as st

from recruiting_tree import KnowledgeTree, Node, NodeKind, make_column_spec, make_formula_spec

_API_ROOT = "https://api.github.com/repos"
TREE_JSON_PATH = "recruiting_kb_data.json"
CSV_DIR_PATH = "recruiting_csvs"

_VS_SPLIT_RE = re.compile(r"\s+vs\.?\s+", re.IGNORECASE)
_STATS_SUFFIX_RE = re.compile(r"\s*-\s*stats\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class GameInfo:
    path: str          # repo-relative path within VolleyData, e.g. "recruiting_csvs/foo.csv"
    filename: str
    opponent: str


def _parse_opponent(filename: str) -> str:
    stem = os.path.splitext(filename)[0]
    parts = _VS_SPLIT_RE.split(stem, maxsplit=1)
    tail = parts[1] if len(parts) == 2 else stem
    return _STATS_SUFFIX_RE.sub("", tail).strip()


def _get_secret(name: str) -> Optional[str]:
    """st.secrets first, os.environ fallback for local testing -- wrapped
    in try/except since st.secrets raises (rather than acting like an
    empty dict) when no secrets.toml exists at all."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.environ.get(name)


def _require_secret(name: str) -> str:
    value = _get_secret(name)
    if not value:
        raise RuntimeError(f"{name} isn't set (as an environment variable or Streamlit secret).")
    return value


def _headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}


def _contents_url(repo: str, path: str) -> str:
    return f"{_API_ROOT}/{repo}/contents/{path}"


# ──────────────────────────────────────────────────────────────
# TREE PERSISTENCE
# ──────────────────────────────────────────────────────────────

def _spec_to_dict(spec) -> dict:
    payload = spec.payload
    kind = payload.get("kind")
    entry = {"kind": kind}
    if kind == "column":
        entry["column_ref"] = payload.get("column_ref")
    elif kind == "formula":
        entry["formula_expr"] = payload.get("formula_expr")
        entry["human_description"] = payload.get("human_description", "")
    return entry


def _spec_from_dict(entry: dict):
    if entry.get("kind") == "column":
        return make_column_spec(entry["column_ref"])
    if entry.get("kind") == "formula":
        return make_formula_spec(entry["formula_expr"], entry.get("human_description", ""))
    raise ValueError(f"Unknown stored spec kind: {entry.get('kind')!r}")


def tree_to_json_dict(tree: KnowledgeTree) -> dict:
    root = tree.committed.get(tree.root_id)
    groups: dict = {}
    if root:
        for branch_id in root.children:
            branch = tree.committed.get(branch_id)
            if not branch:
                continue
            metrics = {}
            for leaf_id in branch.children:
                leaf = tree.committed.get(leaf_id)
                if not leaf or not leaf.spec:
                    continue
                entry = _spec_to_dict(leaf.spec)
                entry.update({
                    "node_id": leaf.node_id,
                    "aliases": leaf.aliases,
                    "authored_by": leaf.authored_by,
                    "created_at": leaf.created_at,
                })
                metrics[leaf.label] = entry
            groups[branch.label] = {"branch_node_id": branch.node_id, "metrics": metrics}
    return {"recruiting": groups}


def _tree_from_json_dict(data: dict) -> KnowledgeTree:
    tree = KnowledgeTree()
    tree.add_root()
    root = tree.committed[tree.root_id]

    for branch_label, branch_data in data.get("recruiting", {}).items():
        branch_id = branch_data.get("branch_node_id") or tree._new_id(branch_label)
        branch_node = Node(node_id=branch_id, label=branch_label,
                            kind=NodeKind.BRANCH, parent_id=tree.root_id)
        tree.committed[branch_id] = branch_node
        root.children.append(branch_id)

        for leaf_label, entry in branch_data.get("metrics", {}).items():
            spec = _spec_from_dict(entry)
            leaf_id = entry.get("node_id") or tree._new_id(leaf_label)
            leaf_node = Node(
                node_id=leaf_id, label=leaf_label, kind=NodeKind.LEAF,
                parent_id=branch_id, spec=spec,
                aliases=entry.get("aliases", []),
                authored_by=entry.get("authored_by", "system"),
                created_at=entry.get("created_at", ""),
            )
            tree.committed[leaf_id] = leaf_node
            branch_node.children.append(leaf_id)

    tree.staging = copy.deepcopy(tree.committed)
    return tree


def save_committed_tree(tree: KnowledgeTree) -> None:
    """
    Two-step Contents API update: GET the file's current sha, then PUT
    the new content with that sha (required to update rather than
    create). Raises RuntimeError on any failure in either step -- there
    is no local fallback, so a failure here means this edit will not
    survive a redeploy/restart even though it's already merged into the
    in-memory tree for this session.
    """
    repo = _require_secret("GITHUB_DATA_REPO")
    token = _require_secret("GITHUB_DATA_TOKEN")
    url = _contents_url(repo, TREE_JSON_PATH)
    headers = _headers(token)

    get_response = requests.get(url, headers=headers, timeout=15)
    if get_response.status_code != 200:
        raise RuntimeError(
            f"Couldn't fetch current sha for {TREE_JSON_PATH} in {repo} "
            f"({get_response.status_code}): {get_response.text[:300]}"
        )
    sha = get_response.json()["sha"]

    content = json.dumps(tree_to_json_dict(tree), indent=2)
    put_payload = {
        "message": "Update recruiting_kb_data.json via dashboard",
        "content": base64.b64encode(content.encode("utf-8")).decode("ascii"),
        "sha": sha,
    }
    put_response = requests.put(url, headers=headers, json=put_payload, timeout=15)
    if put_response.status_code not in (200, 201):
        raise RuntimeError(
            f"GitHub push failed for {TREE_JSON_PATH} in {repo} "
            f"({put_response.status_code}): {put_response.text[:300]}"
        )


def load_committed_tree() -> Optional[KnowledgeTree]:
    """Returns None on ANY failure (missing secret, network error,
    file not found, malformed JSON/tree shape) -- callers must fall back
    to seeding a fresh tree in every one of those cases, never crash."""
    try:
        repo = _require_secret("GITHUB_DATA_REPO")
        token = _require_secret("GITHUB_DATA_TOKEN")
        response = requests.get(_contents_url(repo, TREE_JSON_PATH), headers=_headers(token), timeout=15)
        if response.status_code != 200:
            return None
        content = base64.b64decode(response.json()["content"]).decode("utf-8")
        data = json.loads(content)
        return _tree_from_json_dict(data)
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────
# CSV FETCHING -- lists and fetches recruiting_csvs/ from the same
# private repo. Cached the same way the app's local-file reads used to
# be, so a Streamlit rerun doesn't refetch over the network every time.
# ──────────────────────────────────────────────────────────────

@st.cache_data
def get_games() -> List[GameInfo]:
    repo = _require_secret("GITHUB_DATA_REPO")
    token = _require_secret("GITHUB_DATA_TOKEN")
    response = requests.get(_contents_url(repo, CSV_DIR_PATH), headers=_headers(token), timeout=15)
    if response.status_code != 200:
        raise RuntimeError(
            f"Couldn't list {CSV_DIR_PATH} in {repo} ({response.status_code}): {response.text[:300]}"
        )
    entries = response.json()
    games = [
        GameInfo(path=entry["path"], filename=entry["name"], opponent=_parse_opponent(entry["name"]))
        for entry in entries
        if entry.get("type") == "file" and entry["name"].lower().endswith(".csv")
    ]
    return sorted(games, key=lambda g: g.filename)


@st.cache_data
def load_game_df(path: str) -> pd.DataFrame:
    repo = _require_secret("GITHUB_DATA_REPO")
    token = _require_secret("GITHUB_DATA_TOKEN")
    response = requests.get(_contents_url(repo, path), headers=_headers(token), timeout=15)
    if response.status_code != 200:
        raise RuntimeError(f"Couldn't fetch {path} in {repo} ({response.status_code}): {response.text[:300]}")
    content = base64.b64decode(response.json()["content"]).decode("utf-8")
    return pd.read_csv(io.StringIO(content))
