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
import glob
import io
import json
import os
import re
import tempfile
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


def _raw_headers(token: str) -> dict:
    """The Contents API inlines file bytes as base64 ONLY below 1 MB.
    At or above that it answers with "encoding": "none" and an empty
    content field -- a 200 that contains no file. This Accept asks for
    the bytes themselves, which has no size limit."""
    return {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github.v3.raw"}


def _fetch_file_bytes(repo: str, token: str, path: str, timeout: int = 30) -> bytes:
    """
    One file out of the private repo.

    Keeps the inline base64 path for small files -- unchanged, because
    that is what every existing caller and test exercises -- and falls
    back to a raw refetch only when GitHub says it could not inline the
    content. Detecting the case rather than always using raw means the
    working path stays byte-identical and the fix costs one extra
    request only on the files that were previously broken.
    """
    url = _contents_url(repo, path)
    response = requests.get(url, headers=_headers(token), timeout=timeout)
    if response.status_code != 200:
        raise RuntimeError(
            f"Couldn't fetch {path} in {repo} ({response.status_code}): {response.text[:300]}"
        )

    payload = response.json()
    content = payload.get("content")
    # Default "base64" when the field is absent: that is what the API
    # sends for every inlined file, and assuming it here keeps this
    # exactly as faithful to the previous one-line implementation as it
    # can be. An over-1MB file arrives with empty content (and
    # "encoding": "none"), which is the only case that falls through.
    if content and payload.get("encoding", "base64") == "base64":
        return base64.b64decode(content)

    raw = requests.get(url, headers=_raw_headers(token), timeout=timeout)
    if raw.status_code != 200:
        raise RuntimeError(
            f"Couldn't fetch {path} in {repo} as raw ({raw.status_code}): {raw.text[:300]}"
        )
    return raw.content


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


def save_committed_tree(tree: KnowledgeTree, path: str = None,
                         serializer=None) -> None:
    """
    Two-step Contents API update: GET the file's current sha, then PUT
    the new content with that sha (required to update rather than
    create). Raises RuntimeError on any failure in either step -- there
    is no local fallback, so a failure here means this edit will not
    survive a redeploy/restart even though it's already merged into the
    in-memory tree for this session.
    """
    # `path`/`serializer` let a second knowledge base (the event-grain
    # one) persist to the same private repo through the same two-step
    # update, without this module learning what an event spec is.
    path = path or TREE_JSON_PATH
    serializer = serializer or tree_to_json_dict

    repo = _require_secret("GITHUB_DATA_REPO")
    token = _require_secret("GITHUB_DATA_TOKEN")
    url = _contents_url(repo, path)
    headers = _headers(token)

    get_response = requests.get(url, headers=headers, timeout=15)
    if get_response.status_code == 404:
        # First save of a knowledge base that does not exist yet. The
        # Contents API creates a file when the PUT carries NO sha, and
        # rejects it when the sha is wrong -- so "no sha" is the create
        # case, not a missing precondition. Distinguishing this from a
        # real failure matters because a second knowledge base always
        # starts here.
        sha = None
    elif get_response.status_code != 200:
        raise RuntimeError(
            f"Couldn't fetch current sha for {path} in {repo} "
            f"({get_response.status_code}): {get_response.text[:300]}"
        )
    else:
        sha = get_response.json()["sha"]

    content = json.dumps(serializer(tree), indent=2)
    put_payload = {
        "message": f"{'Update' if sha else 'Create'} {path} via dashboard",
        "content": base64.b64encode(content.encode("utf-8")).decode("ascii"),
    }
    if sha is not None:
        put_payload["sha"] = sha
    put_response = requests.put(url, headers=headers, json=put_payload, timeout=15)
    if put_response.status_code not in (200, 201):
        raise RuntimeError(
            f"GitHub push failed for {path} in {repo} "
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
    content = _fetch_file_bytes(repo, token, path, timeout=15).decode("utf-8")
    return pd.read_csv(io.StringIO(content))


# ──────────────────────────────────────────────────────────────
# DATAVOLLEY (.dvw) MATCH FILES
# ──────────────────────────────────────────────────────────────
# Same private repo, separate folder. The .dvw files have to live here
# rather than on local disk because Streamlit Community Cloud deploys
# the PUBLIC code repo and has no access to any local corpus -- a path
# like /fs/... exists only on the machine it was written on.
#
# pydatavolley reads a PATH, not bytes, so fetched files are written to
# a cache directory and their local paths returned. The cache is keyed
# by the repo-relative path, so a match is downloaded once per session.

DVW_DIR_PATH = "dvw"

#: Set this to a directory of .dvw files to skip the network entirely.
#: Intended for running on a machine that already has the corpus: same
#: Source, same results, no round trip.
DVW_LOCAL_DIR_ENV = "VOLLEY_DVW_DIR"


def _dvw_cache_dir() -> str:
    directory = os.path.join(tempfile.gettempdir(), "volleydash_dvw_cache")
    os.makedirs(directory, exist_ok=True)
    return directory


def local_dvw_dir() -> Optional[str]:
    """The local corpus directory, if one is configured AND actually has
    .dvw files in it. Returns None otherwise so callers fall through to
    the repo rather than failing on a stale environment variable."""
    directory = os.environ.get(DVW_LOCAL_DIR_ENV)
    if directory and os.path.isdir(directory) and glob.glob(os.path.join(directory, "*.dvw")):
        return directory
    return None


@st.cache_data(show_spinner=False)
def list_dvw_files() -> List[str]:
    """Repo-relative paths of every .dvw in the private repo's dvw/
    folder. Empty list when the folder does not exist yet -- that is a
    "no matches uploaded" state, not an error."""
    repo = _require_secret("GITHUB_DATA_REPO")
    token = _require_secret("GITHUB_DATA_TOKEN")
    response = requests.get(_contents_url(repo, DVW_DIR_PATH), headers=_headers(token), timeout=15)
    if response.status_code == 404:
        return []
    if response.status_code != 200:
        raise RuntimeError(
            f"Couldn't list {DVW_DIR_PATH} in {repo} ({response.status_code}): {response.text[:300]}"
        )
    return sorted(
        entry["path"] for entry in response.json()
        if entry.get("type") == "file" and entry["name"].lower().endswith(".dvw")
    )


@st.cache_data(show_spinner=False)
def fetch_dvw_file(path: str) -> str:
    """One .dvw from the repo, written to the cache; returns its local
    path. Cached on `path`, so re-asking within a session is free."""
    repo = _require_secret("GITHUB_DATA_REPO")
    token = _require_secret("GITHUB_DATA_TOKEN")
    payload = _fetch_file_bytes(repo, token, path)

    local_path = os.path.join(_dvw_cache_dir(), os.path.basename(path))
    with open(local_path, "wb") as handle:
        handle.write(payload)
    return local_path


def dvw_paths() -> List[str]:
    """
    Local paths to every available match file, whichever way they got
    here. The ONE function the app calls: whether the corpus came off
    disk or out of the private repo is exactly the kind of thing the
    rest of the system should not know.
    """
    directory = local_dvw_dir()
    if directory is not None:
        return sorted(glob.glob(os.path.join(directory, "*.dvw")))
    return [fetch_dvw_file(path) for path in list_dvw_files()]


@st.cache_data(show_spinner=False)
def load_json_file(path: str) -> Optional[dict]:
    """
    Any JSON file out of the private repo, or None if it isn't there.

    None for "absent" and an exception for "present but unreachable":
    a caller seeding a fresh knowledge base needs to know the difference
    between "nothing saved yet" and "saved, but I couldn't read it" --
    conflating them silently discards someone's committed work.
    """
    repo = _require_secret("GITHUB_DATA_REPO")
    token = _require_secret("GITHUB_DATA_TOKEN")
    response = requests.get(_contents_url(repo, path), headers=_headers(token), timeout=15)
    if response.status_code == 404:
        return None
    if response.status_code != 200:
        raise RuntimeError(
            f"Couldn't fetch {path} in {repo} ({response.status_code}): {response.text[:300]}"
        )
    payload = response.json()
    content = payload.get("content")
    if content and payload.get("encoding", "base64") == "base64":
        return json.loads(base64.b64decode(content).decode("utf-8"))
    return json.loads(_fetch_file_bytes(repo, token, path).decode("utf-8"))
