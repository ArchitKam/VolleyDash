"""
test_data_store.py
====================
Tests for recruiting_data_store.py -- the one module that talks to the
private VolleyData repo. All GitHub HTTP calls are mocked, so this runs
without real credentials, real GitHub, or a real Streamlit runtime.

Run with: pytest test_data_store.py -v
"""

import base64
import json

import pandas as pd
import pytest

import recruiting_data_store as store
from recruiting_tree import KnowledgeTree, NodeKind, seed_recruiting_tree


@pytest.fixture(autouse=True)
def _clear_caches_and_secrets(monkeypatch):
    """@st.cache_data is process-global -- clear it before every test so
    one test's mocked response can't leak into the next. Also isolate
    secrets from whatever's actually on this machine: a real
    .streamlit/secrets.toml (e.g. for local manual testing against the
    real VolleyData repo) would otherwise make st.secrets see real
    values regardless of monkeypatch.delenv, since that only clears
    environment variables -- st.secrets reads the TOML file directly,
    a separate source _get_secret checks FIRST. Replacing it with a
    plain empty dict here makes every test start from a clean "nothing
    configured" state; individual tests opt into "configured" via
    monkeypatch.setenv, which _get_secret still falls back to."""
    store.get_games.clear()
    store.load_game_df.clear()
    monkeypatch.setattr(store.st, "secrets", {})
    for key in ("GITHUB_DATA_REPO", "GITHUB_DATA_TOKEN"):
        monkeypatch.delenv(key, raising=False)
    yield


class _FakeResponse:
    def __init__(self, status_code, json_body=None, text=""):
        self.status_code = status_code
        self._json_body = json_body
        self.text = text or json.dumps(json_body or {})

    def json(self):
        return self._json_body


def _b64(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


# ──────────────────────────────────────────────────────────────
# SECRETS
# ──────────────────────────────────────────────────────────────

def test_require_secret_missing_raises():
    with pytest.raises(RuntimeError, match="GITHUB_DATA_REPO"):
        store._require_secret("GITHUB_DATA_REPO")


def test_get_secret_env_fallback(monkeypatch):
    monkeypatch.setenv("GITHUB_DATA_REPO", "ArchitKam/VolleyData")
    assert store._get_secret("GITHUB_DATA_REPO") == "ArchitKam/VolleyData"


# ──────────────────────────────────────────────────────────────
# save_committed_tree -- fail loudly on ANY failure
# ──────────────────────────────────────────────────────────────

def test_save_committed_tree_missing_secrets_raises_before_any_http_call(monkeypatch):
    calls = []
    monkeypatch.setattr(store.requests, "get", lambda *a, **k: calls.append("get"))
    monkeypatch.setattr(store.requests, "put", lambda *a, **k: calls.append("put"))
    tree, _ = seed_recruiting_tree()

    with pytest.raises(RuntimeError):
        store.save_committed_tree(tree)
    assert calls == []


def test_save_committed_tree_sha_fetch_then_put_sequence(monkeypatch):
    """The core contract: GET the current sha first, then PUT with that
    exact sha -- required by the Contents API to update (not create) an
    existing file."""
    monkeypatch.setenv("GITHUB_DATA_REPO", "ArchitKam/VolleyData")
    monkeypatch.setenv("GITHUB_DATA_TOKEN", "fake-token")
    tree, _ = seed_recruiting_tree()

    captured_put = {}

    def fake_get(url, headers=None, timeout=None):
        assert url == f"{store._API_ROOT}/ArchitKam/VolleyData/contents/{store.TREE_JSON_PATH}"
        assert headers["Authorization"] == "Bearer fake-token"
        return _FakeResponse(200, {"sha": "abc123"})

    def fake_put(url, headers=None, json=None, timeout=None):
        captured_put["url"] = url
        captured_put["payload"] = json
        return _FakeResponse(200)

    monkeypatch.setattr(store.requests, "get", fake_get)
    monkeypatch.setattr(store.requests, "put", fake_put)

    store.save_committed_tree(tree)

    assert captured_put["payload"]["sha"] == "abc123"
    decoded = base64.b64decode(captured_put["payload"]["content"]).decode("utf-8")
    assert json.loads(decoded) == store.tree_to_json_dict(tree)


def test_save_committed_tree_get_failure_raises(monkeypatch):
    """A GET that FAILS still aborts the save. 404 is excluded on
    purpose -- see the test below -- so this uses a server error, which
    is the case the guard is actually for: never PUT over a file whose
    current state could not be read."""
    monkeypatch.setenv("GITHUB_DATA_REPO", "ArchitKam/VolleyData")
    monkeypatch.setenv("GITHUB_DATA_TOKEN", "fake-token")
    tree, _ = seed_recruiting_tree()

    monkeypatch.setattr(store.requests, "get", lambda *a, **k: _FakeResponse(500, text="Server Error"))
    put_calls = []
    monkeypatch.setattr(store.requests, "put", lambda *a, **k: put_calls.append(1))

    with pytest.raises(RuntimeError, match="sha"):
        store.save_committed_tree(tree)
    assert not put_calls, "a failed sha lookup must not be followed by a write"


def test_a_missing_file_is_created_rather_than_treated_as_a_failure(monkeypatch):
    """Previously a 404 here raised, which meant a knowledge base that
    did not exist yet could never be saved -- so seed-then-save could
    never bootstrap, for either workspace."""
    monkeypatch.setenv("GITHUB_DATA_REPO", "ArchitKam/VolleyData")
    monkeypatch.setenv("GITHUB_DATA_TOKEN", "fake-token")
    tree, _ = seed_recruiting_tree()

    monkeypatch.setattr(store.requests, "get", lambda *a, **k: _FakeResponse(404, text="Not Found"))
    captured = {}

    def _fake_put(url, headers=None, json=None, timeout=None):
        captured["payload"] = json
        return _FakeResponse(201, {})

    monkeypatch.setattr(store.requests, "put", _fake_put)
    store.save_committed_tree(tree)
    assert "sha" not in captured["payload"]


def test_save_committed_tree_put_failure_raises(monkeypatch):
    monkeypatch.setenv("GITHUB_DATA_REPO", "ArchitKam/VolleyData")
    monkeypatch.setenv("GITHUB_DATA_TOKEN", "fake-token")
    tree, _ = seed_recruiting_tree()

    monkeypatch.setattr(store.requests, "get", lambda *a, **k: _FakeResponse(200, {"sha": "abc123"}))
    monkeypatch.setattr(store.requests, "put", lambda *a, **k: _FakeResponse(409, text="Conflict"))

    with pytest.raises(RuntimeError, match="push failed"):
        store.save_committed_tree(tree)


# ──────────────────────────────────────────────────────────────
# load_committed_tree -- return None on ANY failure
# ──────────────────────────────────────────────────────────────

def test_load_committed_tree_returns_none_on_missing_secrets():
    assert store.load_committed_tree() is None


def test_load_committed_tree_returns_none_on_network_error(monkeypatch):
    monkeypatch.setenv("GITHUB_DATA_REPO", "ArchitKam/VolleyData")
    monkeypatch.setenv("GITHUB_DATA_TOKEN", "fake-token")

    def raise_connection_error(*a, **k):
        raise store.requests.exceptions.ConnectionError("no route to host")

    monkeypatch.setattr(store.requests, "get", raise_connection_error)
    assert store.load_committed_tree() is None


def test_load_committed_tree_returns_none_on_404(monkeypatch):
    monkeypatch.setenv("GITHUB_DATA_REPO", "ArchitKam/VolleyData")
    monkeypatch.setenv("GITHUB_DATA_TOKEN", "fake-token")
    monkeypatch.setattr(store.requests, "get", lambda *a, **k: _FakeResponse(404, text="Not Found"))
    assert store.load_committed_tree() is None


def test_load_committed_tree_returns_none_on_malformed_json(monkeypatch):
    monkeypatch.setenv("GITHUB_DATA_REPO", "ArchitKam/VolleyData")
    monkeypatch.setenv("GITHUB_DATA_TOKEN", "fake-token")
    monkeypatch.setattr(
        store.requests, "get",
        lambda *a, **k: _FakeResponse(200, {"content": _b64("not valid json")}),
    )
    assert store.load_committed_tree() is None


def test_load_committed_tree_round_trips_a_real_seeded_tree(monkeypatch):
    monkeypatch.setenv("GITHUB_DATA_REPO", "ArchitKam/VolleyData")
    monkeypatch.setenv("GITHUB_DATA_TOKEN", "fake-token")

    original, _ = seed_recruiting_tree()
    payload = json.dumps(store.tree_to_json_dict(original))
    monkeypatch.setattr(
        store.requests, "get",
        lambda *a, **k: _FakeResponse(200, {"content": _b64(payload)}),
    )

    loaded = store.load_committed_tree()
    assert loaded is not None
    original_leaves = sorted(n.label for n in original.committed.values() if n.kind == NodeKind.LEAF)
    loaded_leaves = sorted(n.label for n in loaded.committed.values() if n.kind == NodeKind.LEAF)
    assert original_leaves == loaded_leaves


# ──────────────────────────────────────────────────────────────
# get_games / load_game_df
# ──────────────────────────────────────────────────────────────

def test_get_games_lists_only_csv_files_and_parses_opponent(monkeypatch):
    monkeypatch.setenv("GITHUB_DATA_REPO", "ArchitKam/VolleyData")
    monkeypatch.setenv("GITHUB_DATA_TOKEN", "fake-token")

    listing = [
        {"name": "01 Nat vs Vegas Aces - Stats.csv", "path": "recruiting_csvs/01 Nat vs Vegas Aces - Stats.csv", "type": "file"},
        {"name": "02 Nat vs Mavs 816 - Stats.csv", "path": "recruiting_csvs/02 Nat vs Mavs 816 - Stats.csv", "type": "file"},
        {"name": "README.md", "path": "recruiting_csvs/README.md", "type": "file"},
        {"name": "subdir", "path": "recruiting_csvs/subdir", "type": "dir"},
    ]
    monkeypatch.setattr(store.requests, "get", lambda *a, **k: _FakeResponse(200, listing))

    games = store.get_games()
    assert [g.filename for g in games] == [
        "01 Nat vs Vegas Aces - Stats.csv", "02 Nat vs Mavs 816 - Stats.csv",
    ]
    assert games[0].opponent == "Vegas Aces"
    assert games[1].opponent == "Mavs 816"


def test_get_games_failure_raises(monkeypatch):
    monkeypatch.setenv("GITHUB_DATA_REPO", "ArchitKam/VolleyData")
    monkeypatch.setenv("GITHUB_DATA_TOKEN", "fake-token")
    monkeypatch.setattr(store.requests, "get", lambda *a, **k: _FakeResponse(404, text="Not Found"))

    with pytest.raises(RuntimeError, match="recruiting_csvs"):
        store.get_games()


def test_load_game_df_fetches_and_parses_csv_content(monkeypatch):
    monkeypatch.setenv("GITHUB_DATA_REPO", "ArchitKam/VolleyData")
    monkeypatch.setenv("GITHUB_DATA_TOKEN", "fake-token")

    csv_text = "Name,Attack K,Attack E\n#7 Sloan T.,12,3\n"
    monkeypatch.setattr(
        store.requests, "get",
        lambda *a, **k: _FakeResponse(200, {"content": _b64(csv_text)}),
    )

    df = store.load_game_df("recruiting_csvs/01 Nat vs Vegas Aces - Stats.csv")
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["Name", "Attack K", "Attack E"]
    assert df.iloc[0]["Name"] == "#7 Sloan T."
    assert df.iloc[0]["Attack K"] == 12


# ──────────────────────────────────────────────────────────────
# Files at or above 1 MB
# ──────────────────────────────────────────────────────────────

def test_a_file_too_large_to_inline_is_refetched_as_raw(monkeypatch):
    """GitHub's Contents API inlines file bytes as base64 only BELOW
    1 MB. At or above it the request still returns 200, with
    "encoding": "none" and an empty content field -- a success response
    containing no file. Before this fallback that decoded to b"" and
    surfaced as an empty CSV rather than an error."""
    monkeypatch.setenv("GITHUB_DATA_REPO", "ArchitKam/VolleyData")
    monkeypatch.setenv("GITHUB_DATA_TOKEN", "fake-token")

    big = b"Name,Attack K\n#7 Sloan T.,12\n"
    seen = []

    class _Raw:
        status_code = 200
        content = big
        text = ""

    def _fake_get(url, headers=None, timeout=None):
        seen.append(headers["Accept"])
        if headers["Accept"] == "application/vnd.github.v3.raw":
            return _Raw()
        return _FakeResponse(200, {"content": "", "encoding": "none"})

    monkeypatch.setattr(store.requests, "get", _fake_get)

    assert store._fetch_file_bytes("ArchitKam/VolleyData", "fake-token", "big.csv") == big
    assert seen == ["application/vnd.github+json", "application/vnd.github.v3.raw"], (
        "the raw refetch must happen only after the inline attempt reports it could not"
    )


def test_a_small_file_is_never_refetched(monkeypatch):
    """The fallback must cost nothing on the path that already worked."""
    monkeypatch.setenv("GITHUB_DATA_REPO", "ArchitKam/VolleyData")
    monkeypatch.setenv("GITHUB_DATA_TOKEN", "fake-token")

    calls = []

    def _fake_get(url, headers=None, timeout=None):
        calls.append(headers["Accept"])
        return _FakeResponse(200, {"content": _b64("hello"), "encoding": "base64"})

    monkeypatch.setattr(store.requests, "get", _fake_get)
    assert store._fetch_file_bytes("r", "t", "small.csv") == b"hello"
    assert calls == ["application/vnd.github+json"]


def test_saving_a_tree_that_does_not_exist_yet_creates_it(monkeypatch):
    """A second knowledge base always starts from nothing. The Contents
    API creates a file when the PUT carries NO sha, so a 404 on the sha
    lookup is the create case rather than a failure -- treating it as an
    error made the DVW tree impossible to save the first time."""
    monkeypatch.setenv("GITHUB_DATA_REPO", "ArchitKam/VolleyData")
    monkeypatch.setenv("GITHUB_DATA_TOKEN", "fake-token")

    captured = {}
    monkeypatch.setattr(store.requests, "get", lambda *a, **k: _FakeResponse(404, {}))

    def _fake_put(url, headers=None, json=None, timeout=None):
        captured["payload"] = json
        return _FakeResponse(201, {})

    monkeypatch.setattr(store.requests, "put", _fake_put)

    tree = KnowledgeTree()
    tree.add_root()
    store.save_committed_tree(tree, path="volley_kb_data.json", serializer=lambda t: {"nodes": []})

    assert "sha" not in captured["payload"], "a create must not send a sha"
    assert captured["payload"]["message"].startswith("Create")


def test_saving_an_existing_tree_still_sends_its_sha(monkeypatch):
    monkeypatch.setenv("GITHUB_DATA_REPO", "ArchitKam/VolleyData")
    monkeypatch.setenv("GITHUB_DATA_TOKEN", "fake-token")

    captured = {}
    monkeypatch.setattr(store.requests, "get", lambda *a, **k: _FakeResponse(200, {"sha": "abc123"}))

    def _fake_put(url, headers=None, json=None, timeout=None):
        captured["payload"] = json
        return _FakeResponse(200, {})

    monkeypatch.setattr(store.requests, "put", _fake_put)

    tree = KnowledgeTree()
    tree.add_root()
    store.save_committed_tree(tree, path="volley_kb_data.json", serializer=lambda t: {"nodes": []})

    assert captured["payload"]["sha"] == "abc123"
    assert captured["payload"]["message"].startswith("Update")
