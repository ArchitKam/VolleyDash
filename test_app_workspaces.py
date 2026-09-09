"""
test_app_workspaces.py
=======================
Boots the real Streamlit app in each workspace and asserts the things
that must differ between them -- and the things that must not.

The unit tests prove the engine is right. This proves the WIRING is:
that the dropdown reaches the source, that the source's declared axes
reach the controls, and that switching does not leave one world's state
sitting in front of the other's data.

Skips when the corpus or the data-repo secrets are absent, following the
test_live_* convention in test_integration.py.
"""

import os

import pytest

import recruiting_data_store as data_store

APP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")


def _secrets_available() -> bool:
    try:
        return bool(data_store._get_secret("GITHUB_DATA_REPO")
                    and data_store._get_secret("GITHUB_DATA_TOKEN"))
    except Exception:
        return False


def _corpus_available() -> bool:
    return data_store.local_dvw_dir() is not None


requires_secrets = pytest.mark.skipif(not _secrets_available(), reason="no data-repo secrets here")
requires_corpus = pytest.mark.skipif(not _corpus_available(), reason="no local .dvw corpus here")


def _run(workspace_key: str, **state):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP, default_timeout=600)
    at.session_state["workspace_key"] = workspace_key
    for key, value in state.items():
        at.session_state[key] = value
    at.run()
    return at


def _labels(widgets):
    return [w.label for w in widgets]


def _has(widgets, prefix: str) -> bool:
    return any(label.startswith(prefix) for label in _labels(widgets))


# ── both worlds boot ───────────────────────────────────────────

@requires_secrets
def test_recruiting_boots_without_exceptions():
    at = _run("recruiting")
    assert not at.exception, [str(e.value) for e in at.exception]
    assert at.title[0].value == "🏐 Volleyball Knowledge Base"


@requires_secrets
@requires_corpus
def test_player_analysis_boots_without_exceptions():
    at = _run("player_analysis")
    assert not at.exception, [str(e.value) for e in at.exception]


# ── the capability difference is visible in the UI ─────────────

@requires_secrets
def test_recruiting_offers_no_set_control():
    """A Huddle row is a whole match. Offering a set picker there would
    imply an answer exists; the control is absent, not disabled."""
    at = _run("recruiting")
    assert not _has(at.multiselect, "Sets"), _labels(at.multiselect)
    assert _has(at.multiselect, "Games")


@requires_secrets
@requires_corpus
def test_player_analysis_offers_a_set_control_covering_a_five_set_match():
    at = _run("player_analysis")
    sets = next(m for m in at.multiselect if m.label.startswith("Sets"))
    assert sets.options == ["Set 1", "Set 2", "Set 3", "Set 4", "Set 5"]
    assert sets.value == [], "all sets together is the default"


@requires_secrets
def test_only_the_event_world_offers_a_team_picker():
    """A .dvw records BOTH teams; a Huddle export is already one team's,
    so there is nothing to choose."""
    assert not _has(_run("recruiting").selectbox, "Team")


@requires_secrets
@requires_corpus
def test_the_team_picker_defaults_to_the_team_the_corpus_is_about():
    """Without this, MatchInfo.opponent falls back to the visiting team
    and an away match gets labelled with the scouted team's own name."""
    at = _run("player_analysis")
    team = next(s for s in at.selectbox if s.label == "Team")
    assert team.value == "University of Maryland"

    games = next(m for m in at.multiselect if m.label.startswith("Games"))
    assert games.options, "expected matches to load"
    assert not any("Maryland" in label for label in games.options), (
        "a game must be labelled by the OPPONENT, never by the team of interest"
    )


# ── switching does not leak state between worlds ───────────────

def test_the_switch_callback_clears_every_scoped_key():
    """
    Checked against a plain dict rather than a live session.

    workspace.clear_scoped_state takes the session mapping as an
    argument precisely so this test needs no Streamlit runtime and no
    `import app` -- importing app.py EXECUTES it, which in a bare
    context leaves an st.form block open and makes every later AppTest
    fail with an unrelated error.
    """
    import workspace

    state = dict({
        "qa_action_results": [{"stale": True}],
        "qa_last_decomposition": {"actions": []},
        "qa_player_filter": ["#7 Sloan T."],
        "qa_encodings": {"x": "Player"},
        "qa_enc_0_x": "Player",
        "qa_playerfilter_#7 Sloan T.": True,
        "qa_games_multiselect": ["Game A"],
        "selected_node_id": "node-1",
        "editing_node_id": "node-1",
    })

    workspace.clear_scoped_state(state)

    assert state["qa_action_results"] == []
    assert state["qa_last_decomposition"] is None
    assert state["qa_player_filter"] == []
    assert state["qa_encodings"] == {}
    assert state["selected_node_id"] is None
    assert state["editing_node_id"] is None
    assert "qa_enc_0_x" not in state, "chart widget state is keyed per result set"
    assert "qa_playerfilter_#7 Sloan T." not in state
    assert "qa_games_multiselect" not in state, (
        "the games picker is keyed by label and would carry one world's opponents "
        "into the other's list"
    )


# ── the two knowledge bases stay distinct ──────────────────────

@requires_secrets
@requires_corpus
def test_each_workspace_has_its_own_knowledge_base():
    import workspace

    recruiting = workspace.build_recruiting()
    events = workspace.build_player_analysis()

    assert recruiting.tree_path != events.tree_path
    assert recruiting.tree is not events.tree

    def labels(ws):
        from recruiting_tree import NodeKind
        return {n.label for n in ws.tree.committed.values() if n.kind == NodeKind.LEAF}

    event_only = labels(events) - labels(recruiting)
    assert event_only, "the event tree must contain metrics the CSV tree cannot express"
    assert events.supports_sets and not recruiting.supports_sets


# ── degrading instead of dying ─────────────────────────────────

def test_recruiting_is_always_available():
    """It depends on none of the .dvw stack, so no .dvw dependency
    problem may ever remove it."""
    import workspace

    assert workspace.RECRUITING in workspace.available_keys()


def test_a_dvw_import_failure_hides_that_workspace_instead_of_killing_the_app(monkeypatch):
    """
    The failure that took down a live demo: dvw_patch raised at import,
    app.py imported source_dvw at module scope, and the WHOLE app died
    -- recruiting included, which needs none of it.

    Availability is now asked for rather than assumed, so an unsupported
    pandas removes one entry from a dropdown.
    """
    import builtins

    import workspace

    real_import = builtins.__import__

    def _fail_on_source_dvw(name, *args, **kwargs):
        if name == "source_dvw":
            raise RuntimeError("pandas 3.0.3 is too new for pydatavolley")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(__import__("sys").modules, "source_dvw", raising=False)
    monkeypatch.setattr(builtins, "__import__", _fail_on_source_dvw)

    reason = workspace.dvw_unavailable_reason()
    assert reason and "too new" in reason
    assert workspace.available_keys() == [workspace.RECRUITING]


def test_when_dvw_works_both_workspaces_are_offered():
    import workspace

    if workspace.dvw_unavailable_reason() is not None:
        pytest.skip("the .dvw stack is unavailable in this environment")
    assert workspace.available_keys() == [workspace.RECRUITING, workspace.PLAYER_ANALYSIS]


@requires_secrets
@requires_corpus
def test_the_knowledge_base_tab_names_which_knowledge_base_it_is():
    """
    The two trees deliberately share branch NAMES so the router's
    synonyms resolve a category question in either world -- which makes
    them look nearly identical on screen. The tab must therefore SAY
    which one it is, or "am I editing the right one" is unanswerable
    without opening a metric and reading its definition.
    """
    seen = {}
    for key in ("recruiting", "player_analysis"):
        at = _run(key)
        header = next(h.value for h in at.header if "Metrics Tree" in h.value)
        caption = next(c.value for c in at.caption if "committed metrics" in c.value)
        seen[key] = (header, caption)

    assert "Recruiting" in seen["recruiting"][0]
    assert "Player analysis" in seen["player_analysis"][0]
    assert seen["recruiting"][0] != seen["player_analysis"][0]

    # The file is named too, so the claim is checkable against the repo.
    assert "recruiting_kb_data.json" in seen["recruiting"][1]
    assert "volley_kb_data.json" in seen["player_analysis"][1]


@requires_secrets
@requires_corpus
def test_the_two_trees_really_are_different_trees():
    """Guards the thing the label asserts: if these ever became the same
    object, the label would be a comforting lie."""
    import workspace
    from recruiting_tree import NodeKind

    recruiting = workspace.build_recruiting()
    events = workspace.build_player_analysis()

    def leaves(ws):
        return {n.label for n in ws.tree.committed.values() if n.kind == NodeKind.LEAF}

    assert recruiting.tree is not events.tree
    assert leaves(recruiting) != leaves(events)
    assert "Freeballs" in leaves(events), "an event-only metric"
    assert "Freeballs" not in leaves(recruiting)
