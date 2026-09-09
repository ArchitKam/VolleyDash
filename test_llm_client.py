"""
test_llm_client.py
====================
Tests for recruiting_llm.call_llm()'s transport/fallback logic: the
model-fallback chain (try LLM_MODEL, fall back to LLM_FALLBACK_MODELS on
a rate limit or a deprecated/missing model), and that every Groq-side
failure converts to LLMUnavailableError rather than crashing. All HTTP
is mocked -- no real Groq calls, no API key spend.

Run with: pytest test_llm_client.py -v
"""

from unittest.mock import patch

import httpx
import openai
import pytest

import recruiting_llm as rl


@pytest.fixture(autouse=True)
def _fake_api_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "fake-key-for-tests")


def _rate_limit_error() -> openai.RateLimitError:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(429, request=request, json={"error": {"message": "rate limited"}})
    return openai.RateLimitError("Rate limit exceeded", response=response, body=None)


def _not_found_error() -> openai.NotFoundError:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(404, request=request, json={"error": {"message": "model not found"}})
    return openai.NotFoundError("Model not found", response=response, body=None)


def _auth_error() -> openai.AuthenticationError:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(401, request=request, json={"error": {"message": "bad key"}})
    return openai.AuthenticationError("Invalid API key", response=response, body=None)


def _fake_response(text: str):
    class _Message:
        content = text

    class _Choice:
        message = _Message()

    class _Response:
        choices = [_Choice()]

    return _Response()


def test_missing_api_key_raises_without_any_http_call(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with patch("openai.resources.chat.completions.Completions.create") as mock_create:
        with pytest.raises(rl.LLMUnavailableError, match="GROQ_API_KEY"):
            rl.call_llm("sys", "user")
        mock_create.assert_not_called()


def test_success_on_primary_model_never_tries_fallback():
    calls = []

    def fake_create(self, **kwargs):
        calls.append(kwargs["model"])
        return _fake_response("primary worked")

    with patch("openai.resources.chat.completions.Completions.create", fake_create):
        result = rl.call_llm("sys", "user")
    assert calls == [rl.LLM_MODEL]
    assert result == "primary worked"


def test_rate_limit_on_primary_falls_back_to_next_model():
    calls = []

    def fake_create(self, **kwargs):
        calls.append(kwargs["model"])
        if kwargs["model"] == rl.LLM_MODEL:
            raise _rate_limit_error()
        return _fake_response("fallback worked")

    with patch("openai.resources.chat.completions.Completions.create", fake_create):
        result = rl.call_llm("sys", "user")
    assert calls == [rl.LLM_MODEL] + rl.LLM_FALLBACK_MODELS[:1]
    assert result == "fallback worked"


def test_not_found_on_primary_also_falls_back():
    """A deprecated/renamed model (what actually happened once already
    with llama-3.1-8b-instant) should trigger the same fallback path as
    a rate limit, not a hard failure."""
    calls = []

    def fake_create(self, **kwargs):
        calls.append(kwargs["model"])
        if kwargs["model"] == rl.LLM_MODEL:
            raise _not_found_error()
        return _fake_response("fallback worked")

    with patch("openai.resources.chat.completions.Completions.create", fake_create):
        result = rl.call_llm("sys", "user")
    assert calls == [rl.LLM_MODEL] + rl.LLM_FALLBACK_MODELS[:1]
    assert result == "fallback worked"


def test_rate_limited_on_every_model_raises_llm_unavailable():
    def always_rate_limited(self, **kwargs):
        raise _rate_limit_error()

    with patch("openai.resources.chat.completions.Completions.create", always_rate_limited):
        with pytest.raises(rl.LLMUnavailableError, match="every configured model"):
            rl.call_llm("sys", "user")


def test_non_rate_limit_error_fails_immediately_without_trying_fallback():
    """A connection/auth/5xx failure affects Groq (or the account) as a
    whole -- retrying a different model wouldn't help, so this should
    NOT burn through the fallback chain, unlike rate-limit/not-found."""
    calls = []

    def fake_create(self, **kwargs):
        calls.append(kwargs["model"])
        raise _auth_error()

    with patch("openai.resources.chat.completions.Completions.create", fake_create):
        with pytest.raises(rl.LLMUnavailableError, match="AuthenticationError"):
            rl.call_llm("sys", "user")
    assert calls == [rl.LLM_MODEL]  # never tried the fallback


def test_strips_markdown_code_fences():
    def fake_create(self, **kwargs):
        return _fake_response('```json\n{"a": 1}\n```')

    with patch("openai.resources.chat.completions.Completions.create", fake_create):
        result = rl.call_llm("sys", "user")
    assert result == '{"a": 1}'


# ── set_hint (the Set drill-down's router field) ───────────────

class TestSetHint:
    """set_hint becomes an identity value on the Set axis, so a hint that
    does not normalise to exactly the label the source emits would filter
    to nothing -- which reads as "she did not play" rather than "the
    filter did not match". Normalisation is therefore the router's job,
    not the query layer's."""

    @staticmethod
    def _tree():
        from recruiting_tree import KnowledgeTree
        tree = KnowledgeTree()
        tree.add_root()
        return tree

    def test_recognised_forms_all_normalise_to_one_label(self):
        from recruiting_llm import _repair_set_hint
        for raw in ("set 3", "Set 3", "the 3rd set", "3", 3, 3.0):
            assert _repair_set_hint(raw, []) == "Set 3", raw

    def test_absent_or_unparseable_is_no_filter(self):
        from recruiting_llm import _repair_set_hint
        for raw in (None, "", "   ", "the deciding set", True, False):
            assert _repair_set_hint(raw, []) is None, raw

    def test_out_of_range_is_dropped_with_a_note_not_clamped(self):
        """"set 25" is the model echoing a score. Clamping it to set 5
        would answer a question nobody asked."""
        from recruiting_llm import _repair_set_hint
        notes = []
        assert _repair_set_hint("set 25", notes) is None
        assert notes and "at most 5 sets" in notes[0]
        assert _repair_set_hint(0, []) is None

    def test_repair_carries_set_hint_onto_the_action(self):
        from recruiting_llm import _validate_and_repair
        repaired = _validate_and_repair(
            {"actions": [{"metric_of_interest": "Kills", "skill_group": None,
                          "player": "Sloan", "game_hint": None, "set_hint": "set 2",
                          "title": "t", "pipeline": []}]},
            self._tree(),
        )
        assert repaired["actions"][0]["set_hint"] == "Set 2"

    def test_different_sets_are_different_questions_and_do_not_merge(self):
        from recruiting_llm import merge_same_shape_actions
        actions = [
            {"metric_of_interest": "Kills", "skill_group": None, "player": "Sloan",
             "game_hint": None, "set_hint": "Set 1", "is_committed": True, "pipeline": []},
            {"metric_of_interest": "Kills", "skill_group": None, "player": "Azana",
             "game_hint": None, "set_hint": "Set 2", "is_committed": True, "pipeline": []},
        ]
        assert len(merge_same_shape_actions(actions)) == 2

    def test_same_set_still_merges_two_players(self):
        from recruiting_llm import merge_same_shape_actions
        actions = [
            {"metric_of_interest": "Kills", "skill_group": None, "player": "Sloan",
             "game_hint": None, "set_hint": "Set 1", "is_committed": True, "pipeline": []},
            {"metric_of_interest": "Kills", "skill_group": None, "player": "Azana",
             "game_hint": None, "set_hint": "Set 1", "is_committed": True, "pipeline": []},
        ]
        merged = merge_same_shape_actions(actions)
        assert len(merged) == 1 and merged[0]["player"] == ["Sloan", "Azana"]

    def test_the_field_is_only_offered_when_the_source_can_answer_it(self):
        """A Huddle CSV row is a whole match; advertising a set filter
        against it invites a question with no honest answer."""
        from recruiting_llm import _build_router_system_prompt
        tree = self._tree()
        assert "set_hint" not in _build_router_system_prompt([], tree, ["Game A"])
        assert "set_hint" in _build_router_system_prompt([], tree, ["Game A"], supports_sets=True)
