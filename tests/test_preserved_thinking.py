"""The preserved-thinking pair and the eval kwargs guard, at the route level.

``CR2-F-3`` ``--preserved-thinking`` puts ``enable_thinking=true``/``clear_thinking=false``
on every forwarded ``/v1/messages`` body a backend honors it for; a non-mapping
``chat_template_kwargs`` is replaced rather than skipped, and the flag-off arm writes nothing.
PR1-R5 narrows the guarantee: SGLang's endpoint discards the pair, vLLM's supports it. The
eval transform's ``chat_template_kwargs`` guard is the same replacement on the chat route.
"""

import pytest

from buster_ripper import backend, config
from buster_ripper.routes.chat import _transform_eval_response
from tests.relay_bodies import _CHAT, _MESSAGES, _sent_body
from tests.upstream_stub import StubUpstream


class TestEvalChatTemplateKwargsGuard:
    """PR1-R4-4: a non-mapping ``chat_template_kwargs`` is replaced, not skipped.

    The eval transform's ``setdefault`` returned the client's ``null``/list/scalar and
    the nested ``setdefault`` raised ``AttributeError``, which the transform's broad
    handler swallowed: the request went upstream with no pair and no budget.
    """

    @pytest.mark.parametrize("value", [None, [], "x"], ids=["null", "list", "scalar"])
    async def test_a_non_mapping_is_replaced_and_the_pair_goes_out(self, proxy, mocker, value):
        mocker.patch.object(config, "EVAL_MODE", True)
        mocker.patch.object(config, "EVAL_PROFILE", config.PROFILES["glm47"])
        mocker.patch.object(config, "EVAL_THINKING", True)
        mocker.patch.object(config, "EVAL_THINKING_BUDGET", 128)
        post = mocker.patch.object(proxy.server.app["session"], "post",
                                    return_value=StubUpstream())

        response = await proxy.post("/v1/chat/completions",
                                    json={**_CHAT, "chat_template_kwargs": value})

        assert response.status == 200, "the route 500ed on a caller-supplied field"
        sent = _sent_body(post.call_args)
        assert sent["chat_template_kwargs"] == {
            "enable_thinking": True, "thinking_budget": 128}

    async def test_a_mapping_from_the_client_is_merged_not_flattened(self, proxy, mocker):
        """Replacement is for non-mappings only: ``setdefault`` semantics on a real
        mapping are unchanged, including a client's own budget."""
        mocker.patch.object(config, "EVAL_MODE", True)
        mocker.patch.object(config, "EVAL_PROFILE", config.PROFILES["glm47"])
        mocker.patch.object(config, "EVAL_THINKING", True)
        mocker.patch.object(config, "EVAL_THINKING_BUDGET", 128)
        post = mocker.patch.object(proxy.server.app["session"], "post",
                                    return_value=StubUpstream())

        await proxy.post("/v1/chat/completions", json={
            **_CHAT, "chat_template_kwargs": {"thinking_budget": 64}})

        sent = _sent_body(post.call_args)
        assert sent["chat_template_kwargs"] == {
            "thinking_budget": 64, "enable_thinking": True}


class TestEvalResponseTransforms:
    """The eval transform chain's arms, pinned at the function boundary."""

    def test_eval_response_transforms_today(self):
        """``(content, fixed)`` per enabled step, and the fixed flag carried through."""
        profile = config.PROFILES["glm47"]

        msg = {"content": "<think>reason</think>answer"}
        assert _transform_eval_response(msg, msg["content"], profile, False) == ("answer", True)

        msg = {"content": "", "reasoning_content": "reason"}
        assert _transform_eval_response(msg, msg["content"], profile, False) == ("reason", True)

        msg = {"content": "```python\nx = 1\n```"}
        assert _transform_eval_response(msg, msg["content"], profile, False) == ("x = 1", True)

        assert _transform_eval_response({"content": "plain"}, "plain", profile, False) == ("plain", False)
        assert _transform_eval_response({"content": "plain"}, "plain", profile, True) == ("plain", True)


class TestPreservedThinkingGuard:
    """CR2-F-3: the flag's pair goes out on every forwarded body a backend honors it for."""

    @pytest.mark.parametrize("value", [
        pytest.param(None, id="null"),
        pytest.param([], id="empty-list"),
        pytest.param([{"enable_thinking": False}], id="list-of-mappings"),
        pytest.param("enable", id="string-scalar"),
        pytest.param(7, id="number-scalar"),
        pytest.param(True, id="boolean-scalar"),
    ])
    async def test_a_non_mapping_is_replaced_and_the_pair_goes_out(self, proxy, mocker, value):
        """The shipped launcher default, at the route level: pre-guard this is an
        ``AttributeError`` out of ``handle_messages`` — HTTP 500 and no upstream call."""
        mocker.patch.object(config, "PRESERVED_THINKING", True)
        post = mocker.patch.object(proxy.server.app["session"], "post",
                                   return_value=StubUpstream())

        response = await proxy.post("/v1/messages", json={**_MESSAGES,
                                                          "chat_template_kwargs": value})

        assert response.status == 200, "the route 500ed on a caller-supplied field"
        assert post.call_count == 1, "the request never reached the upstream"
        assert _sent_body(post.call_args)["chat_template_kwargs"] == {
            "enable_thinking": True, "clear_thinking": False}

    @pytest.mark.parametrize("value, expected", [
        pytest.param({"thinking_budget": 128},
                     {"thinking_budget": 128, "enable_thinking": True, "clear_thinking": False},
                     id="client-kwargs-kept"),
        pytest.param({"enable_thinking": False},
                     {"enable_thinking": False, "clear_thinking": False},
                     id="client-override-still-wins"),
    ])
    async def test_a_mapping_from_the_client_is_merged_not_flattened(self, proxy, mocker, value,
                                                                     expected):
        """The replacement is for non-mappings only: ``setdefault`` semantics on a real
        mapping are what they always were, including a client turning thinking off."""
        mocker.patch.object(config, "PRESERVED_THINKING", True)
        post = mocker.patch.object(proxy.server.app["session"], "post",
                                   return_value=StubUpstream())

        await proxy.post("/v1/messages", json={**_MESSAGES, "chat_template_kwargs": value})

        assert _sent_body(post.call_args)["chat_template_kwargs"] == expected

    @pytest.mark.parametrize("backend_type", ["vllm", "unknown"])
    async def test_an_absent_field_still_gets_the_pair(self, proxy, mocker, backend_type):
        """The guarantee the flag's own help text states, for the ordinary request, on both
        backends whose Anthropic endpoint honors the pair."""
        mocker.patch.object(config, "PRESERVED_THINKING", True)
        mocker.patch.object(backend, "get_backend_state", return_value=backend_type)
        post = mocker.patch.object(proxy.server.app["session"], "post",
                                   return_value=StubUpstream())

        await proxy.post("/v1/messages", json=_MESSAGES)

        assert _sent_body(post.call_args)["chat_template_kwargs"] == {
            "enable_thinking": True, "clear_thinking": False}

    @pytest.mark.parametrize("value", [
        pytest.param(None, id="null"),
        pytest.param("enable", id="scalar"),
        pytest.param({"thinking_budget": 1}, id="mapping"),
    ])
    async def test_the_flag_off_arm_forwards_the_body_untouched(self, proxy, mocker, value):
        """No cost on the off arm: with the flag off this route must not write the key,
        so a build that moves the injection outside the guard fails here."""
        mocker.patch.object(config, "PRESERVED_THINKING", False)
        post = mocker.patch.object(proxy.server.app["session"], "post",
                                   return_value=StubUpstream())

        await proxy.post("/v1/messages", json={**_MESSAGES, "chat_template_kwargs": value})

        assert _sent_body(post.call_args)["chat_template_kwargs"] == value


class TestPreservedThinkingSglangGate:
    """PR1-R5: SGLang v0.5.9's ``/v1/messages`` discards ``chat_template_kwargs``.

    The flag's pair was injected into that body anyway, advertising an effect the endpoint
    never applies. The gate skips SGLang only: a client-sent mapping still travels as sent,
    and the vLLM/unknown arms above keep their pair.
    """

    async def test_sglang_gets_no_injected_pair(self, proxy, mocker):
        mocker.patch.object(config, "PRESERVED_THINKING", True)
        mocker.patch.object(backend, "get_backend_state", return_value="sglang")
        post = mocker.patch.object(proxy.server.app["session"], "post",
                                   return_value=StubUpstream())

        response = await proxy.post("/v1/messages", json=_MESSAGES)

        assert response.status == 200
        assert "chat_template_kwargs" not in _sent_body(post.call_args)

    async def test_a_client_sent_mapping_is_forwarded_unchanged(self, proxy, mocker):
        mocker.patch.object(config, "PRESERVED_THINKING", True)
        mocker.patch.object(backend, "get_backend_state", return_value="sglang")
        value = {"enable_thinking": False}
        post = mocker.patch.object(proxy.server.app["session"], "post",
                                   return_value=StubUpstream())

        await proxy.post("/v1/messages",
                         json={**_MESSAGES, "chat_template_kwargs": value})

        assert _sent_body(post.call_args)["chat_template_kwargs"] == value
