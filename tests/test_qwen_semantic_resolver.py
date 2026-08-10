"""Managed llama.cpp backend tests for the text-only Qwen semantic resolver."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.translation.semantic_context import (
    SemanticContextRequest,
    parse_translation_risk_resolution,
    render_translation_risk_resolver_prompt,
    resolve_translation_risk_with_fallback,
)
from providers.translation.qwen_semantic_resolver import (
    DEFAULT_QWEN_SEMANTIC_MODEL_PATH,
    QwenSemanticResolverProvider,
)


def _request() -> SemanticContextRequest:
    return SemanticContextRequest(
        previous_context=("The panels keep swinging open.",),
        target_source="The lock can hold two panels.",
        next_context=("Use it before the storm arrives.",),
    )


def _valid_payload() -> dict:
    return {
        "rewrite_needed": True,
        "confidence": 0.95,
        "risk_types": ["lexical_sense"],
        "semantic_notes": [
            {
                "span": "hold",
                "resolved_meaning": "keep fixed in place",
                "evidence": "The nearby panels keep moving.",
            }
        ],
        "question_type": None,
        "tense_aspect": "modal capability",
        "referents": [],
        "clarified_target": "The lock can secure two panels.",
    }


def _chat_response(content: str) -> dict:
    return {
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {
            "prompt_tokens": 120,
            "completion_tokens": 54,
            "total_tokens": 174,
        },
        "timings": {
            "prompt_n": 120,
            "predicted_n": 54,
            "predicted_ms": 900.0,
            "predicted_per_second": 60.0,
        },
    }


def test_expected_model_identity_is_accepted() -> None:
    provider = QwenSemanticResolverProvider()

    assert provider._server_identity_matches(
        {"model_path": DEFAULT_QWEN_SEMANTIC_MODEL_PATH}
    )
    assert provider._server_identity_matches(
        {"model_path": Path(DEFAULT_QWEN_SEMANTIC_MODEL_PATH).name}
    )


def test_wrong_model_identity_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = QwenSemanticResolverProvider()
    monkeypatch.setattr(
        provider,
        "_probe_server",
        lambda: (True, {"model_path": r"C:\AI\Models\other.gguf"}),
    )

    with pytest.raises(RuntimeError, match="unexpected model/server"):
        provider.load()

    assert provider.owned_process is False
    assert provider.is_loaded is False


def test_expected_existing_server_is_reused(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = QwenSemanticResolverProvider()
    props = {
        "model_path": DEFAULT_QWEN_SEMANTIC_MODEL_PATH,
        "chat_template": "{% for message in messages %}qwen{% endfor %}",
        "chat_template_caps": {"supports_system_role": True},
    }
    monkeypatch.setattr(provider, "_probe_server", lambda: (True, props))

    provider.load()

    assert provider.is_loaded is True
    assert provider.owned_process is False
    assert provider.chat_template_sha256 is not None
    provider.unload()
    assert provider.port_closed_after_unload is None


def test_successful_json_chat_response_updates_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = QwenSemanticResolverProvider(
        prompt_renderer=render_translation_risk_resolver_prompt,
    )
    provider._loaded = True
    monkeypatch.setattr(
        provider,
        "_request_json",
        lambda *args, **kwargs: _chat_response(json.dumps(_valid_payload())),
    )
    monkeypatch.setattr(provider, "_sample_gpu_memory_gb", lambda: 0.0)

    raw = provider.resolve(_request())
    parsed = parse_translation_risk_resolution(raw)

    assert parsed.rewrite_needed is True
    assert parsed.clarified_target == "The lock can secure two panels."
    assert provider.metrics.resolver_calls == 1
    assert provider.metrics.resolver_failures == 0
    assert provider.metrics.input_token_count == 120
    assert provider.metrics.generated_token_count == 54
    assert provider.metrics.server_generation_seconds == pytest.approx(0.9)
    assert provider.metrics.tokens_per_second == pytest.approx(60.0)


def test_chat_payload_uses_native_template_and_deterministic_json_schema() -> None:
    provider = QwenSemanticResolverProvider(
        prompt_renderer=render_translation_risk_resolver_prompt,
    )
    payload = provider._chat_payload("resolver prompt")

    assert payload["messages"] == [{"role": "user", "content": "resolver prompt"}]
    assert payload["temperature"] == 0.0
    assert payload["seed"] == 0
    assert payload["stream"] is False
    assert payload["json_schema"]["additionalProperties"] is False
    assert "chat_template" not in payload


def test_malformed_json_response_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = QwenSemanticResolverProvider(
        prompt_renderer=render_translation_risk_resolver_prompt,
    )
    provider._loaded = True
    monkeypatch.setattr(
        provider,
        "_request_json",
        lambda *args, **kwargs: _chat_response("not-json"),
    )
    monkeypatch.setattr(provider, "_sample_gpu_memory_gb", lambda: 0.0)

    outcome = resolve_translation_risk_with_fallback(_request(), provider.resolve)

    assert outcome.malformed_json is True
    assert outcome.decision.selected_target == _request().target_source


def test_server_error_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = QwenSemanticResolverProvider(
        prompt_renderer=render_translation_risk_resolver_prompt,
    )
    provider._loaded = True

    def fail(*args, **kwargs):
        raise RuntimeError("server error")

    monkeypatch.setattr(provider, "_request_json", fail)
    outcome = resolve_translation_risk_with_fallback(_request(), provider.resolve)

    assert outcome.resolver_failed is True
    assert outcome.decision.selected_target == _request().target_source
    assert provider.metrics.resolver_failures == 1


def test_reasoning_contamination_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = QwenSemanticResolverProvider()
    response = _chat_response(json.dumps(_valid_payload()))
    response["choices"][0]["message"]["reasoning_content"] = "hidden reasoning"
    monkeypatch.setattr(
        provider,
        "_request_json",
        lambda *args, **kwargs: response,
    )

    with pytest.raises(RuntimeError, match="reasoning content contaminated"):
        provider._query_chat_completion("prompt")

    assert provider.metrics.reasoning_contamination_count == 1


def test_managed_load_and_unload_only_owned_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = QwenSemanticResolverProvider(startup_timeout=1.0)

    class FakeProcess:
        pid = 4321
        returncode = None

        def __init__(self) -> None:
            self.terminated = False
            self.killed = False

        def poll(self):
            return None

        def terminate(self) -> None:
            self.terminated = True

        def wait(self, timeout=None) -> int:
            self.returncode = 0
            return 0

        def kill(self) -> None:
            self.killed = True

    fake_process = FakeProcess()
    monkeypatch.setattr(provider, "_probe_server", lambda: (False, None))
    monkeypatch.setattr(provider, "_port_is_open", lambda: False)
    monkeypatch.setattr(
        provider,
        "_wait_for_ready",
        lambda: {
            "model_path": DEFAULT_QWEN_SEMANTIC_MODEL_PATH,
            "chat_template": "qwen-template",
        },
    )
    monkeypatch.setattr(provider, "_sample_gpu_memory_gb", lambda: 0.0)
    monkeypatch.setattr("os.path.isfile", lambda _: True)
    monkeypatch.setattr("subprocess.Popen", lambda *args, **kwargs: fake_process)

    provider.load()

    assert provider.owned_process is True
    assert provider.is_loaded is True
    assert "serve" in provider.last_server_command
    assert "--reasoning" in provider.last_server_command
    assert provider.last_server_command[
        provider.last_server_command.index("--reasoning") + 1
    ] == "off"
    assert provider.last_server_command[
        provider.last_server_command.index("--port") + 1
    ] == "8082"

    provider.unload()

    assert fake_process.terminated is True
    assert fake_process.killed is False
    assert provider.owned_process is False
    assert provider.is_loaded is False
    assert provider.port_closed_after_unload is True


def test_semantic_resolver_has_no_transformers_or_bitsandbytes_dependency() -> None:
    source = Path("providers/translation/qwen_semantic_resolver.py").read_text(
        encoding="utf-8"
    )

    assert "from transformers" not in source
    assert "import transformers" not in source
    assert "BitsAndBytesConfig" not in source
    assert "bitsandbytes" not in source.casefold()
    assert "import torch" not in source
