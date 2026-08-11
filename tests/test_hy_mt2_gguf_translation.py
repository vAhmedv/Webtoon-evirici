"""Focused unit/integration tests for the Hy-MT2 production translator."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from core.config import TranslatorConfig, load_config
from core.translation.protection import (
    detect_named_terms_in_items,
    protect_source_text,
    restore_protected_translation,
)
from providers.translation import (
    HyMT2GGUFTranslationProvider,
    TranslationInput,
    TranslationItem,
    get_configured_translation_provider,
    get_translation_provider,
)
from providers.translation.hy_mt2_gguf_translation import (
    DEFAULT_HY_MT2_MODEL_PATH,
    DEFAULT_HY_MT2_SERVER_URL,
    clean_hy_mt2_output,
    render_hy_mt2_prompt,
)


class TestHyMT2ProductionProvider(unittest.TestCase):
    def _ready_provider(self) -> HyMT2GGUFTranslationProvider:
        provider = HyMT2GGUFTranslationProvider(managed=False)
        provider._loaded = True
        self.health = patch.object(provider, "_check_health", return_value=True)
        self.identity = patch.object(
            provider, "_server_identity_compatible", return_value=True
        )
        self.health.start()
        self.identity.start()
        self.addCleanup(self.health.stop)
        self.addCleanup(self.identity.stop)
        return provider

    def test_provider_defaults_and_dedicated_port(self):
        provider = HyMT2GGUFTranslationProvider(managed=False)
        self.assertEqual(provider.model_path, DEFAULT_HY_MT2_MODEL_PATH)
        self.assertEqual(provider.server_url, DEFAULT_HY_MT2_SERVER_URL)
        self.assertTrue(provider.server_url.endswith(":8085"))

    def test_native_prompt_is_exact_and_preserves_sentinel(self):
        prepared = "Activate __WTTERM0001__."
        self.assertEqual(
            render_hy_mt2_prompt(prepared),
            "<|startoftext|>Translate the following text into Turkish. Note that "
            "you should only output the translated result without any additional "
            "explanation:\nActivate __WTTERM0001__.<|extra_0|>",
        )

    def test_completion_payload_and_response_extraction(self):
        provider = HyMT2GGUFTranslationProvider(managed=False)
        response = MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = json.dumps(
            {
                "content": "Merhaba.<|eos|>",
                "tokens_evaluated": 21,
                "tokens_predicted": 4,
                "timings": {"predicted_ms": 100},
            }
        ).encode("utf-8")
        with patch("urllib.request.urlopen", return_value=response) as urlopen:
            raw, prompt_n, predicted_n, seconds = provider._query_chat_completion("Hello.")
        request = urlopen.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(request.full_url, DEFAULT_HY_MT2_SERVER_URL + "/completion")
        self.assertEqual(payload["temperature"], 0.0)
        self.assertEqual(payload["top_k"], 1)
        self.assertEqual(payload["seed"], 0)
        self.assertEqual(payload["n_predict"], 128)
        self.assertNotIn("messages", payload)
        self.assertEqual((raw, prompt_n, predicted_n, seconds), ("Merhaba.<|eos|>", 21, 4, 0.1))
        self.assertEqual(clean_hy_mt2_output(raw, payload["prompt"]), "Merhaba.")

    def test_server_error_retries_once_and_propagates_review(self):
        provider = self._ready_provider()
        with patch.object(provider, "_query_chat_completion", side_effect=OSError("down")):
            output = provider.translate(
                TranslationInput(items=[TranslationItem(1, "Hello there.", 1)])
            )
        result = output.results[0]
        self.assertIsNone(result.translation)
        self.assertEqual(result.validation_warnings, ["translation_server_error"])
        self.assertTrue(result.requires_review)
        self.assertEqual(provider.metrics.generation_call_count, 2)

    def test_bracketed_named_ability_is_generically_protected_and_restored(self):
        item = TranslationItem(1, "I came to test [FORGE MASTER].", 1)
        terms = detect_named_terms_in_items([item])
        self.assertEqual(terms, {"FORGE MASTER"})
        protected, mapping = protect_source_text(item.source, {}, terms)
        self.assertRegex(protected, r"\[__WTTERM\d{4}__\]")
        sentinel = next(iter(mapping))
        restored = restore_protected_translation(
            f"{sentinel}'ı denemeye geldim.", mapping
        )
        self.assertEqual(restored, "FORGE MASTER'i denemeye geldim.")

    def test_production_path_normalizes_then_protects_and_restores(self):
        provider = self._ready_provider()
        captured = {}

        def request(prepared: str, label: str):
            captured["prepared"] = prepared
            sentinel = prepared[prepared.index("__WTTERM") : prepared.index("__", prepared.index("__WTTERM") + 2) + 2]
            return f"[{sentinel}]'ı denemek için ormana geldim.", f"[{sentinel}]'ı denemek için ormana geldim.", False

        with patch.object(provider, "_request_translation", side_effect=request):
            output = provider.translate(
                TranslationInput(
                    items=[TranslationItem(1, "I ONLY CAME TO TEST [FORGE MASTER]...", 1)]
                )
            )
        trace = provider.last_traces[0]
        self.assertEqual(trace.normalized_input, "I only came to test [FORGE MASTER]...")
        self.assertIn("[__WTTERM", captured["prepared"])
        self.assertIn("[FORGE MASTER]", output.results[0].translation)
        self.assertNotIn("__WTTERM", output.results[0].translation)

    def test_explanation_wrapper_is_rejected_by_shared_guard(self):
        provider = self._ready_provider()
        with patch.object(
            provider,
            "_request_translation",
            return_value=(
                "The most accurate translation is Merhaba.",
                "The most accurate translation is Merhaba.",
                False,
            ),
        ):
            result = provider.translate(
                TranslationInput(items=[TranslationItem(1, "Hello.", 1)])
            ).results[0]
        self.assertIsNone(result.translation)
        self.assertIn("chatbot_or_explanation_output", result.validation_warnings)
        self.assertTrue(result.requires_review)

    def test_multiple_terms_and_turkish_suffixes_restore(self):
        protected, mapping = protect_source_text(
            "ALICE gave BLOOD AXE to BOB.",
            {"BLOOD AXE": "Kan Baltası"},
            {"ALICE", "BOB"},
            {"ALICE", "BOB"},
        )
        sentinels = list(mapping)
        self.assertEqual(len(sentinels), 3)
        translated = f"{sentinels[0]}'ın {sentinels[1]}'a verdiği {sentinels[2]}'dır."
        restored = restore_protected_translation(translated, mapping)
        self.assertNotIn("__WTTERM", restored)
        self.assertTrue(all(meta.target_base in restored for meta in mapping.values()))

    def test_incompatible_existing_server_is_not_terminated(self):
        provider = HyMT2GGUFTranslationProvider(managed=True)
        with patch.object(provider, "_check_health", return_value=True), patch.object(
            provider, "_wait_for_compatible_identity", return_value=False
        ):
            with self.assertRaisesRegex(RuntimeError, "incompatible"):
                provider.load()
        self.assertFalse(provider._owned_process)

    def test_identity_check_tolerates_transient_metadata_readiness(self):
        provider = HyMT2GGUFTranslationProvider(managed=False)
        with patch.object(
            provider, "_server_identity_compatible", side_effect=[False, False, True]
        ), patch("time.sleep"):
            self.assertTrue(provider._wait_for_compatible_identity(timeout_sec=1))

    def test_server_identity_falls_back_to_native_props(self):
        provider = HyMT2GGUFTranslationProvider(managed=False)
        props = MagicMock()
        props.__enter__.return_value = props
        props.read.return_value = json.dumps(
            {"model_path": DEFAULT_HY_MT2_MODEL_PATH}
        ).encode("utf-8")
        with patch("urllib.request.urlopen", return_value=props):
            self.assertTrue(provider._server_identity_compatible())

    def test_factory_and_translator_config_selection(self):
        explicit = get_translation_provider("hy_mt2_gguf", managed=False)
        self.assertIsInstance(explicit, HyMT2GGUFTranslationProvider)
        config = TranslatorConfig(
            enabled=True,
            provider="hy_mt2_gguf",
            model_path="X.gguf",
            llama_executable="llama-server.exe",
            server_url="http://127.0.0.1:9000",
        )
        configured = get_configured_translation_provider(config)
        self.assertEqual(configured.model_path, "X.gguf")
        self.assertEqual(configured.executable_path, "llama-server.exe")
        self.assertEqual(configured.server_url, "http://127.0.0.1:9000")

    def test_config_loader_accepts_server_url(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(
                "translator:\n  provider: hy_mt2_gguf\n  server_url: http://127.0.0.1:8085\n",
                encoding="utf-8",
            )
            config = load_config(path)
        self.assertEqual(config.translator.provider, "hy_mt2_gguf")
        self.assertEqual(config.translator.server_url, DEFAULT_HY_MT2_SERVER_URL)


if __name__ == "__main__":
    unittest.main()
