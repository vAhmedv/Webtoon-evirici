"""Unit tests for QwenGGUFTranslationProvider (llama-server backend adapter).

Verifies serialization, HTTP communication, health checks, error handling,
JSON response parsing, fidelity flags, truncation recovery, and CandidateStore non-mutation.
"""
import json
import unittest
from unittest.mock import MagicMock, patch

from core.translation.profile_discovery import CandidateStore
from providers.translation import (
    QwenGGUFTranslationProvider,
    TranslationInput,
    TranslationItem,
    get_translation_provider,
)


class TestQwenGGUFTranslationProvider(unittest.TestCase):

    def test_factory_function(self):
        provider = get_translation_provider(backend="gguf")
        self.assertIsInstance(provider, QwenGGUFTranslationProvider)
        self.assertEqual(provider.name, "Qwen3.5-9B-GGUF-Translation")

    def test_provider_initialization_defaults(self):
        provider = QwenGGUFTranslationProvider()
        self.assertFalse(provider._loaded)
        self.assertEqual(provider.metrics.translation_model, "Qwen3.5-9B-Q5_K_M-GGUF")

    @patch("urllib.request.urlopen")
    def test_health_check_healthy(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        provider = QwenGGUFTranslationProvider()
        self.assertTrue(provider._check_health())

    @patch("urllib.request.urlopen")
    def test_health_check_unhealthy(self, mock_urlopen):
        mock_urlopen.side_effect = Exception("Connection refused")
        provider = QwenGGUFTranslationProvider()
        self.assertFalse(provider._check_health())

    @patch.object(QwenGGUFTranslationProvider, "_check_health", return_value=True)
    def test_load_with_preexisting_server(self, mock_health):
        provider = QwenGGUFTranslationProvider(auto_start_server=False)
        provider.load()
        self.assertTrue(provider._loaded)
        self.assertFalse(provider._owns_server)
        provider.unload()
        self.assertFalse(provider._loaded)

    def test_load_disabled_autostart_raises_runtime_error(self):
        with patch.object(QwenGGUFTranslationProvider, "_check_health", return_value=False):
            provider = QwenGGUFTranslationProvider(auto_start_server=False)
            with self.assertRaises(RuntimeError) as cm:
                provider.load()
            self.assertIn("llama-server unavailable", str(cm.exception))

    @patch.object(QwenGGUFTranslationProvider, "_check_health", return_value=True)
    @patch("urllib.request.urlopen")
    def test_system_role_and_deterministic_sampling(self, mock_urlopen, mock_health):
        mock_completion = MagicMock()
        mock_completion.status = 200
        response_body = {
            "choices": [{"message": {"content": json.dumps({"translations": [{"id": 1, "source": "Hi", "translation": "Merhaba"}]})}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}
        }
        mock_completion.read.return_value = json.dumps(response_body).encode("utf-8")
        mock_completion.__enter__.return_value = mock_completion
        mock_urlopen.return_value = mock_completion

        provider = QwenGGUFTranslationProvider(auto_start_server=False)
        provider.load()

        inp = TranslationInput(items=[TranslationItem(region_id=1, source="Hi", reading_order=1)])
        provider.translate(inp)

        # Inspect request sent to urlopen
        req = mock_urlopen.call_args[0][0]
        payload = json.loads(req.data.decode("utf-8"))

        self.assertEqual(len(payload["messages"]), 2)
        self.assertEqual(payload["messages"][0]["role"], "system")
        self.assertEqual(payload["messages"][1]["role"], "user")
        self.assertEqual(payload["temperature"], 0.0)

    @patch.object(QwenGGUFTranslationProvider, "_check_health", return_value=True)
    @patch("urllib.request.urlopen")
    def test_translate_structured_json_response(self, mock_urlopen, mock_health):
        mock_completion = MagicMock()
        mock_completion.status = 200
        response_body = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps({
                            "translations": [
                                {
                                    "id": 1,
                                    "source": "Relax, kid.",
                                    "translation": "Sakin ol, çocuk.",
                                    "term_usages": [],
                                    "fidelity_flags": []
                                },
                                {
                                    "id": 2,
                                    "source": "My name is Luo Tian.",
                                    "translation": "Benim adım Luo Tian.",
                                    "term_usages": [],
                                    "fidelity_flags": []
                                }
                            ]
                        })
                    }
                }
            ],
            "usage": {"prompt_tokens": 80, "completion_tokens": 30}
        }
        mock_completion.read.return_value = json.dumps(response_body).encode("utf-8")
        mock_completion.__enter__.return_value = mock_completion
        mock_urlopen.return_value = mock_completion

        provider = QwenGGUFTranslationProvider(auto_start_server=False)
        provider.load()

        inp = TranslationInput(
            items=[
                TranslationItem(region_id=1, source="Relax, kid.", reading_order=1),
                TranslationItem(region_id=2, source="My name is Luo Tian.", reading_order=2),
            ]
        )

        out = provider.translate(inp)
        self.assertEqual(len(out.results), 2)
        self.assertEqual(out.results[0].translation, "Sakin ol, çocuk.")
        self.assertEqual(out.results[1].translation, "Benim adım Luo Tian.")
        self.assertFalse(out.results[0].requires_review)
        self.assertEqual(provider.metrics.input_token_count, 80)
        self.assertEqual(provider.metrics.generated_token_count, 30)

    @patch.object(QwenGGUFTranslationProvider, "_check_health", return_value=True)
    @patch("urllib.request.urlopen")
    def test_gguf_truncation_recovery(self, mock_urlopen, mock_health):
        # 1st response: truncated (missing item 2), 950 tokens
        resp1_body = {
            "choices": [{"message": {"content": json.dumps({"translations": [{"id": 1, "source": "Item 1", "translation": "Çeviri 1"}]})}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 950}
        }
        # 2nd response (sub_a): item 1
        resp2_body = {
            "choices": [{"message": {"content": json.dumps({"translations": [{"id": 1, "source": "Item 1", "translation": "Çeviri 1"}]})}}],
            "usage": {"prompt_tokens": 50, "completion_tokens": 20}
        }
        # 3rd response (sub_b): item 2
        resp3_body = {
            "choices": [{"message": {"content": json.dumps({"translations": [{"id": 2, "source": "Item 2", "translation": "Çeviri 2"}]})}}],
            "usage": {"prompt_tokens": 50, "completion_tokens": 20}
        }

        def make_mock(body):
            m = MagicMock()
            m.status = 200
            m.read.return_value = json.dumps(body).encode("utf-8")
            m.__enter__.return_value = m
            return m

        mock_urlopen.side_effect = [make_mock(resp1_body), make_mock(resp2_body), make_mock(resp3_body)]

        provider = QwenGGUFTranslationProvider(auto_start_server=False)
        provider.load()

        inp = TranslationInput(
            items=[
                TranslationItem(region_id=1, source="Item 1", reading_order=1),
                TranslationItem(region_id=2, source="Item 2", reading_order=2),
            ]
        )

        out = provider.translate(inp)
        self.assertEqual(len(out.results), 2)
        self.assertEqual(out.results[0].translation, "Çeviri 1")
        self.assertEqual(out.results[1].translation, "Çeviri 2")
        self.assertEqual(provider.metrics.retries, 1)

    @patch.object(QwenGGUFTranslationProvider, "_check_health", return_value=True)
    @patch("urllib.request.urlopen")
    def test_candidate_store_is_not_mutated_during_translate(self, mock_urlopen, mock_health):
        mock_completion = MagicMock()
        mock_completion.status = 200
        response_body = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps({
                            "translations": [
                                {
                                    "id": 10,
                                    "source": "I'm a secret realm guide.",
                                    "translation": "Gizli diyar rehberiyim.",
                                    "term_usages": [],
                                    "fidelity_flags": []
                                }
                            ]
                        })
                    }
                }
            ],
            "usage": {"prompt_tokens": 45, "completion_tokens": 15}
        }
        mock_completion.read.return_value = json.dumps(response_body).encode("utf-8")
        mock_completion.__enter__.return_value = mock_completion
        mock_urlopen.return_value = mock_completion

        provider = QwenGGUFTranslationProvider(auto_start_server=False)
        provider.load()

        store = CandidateStore(series_id="test_series")

        inp = TranslationInput(
            items=[TranslationItem(region_id=10, source="I'm a secret realm guide.", reading_order=1)],
            candidate_store=store,
        )

        store_before_json = json.dumps(store.to_dict())
        provider.translate(inp)
        store_after_json = json.dumps(store.to_dict())

        self.assertEqual(store_before_json, store_after_json, "CandidateStore was mutated inside provider!")


if __name__ == "__main__":
    unittest.main()
