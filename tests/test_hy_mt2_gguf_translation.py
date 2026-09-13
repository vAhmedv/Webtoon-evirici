"""Focused unit/integration tests for the Hy-MT2 production translator."""
from __future__ import annotations

import json
import http.client
import logging
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from core.config import TranslatorConfig, load_config
from core.translation.protection import (
    detect_named_terms_in_items,
    find_dropped_source_tokens,
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
        self.assertEqual(provider.metrics.translation_model, "HY-MT2-7B-Q8_0.gguf")
        self.assertNotIn("Q4_K_M", provider.name)

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
        self.assertEqual(payload["n_predict"], 2048)
        self.assertNotIn("messages", payload)
        self.assertEqual((raw, prompt_n, predicted_n, seconds), ("Merhaba.<|eos|>", 21, 4, 0.1))
        self.assertEqual(clean_hy_mt2_output(raw, payload["prompt"]), "Merhaba.")

    def test_transient_server_error_retries_once_and_propagates_review(self):
        provider = self._ready_provider()
        with patch.object(
            provider,
            "_query_chat_completion",
            side_effect=ConnectionResetError("down"),
        ):
            output = provider.translate(
                TranslationInput(items=[TranslationItem(1, "Hello there.", 1)])
            )
        result = output.results[0]
        self.assertIsNone(result.translation)
        self.assertEqual(result.validation_warnings, ["translation_server_error"])
        self.assertTrue(result.requires_review)
        self.assertEqual(provider.metrics.generation_call_count, 2)

    def test_remote_disconnect_retry_logs_external_process_state(self):
        provider = self._ready_provider()
        with patch.object(
            provider,
            "_query_chat_completion",
            side_effect=[
                http.client.RemoteDisconnected("connection closed"),
                ("Merhaba.", 1, 1, 0.01),
            ],
        ), self.assertLogs(
            "providers.translation.hy_mt2_gguf_translation", logging.WARNING
        ) as captured:
            output = provider.translate(
                TranslationInput(items=[TranslationItem(1, "Hello there.", 1)])
            )

        self.assertEqual(output.results[0].translation, "Merhaba.")
        self.assertIn("reason=transient_connection_failure:RemoteDisconnected", captured.output[0])
        self.assertIn("server_pid=external", captured.output[0])
        self.assertIn("process_alive=unknown", captured.output[0])

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

    def test_unload_does_not_terminate_external_server(self):
        provider = HyMT2GGUFTranslationProvider(managed=False)
        external = MagicMock()
        provider._process = external
        provider._owned_process = False
        provider._loaded = True

        provider.unload()

        external.terminate.assert_not_called()
        external.kill.assert_not_called()

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

    # --- P1-A/B: numaralı-batch kayma korumaları (sentetik, model yok) ---

    def _batch_provider(self, responses):
        provider = self._ready_provider()
        patcher = patch.object(provider, "_request_translation", side_effect=list(responses))
        mocked = patcher.start()
        self.addCleanup(patcher.stop)
        return provider, mocked

    def test_duplicate_number_triggers_single_retry(self):
        long_src = "THE OLD KING SLOWLY WALKED BACK HOME TONIGHT"
        batch_raw = "[1] Uzun cevap bir.\n[1] Tekrarlanan satir.\n[2] Kos!"
        provider, mocked = self._batch_provider([
            (batch_raw, batch_raw, False),
            ("Ejderha tek basina geldi.", "Ejderha tek basina geldi.", False),
            ("[1] Kos!", "[1] Kos!", False),
        ])
        out = provider.translate(TranslationInput(items=[
            TranslationItem(1, long_src, 1),
            TranslationItem(2, "GO!", 2),
        ]))
        by_id = {r.region_id: r for r in out.results}
        self.assertIn("tek basina", by_id[1].translation)
        self.assertEqual(by_id[2].translation, "Kos!")
        self.assertFalse(by_id[2].requires_review)
        self.assertEqual(mocked.call_count, 3)  # batch + single + mini-verify

    def test_extreme_length_ratio_triggers_single_retry(self):
        src_a = "THE OLD KING SLOWLY WALKED HOME"
        src_b = "THE YOUNG QUEEN QUIETLY LEFT THE GREAT HALL BEFORE DAWN TODAY"
        essay = "Bu cok uzun bir ceviri metnidir " * 10
        batch_raw = f"[1] {essay}\n[2] Genc kralice sessizce ayrildi."
        provider, mocked = self._batch_provider([
            (batch_raw, batch_raw, False),
            ("Kisa ve dogru.", "Kisa ve dogru.", False),
        ])
        out = provider.translate(TranslationInput(items=[
            TranslationItem(1, src_a, 1),
            TranslationItem(2, src_b, 2),
        ]))
        by_id = {r.region_id: r for r in out.results}
        self.assertEqual(by_id[1].translation, "Kisa ve dogru.")
        self.assertNotIn(essay.strip()[:20], by_id[1].translation)
        self.assertEqual(mocked.call_count, 2)  # batch + single, verify skipped (no fragile)

    def test_verify_agree_keeps_batch_result(self):
        batch_raw = "[1] Hemen git!\n[2] Kral sessizce dinledi."
        provider, mocked = self._batch_provider([
            (batch_raw, batch_raw, False),
            ("[1] Hemen git!", "[1] Hemen git!", False),
        ])
        out = provider.translate(TranslationInput(items=[
            TranslationItem(1, "GO NOW!", 1),
            TranslationItem(2, "THE OLD KING QUIETLY LISTENED FOR A WHILE", 2),
        ]))
        by_id = {r.region_id: r for r in out.results}
        self.assertEqual(by_id[1].translation, "Hemen git!")
        self.assertFalse(by_id[1].requires_review)
        self.assertEqual(mocked.call_count, 2)  # batch + mini, no single tiebreak

    def test_swap_pair_disagreement_flags_both_review(self):
        # 259/260 imzası: bitişik iki kırılgan, oranlar zıt-uçlu (1.7 / 0.3).
        # Mini hemfikir kalır (aynı karışma), tekil izolasyon ayrışır.
        long_src = "THE OLD KING QUIETLY LISTENED FOR A WHILE TODAY"
        batch_raw = "[1] Bu sefer ben uzun uzun yapiyorum iste.\n[2] Bunu...\n[3] Yasli kral bugun bir sure sessizce dinledi."
        mini_raw = "[1] Bu sefer ben uzun uzun yapiyorum iste.\n[2] Bunu..."
        provider, mocked = self._batch_provider([
            (batch_raw, batch_raw, False),
            (mini_raw, mini_raw, False),
            ("Hicbir fikrim yok.", "Hicbir fikrim yok.", False),
            ("Bunu bu sefer yapacagim.", "Bunu bu sefer yapacagim.", False),
        ])
        out = provider.translate(TranslationInput(items=[
            TranslationItem(1, "HAS NO IDEA YET.", 1),
            TranslationItem(2, "THIS TIME, I'M DOING", 2),
            TranslationItem(3, long_src, 3),
        ]))
        by_id = {r.region_id: r for r in out.results}
        self.assertEqual(by_id[1].translation, "Hicbir fikrim yok.")
        self.assertEqual(by_id[2].translation, "Bunu bu sefer yapacagim.")
        for rid in (1, 2):
            self.assertTrue(by_id[rid].requires_review)
            self.assertIn("numbering_inconsistent", by_id[rid].validation_warnings)
        self.assertFalse(by_id[3].requires_review)
        self.assertEqual(mocked.call_count, 4)  # batch + mini + 2 tekil

    def test_swap_pair_agreement_keeps_batch(self):
        long_src = "THE OLD KING QUIETLY LISTENED FOR A WHILE TODAY"
        batch_raw = "[1] Hicbir fikrim yok.\n[2] Bunu bu sefer yapacagim.\n[3] Kral dinledi."
        mini_raw = "[1] Hicbir fikrim yok.\n[2] Bunu bu sefer yapacagim."
        provider, mocked = self._batch_provider([
            (batch_raw, batch_raw, False),
            (mini_raw, mini_raw, False),
            ("Hicbir fikrim yok.", "Hicbir fikrim yok.", False),
            ("Bunu bu sefer yapacagim.", "Bunu bu sefer yapacagim.", False),
        ])
        out = provider.translate(TranslationInput(items=[
            TranslationItem(1, "HAS NO IDEA YET.", 1),
            TranslationItem(2, "THIS TIME, I'M DOING", 2),
            TranslationItem(3, long_src, 3),
        ]))
        by_id = {r.region_id: r for r in out.results}
        self.assertFalse(by_id[1].requires_review)
        self.assertFalse(by_id[2].requires_review)
        # batch + mini; oranlar normal olduğu için tekil kontrol açılmaz
        self.assertEqual(mocked.call_count, 2)

    def test_hedge_lone_item_verified(self):
        long_src = "THE OLD KING QUIETLY LISTENED FOR A WHILE TODAY"
        batch_raw = "[1] Yasli kral bugun bir sure sessizce dinledi.\n[2] Asla!"
        provider, mocked = self._batch_provider([
            (batch_raw, batch_raw, False),
            ("[1] Asla!", "[1] Asla!", False),
            ("Asla teslim olma!", "Asla teslim olma!", False),
        ])
        out = provider.translate(TranslationInput(items=[
            TranslationItem(1, long_src, 1),
            TranslationItem(2, "NEVER SURRENDER!", 2),
        ]))
        by_id = {r.region_id: r for r in out.results}
        self.assertEqual(by_id[2].translation, "Asla teslim olma!")
        self.assertTrue(by_id[2].requires_review)
        self.assertIn("numbering_inconsistent", by_id[2].validation_warnings)
        self.assertFalse(by_id[1].requires_review)

    def test_verify_mismatch_flags_review_with_single_tiebreak(self):
        batch_raw = "[1] Hemen git!\n[2] Kral sessizce dinledi."
        provider, mocked = self._batch_provider([
            (batch_raw, batch_raw, False),
            ("[1] Farkli cevap!", "[1] Farkli cevap!", False),
            ("Selam ver!", "Selam ver!", False),
        ])
        out = provider.translate(TranslationInput(items=[
            TranslationItem(1, "GO NOW!", 1),
            TranslationItem(2, "THE OLD KING QUIETLY LISTENED FOR A WHILE", 2),
        ]))
        by_id = {r.region_id: r for r in out.results}
        self.assertEqual(by_id[1].translation, "Selam ver!")
        self.assertTrue(by_id[1].requires_review)
        self.assertIn("numbering_inconsistent", by_id[1].validation_warnings)
        self.assertEqual(by_id[2].translation, "Kral sessizce dinledi.")
        self.assertEqual(mocked.call_count, 3)  # batch + mini + single


    def test_merge_pair_long_short_flags_both_review(self):
        # F2 spike sınıfı (B255/256): uzun üye şişkin oran (>1.8 — iki
        # içerik tek numarada), kısa üye cılız oran (<0.6). Tekiller
        # ayrışırsa ikisi de REVIEW + tekil metin.
        # Not: kısa üyenin oranı hedge bölgesi dışında (0.5) tutuldu ki
        # bu test merge ağını izole etsin.
        src_long = "ALRIGHT, I THINK I'VE GOT THE HANG OF THIS."
        merged = ("Tamam sanirim artik olayi tamamen cozdum ve Allen da geldi "
                  "simdi hep birlikte basliyoruz haydi bakalim.")
        batch_raw = f"[1] {merged}\n[2] Ah be."
        provider, mocked = self._batch_provider([
            (batch_raw, batch_raw, False),
            ("[1] Ah be.", "[1] Ah be.", False),
            ("Tamamdir, isi kaptim.", "Tamamdir, isi kaptim.", False),
            ("Ama Allen...", "Ama Allen...", False),
        ])
        out = provider.translate(TranslationInput(items=[
            TranslationItem(1, src_long, 1),
            TranslationItem(2, "BUT ALLEN...", 2),
        ]))
        by_id = {r.region_id: r for r in out.results}
        self.assertEqual(by_id[1].translation, "Tamamdir, isi kaptim.")
        self.assertEqual(by_id[2].translation, "Ama Allen...")
        for rid in (1, 2):
            self.assertTrue(by_id[rid].requires_review)
            self.assertIn("numbering_inconsistent", by_id[rid].validation_warnings)
        self.assertEqual(mocked.call_count, 4)  # batch + mini + 2 tekil

    def test_merge_pair_agreement_keeps_batch(self):
        src_a = "THE OLD KING SLOWLY WALKED BACK HOME TONIGHT"
        src_b = "THE YOUNG QUEEN QUIETLY LEFT THE GREAT HALL TODAY"
        batch_raw = "[1] Yasli kral aksama dogru eve yurudu.\n[2] Genc kralice salondan ayrildi."
        provider, mocked = self._batch_provider([
            (batch_raw, batch_raw, False),
        ])
        out = provider.translate(TranslationInput(items=[
            TranslationItem(1, src_a, 1),
            TranslationItem(2, src_b, 2),
        ]))
        by_id = {r.region_id: r for r in out.results}
        self.assertFalse(by_id[1].requires_review)
        self.assertFalse(by_id[2].requires_review)
        self.assertEqual(mocked.call_count, 1)  # yalniz batch


    # --- F2: ad-düşürme bekçisi (sayı + toplu buharlaşma) ---

    def test_dropped_number_token_b87_class(self):
        # "LEVEL ONE" -> "...Seviyesiniz!" : "birinci" düştü. Çeviri
        # korunur ama dropped_number_token + REVIEW ile işaretlenir.
        src = "SETTING YOUR JOB? BUT YOU'RE ONLY LEVEL ONE!"
        bad = "Mesleginizi mi ayarliyorsunuz? Ama siz sadece Seviyesiniz!"
        provider, mocked = self._batch_provider([(bad, bad, False)])
        out = provider.translate(TranslationInput(items=[TranslationItem(1, src, 1)]))
        res = out.results[0]
        self.assertEqual(res.translation, bad)
        self.assertIn("dropped_number_token", res.validation_warnings)
        self.assertNotIn("dropped_content_token", res.validation_warnings)
        self.assertTrue(res.requires_review)
        self.assertEqual(mocked.call_count, 1)  # tekil, ek çağrı yok

    def test_dropped_number_escapes(self):
        # "Seviye 1" / "birinci seviye" doğru karşılıklar — ateşlemez.
        self.assertEqual(
            find_dropped_source_tokens("ONLY LEVEL ONE!", "Sadece Seviye 1!"), []
        )
        self.assertEqual(
            find_dropped_source_tokens("ONLY LEVEL ONE!", "Sadece birinci seviyesin!"), []
        )
        self.assertIn(
            "dropped_number_token",
            find_dropped_source_tokens("ONLY LEVEL ONE!", "Sadece seviyesin!"),
        )

    def test_dropped_content_mass_evaporation(self):
        # B255 sınıfı: uzun kaynak buharlaşıp kısa kalırsa ateşler.
        src = "ALRIGHT, I THINK I'VE GOT THE HANG OF THIS AND THEN WE MARCH HOME"
        self.assertIn(
            "dropped_content_token",
            find_dropped_source_tokens(src, "Tamam."),
        )
        # Normal oranlı çeviride ortak adların çevrilmesi ateşlemez.
        self.assertEqual(find_dropped_source_tokens("INTO THE WORLD!", "Dünyaya!"), [])
        self.assertEqual(find_dropped_source_tokens("ONLINE", "ÇEVRİMİÇİ"), [])

    def test_legit_translation_no_dropped_flag(self):
        provider, mocked = self._batch_provider([("ÇEVRİMİÇİ", "ÇEVRİMİÇİ", False)])
        out = provider.translate(TranslationInput(items=[TranslationItem(1, "ONLINE", 1)]))
        res = out.results[0]
        self.assertEqual(res.translation, "ÇEVRİMİÇİ")
        self.assertNotIn("dropped_number_token", res.validation_warnings)
        self.assertNotIn("dropped_content_token", res.validation_warnings)


if __name__ == "__main__":
    unittest.main()
