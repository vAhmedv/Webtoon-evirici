# scripts/ rehberi

55+ script birikmiştir; hangisinin güncel olduğu belirsizliği bakım maliyetidir.
Taşıma yapmadan önce tek kaynak kuralı:

## Üretim entrypoint'leri (güncel tut)
- `run_production_e2e_chapter.py` — uçtan uca üretim koşumu
- `audit_e2e_real_chapter1.py` — gerçek bölüm doğrulaması (README'deki komut)
- `audit_generalization_chapter2.py` — genelleme denetimi
- `real_chapter_translation_gate_v1.py` — çeviri gate (manuel)
- `hy_mt2_production_gate_v1.py` — Hy-MT2 üretim gate
- `check_gpu.py`, `generate_test_chapter.py`, `process_chapter.py`, `cache_flush.py`, `download_detector_models.py`

## Deneysel / versiyon çöplüğü (arşiv adayı, silme yok)
- `translategemma_quality_gate_v1..v5.py`, `translategemma_smoke_v2.py`,
  `translategemma_standalone_bench.py`, `translategemma_torture_test_v1.py`,
  `translategemma_prompt_ab/c_test.py`, `translategemma_production_smoke_test.py`
- `qwen_translation_smoke_test.py`, `qwen_translation_smoke_test_2item.py`,
  `qwen_translation_generic_smoke_test.py`, `qwen_gguf_smoke_test.py`,
  `qwen_discovery/repair/glossary/evidence_termbase_smoke*`, `qwen_4bit_bench.py`,
  `qwen_ab_control_bench.py`, `qwen_translator_v2_sanity.py`
- `semantic_context_v1/v2/v3_benchmark.py`, `semantic_context_v3_sanity.py`,
  `translation_quality_benchmark.py`, `translation_model_shootout_v1.py`,
  `translation_holdout_v2/v3.py`, `hy_mt2_karga_real30_shootout_v1.py`
- `validate_text_mask_inpainting_v5.py`, `validate_text_mask_inpainting_v51.py`,
  `validate_text_mask_v52_fresh.py`, `validate_v5_3_grouping_ocr.py`,
  `benchmark_v5_3_performance_and_visual_review.py`

## Kural
- Yeni deneme scripti eklerken versiyonlamak yerine (`*_v6.py`) mevcut gate'e
  parametre ekleyin veya `scripts/archive/` altına taşıyın.
- Üretim davranışı değişirse önce `tests/`'e regresyon testi ekleyin.
