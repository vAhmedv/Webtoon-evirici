# scripts/archive — tek-kullanımlık deney scriptleri (2026-09-16)

Üretimde ve testlerde KULLANILMAYAN shootout/bench/smoke/validate/preview
scriptleri buradadır (`git mv` ile taşındı, geçmiş korundu).
Testler yalnızca üst klasördeki 7 script'i import eder veya konumunu sabitler:
`audit_e2e_real_chapter1`, `compare_golden_v1`, `run_golden_remeasure_v1`,
`mine_blind_spots_v1`, `generate_v5_3_visual_review_and_report`,
`benchmark_v5_3_performance_and_visual_review`,
`real_chapter_translation_gate_v1`.
Buradaki dosyalar çalıştırılmak için değil, kanıt arşivi için durur.
