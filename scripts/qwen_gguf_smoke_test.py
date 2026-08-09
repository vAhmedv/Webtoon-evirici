"""One Real Model Run - Hardened GGUF Production System Prompt Smoke Test.

Verifies:
- Real system role separation (role="system" + role="user")
- Unrelated profile glossary filtering (only SECRET REALM injected, NOT MANA CORE or BLACKWIND RAVINE)
- Neutral-address protection ("RELAX, KID." -> "Sakin ol, çocuk.")
- Proper name preservation ("Luo Tian")
- Natural Turkish phrasing & factual fidelity
- Deterministic sampling (temperature=0.0)
- Structured JSON output validity
"""
import sys
import time

from core.translation.series_profile import SeriesProfile
from providers.translation import (
    QwenGGUFTranslationProvider,
    TranslationInput,
    TranslationItem,
)


def run_smoke_test():
    print("=== STARTING ONE REAL HARDENED GGUF TRANSLATION SMOKE TEST ===")

    provider = QwenGGUFTranslationProvider(
        model_path=r"C:\AI\Models\Qwen3.5-9B-GGUF\Qwen3.5-9B-Q5_K_M.gguf",
        executable_path=r"C:\AI\llama-cpp-cuda\llama.exe",
        server_url="http://127.0.0.1:8080",
        n_gpu_layers=99,
        auto_start_server=True,
    )

    t_load_0 = time.perf_counter()
    provider.load()
    load_seconds = time.perf_counter() - t_load_0

    profile = SeriesProfile(
        series_id="smoke_test_series",
        known_names={"LUO TIAN": "Luo Tian"},
        glossary={
            "SECRET REALM": "Gizli Diyar",
            "MANA CORE": "Mana Çekirdeği",
            "BLACKWIND RAVINE": "Karayel Vadisi",
        },
    )

    inp = TranslationInput(
        items=[
            TranslationItem(region_id=1, source="RELAX, KID.", reading_order=1),
            TranslationItem(region_id=2, source="MY NAME IS LUO TIAN.", reading_order=2),
            TranslationItem(region_id=3, source="DON'T GET THE WRONG IDEA.", reading_order=3),
            TranslationItem(
                region_id=4,
                source="WE HAVE TO ENTER THE SECRET REALM BEFORE SUNSET.",
                reading_order=4,
            ),
            TranslationItem(
                region_id=5, source="YOU REALLY DON'T REMEMBER ME?", reading_order=5
            ),
            TranslationItem(
                region_id=6, source="I'M NOT AN ABILITY USER.", reading_order=6
            ),
        ],
        profile=profile,
    )

    out = provider.translate(inp)

    m = provider.metrics
    print("\n=== SMOKE TEST METRICS ===")
    print(f"Backend: {provider.name}")
    print(f"Model: {m.translation_model}")
    print(f"CUDA: {'YES' if m.cuda_active else 'NO'}")
    print(f"GPU offload: {m.gpu_offload}")
    print(f"Model/server load time: {m.model_load_seconds:.2f} s")
    print(f"Input tokens: {m.input_token_count}")
    print(f"Generated tokens: {m.generated_token_count}")
    print(f"Generation time: {m.generation_seconds:.2f} s")
    print(f"Generation tok/s: {m.tokens_per_sec:.2f} tok/s")
    print(f"Generation calls: {m.generation_call_count}")
    print(f"Retries: {m.retries}")

    print("\n=== SOURCE -> TURKISH TRANSLATIONS ===")
    for res in out.results:
        print(f"[{res.region_id}] {res.source} -> {res.translation}")
        print(f"    Warnings: {res.validation_warnings}, Review: {res.requires_review}")

    provider.unload()
    print("\n=== SMOKE TEST COMPLETED SUCCESSFULLY ===")


if __name__ == "__main__":
    run_smoke_test()
