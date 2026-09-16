"""Minimal 4-bit (nf4) generation-speed benchmark for Qwen3.5-9B.

Standalone / does NOT touch production translation code.
Runs ONE short EN->TR text-generation pass (~256 output tokens) and reports:
  Model load time | VRAM after load | Peak VRAM | Input tokens |
  Generated tokens | Generation time | Tokens/sec

4-bit config: load_in_4bit=True, bnb_4bit_quant_type="nf4",
bnb_4bit_compute_dtype=float16, double-quant on.
"""
from __future__ import annotations

import sys
import time
import traceback

import torch  # noqa: E402  (imported early so cuda stats exist)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

MODEL = r"C:\AI\Models\Qwen3.5-9B"
MAX_MEM = {0: "12GiB"}


def _gb(b: float) -> float:
    return b / (1024 ** 3)


PROMPT = (
    "Çevir: The warrior drew his sword and charged forward, his heart pounding "
    "with rage and determination as the ancient battlefield echoed with clashing steel. "
    "Around him, the mist clung to the broken banners and fallen leaves, and in the "
    "distance the crimson sun dipped below the mountains, signaling that the final "
    "confrontation was about to begin.\nTurkish:"
)


def main() -> None:
    from transformers import (
        AutoProcessor,
        AutoModelForImageTextToText,
        BitsAndBytesConfig,
    )

    torch.cuda.empty_cache()

    # ---- Load (4-bit nf4) ----
    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    qconfig = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL,
        local_files_only=True,
        torch_dtype=torch.float16,
        device_map="auto",
        max_memory=MAX_MEM,
        quantization_config=qconfig,
    )
    model.eval()
    load_time = time.perf_counter() - t0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    vram_after_load = _gb(torch.cuda.memory_allocated()) if torch.cuda.is_available() else 0.0

    # ---- Single short EN->TR prompt, ~256 output tokens ----
    input_ids = processor.tokenizer(
        PROMPT, return_tensors="pt"
    ).input_ids.to("cuda")
    input_tokens = int(input_ids.shape[-1])

    t1 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            do_sample=True,
            temperature=0.2,
            top_p=0.9,
            max_new_tokens=256,
            min_new_tokens=256,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
    gen_time = time.perf_counter() - t1
    generated_tokens = int(out.shape[-1] - input_ids.shape[-1])
    peak_vram = _gb(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0.0
    tps = generated_tokens / gen_time if gen_time > 0 else 0.0
    text = processor.decode(out[0], skip_special_tokens=True)
    preview = text[-400:].replace("\r", " ")

    print("=" * 60)
    print("Qwen3.5-9B 4-bit (nf4) generation speed benchmark")
    print("=" * 60)
    print(f"Model load time: {load_time:.2f} s")
    print(f"VRAM after load: {vram_after_load:.2f} GB")
    print(f"Peak VRAM: {peak_vram:.2f} GB")
    print(f"Input tokens: {input_tokens}")
    print(f"Generated tokens: {generated_tokens}")
    print(f"Generation time: {gen_time:.2f} s")
    print(f"Tokens/sec: {tps:.2f}")
    print("-" * 60)
    print("Output tail (preview):")
    print(preview)
    print("-" * 60)
    print("Model unloaded.")
    del model, processor, out
    torch.cuda.empty_cache()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
