#!/usr/bin/env python3
"""
Standalone DeepSeek-OCR-2 smoke test on real bubble crop.
No application code changes. Uses existing .venv.
"""
import sys
import time
import traceback
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from PIL import Image

CROP_PATH = r"test_data\output\region_0_crop.png"
MODEL_ID = "deepseek-community/DeepSeek-OCR-2"


def main():
    t0 = time.perf_counter()
    print(f"Python: {sys.version.split()[0]}")
    print(f"Torch: {torch.__version__}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA: {torch.version.cuda}")

    crop = Image.open(CROP_PATH).convert("RGB")
    print(f"Crop: {CROP_PATH}  size={crop.size}")

    # ── Load processor & model ────────────────────────────────────────────────
    print(f"\nLoading {MODEL_ID} …")
    load_start = time.perf_counter()
    mem_before = None
    try:
        from transformers import AutoProcessor, AutoModelForImageTextToText
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            dtype=torch.bfloat16,
        )
        if device == "cuda":
            model = model.to("cuda:0")
        model.eval()
    except Exception as e:
        print("LOAD_FAIL:")
        traceback.print_exc()
        sys.exit(3)
    load_time = time.perf_counter() - load_start
    print(f"Model loaded in {load_time:.2f}s")

    if device == "cuda":
        mem_before = torch.cuda.memory_allocated() / 1024**2
        print(f"VRAM allocated after load: {mem_before:.1f} MB")
        torch.cuda.reset_peak_memory_stats()

    # ── Prompt 1 ─────────────────────────────────────────────────────────────
    print("\nRunning OCR Prompt 1 …")
    inference_start = time.perf_counter()
    text1 = None
    try:
        inputs = processor(
            images=crop,
            text="<image>\nFree OCR.",
            return_tensors="pt",
        )
        if device == "cuda":
            inputs = {
                k: v.to(torch.bfloat16).to("cuda:0") if v.dtype == torch.float32 else v.to("cuda:0")
                for k, v in inputs.items()
            }
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=256,
            )
        new_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
        text1 = processor.decode(new_ids[0], skip_special_tokens=True)
    except Exception as e:
        print("INFERENCE_FAIL_PROMPT1:")
        traceback.print_exc()
    inference_time = time.perf_counter() - inference_start

    peak_mem = None
    if device == "cuda":
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2
        print(f"VRAM peak: {peak_mem:.1f} MB")
        torch.cuda.reset_peak_memory_stats()

    # ── Prompt 2 (fallback) ──────────────────────────────────────────────────
    text2 = None
    inference_time2 = None
    if text1 is None or text1.strip() == "":
        print("\nPrompt 1 failed/empty, trying Prompt 2 …")
        inference_start = time.perf_counter()
        try:
            inputs = processor(
                images=crop,
                text="<image>\nTranscribe all visible English text exactly as written.",
                return_tensors="pt",
            )
            if device == "cuda":
                inputs = {
                    k: v.to(torch.bfloat16).to("cuda:0") if v.dtype == torch.float32 else v.to("cuda:0")
                    for k, v in inputs.items()
                }
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=256,
                )
            new_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
            text2 = processor.decode(new_ids[0], skip_special_tokens=True)
            inference_time2 = time.perf_counter() - inference_start
            if device == "cuda":
                peak_mem2 = torch.cuda.max_memory_allocated() / 1024**2
                print(f"VRAM peak prompt2: {peak_mem2:.1f} MB")
        except Exception as e:
            print("INFERENCE_FAIL_PROMPT2:")
            traceback.print_exc()
            text2 = None
    else:
        print("\nPrompt 1 succeeded, Prompt 2 not needed.")

    total_time = time.perf_counter() - t0

    # ── Report ────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("SMOKE TEST RESULT (DeepSeek-OCR-2)")
    print("="*60)
    print(f"Load success: YES")
    if text1 is not None:
        print(f"Prompt 1 raw output: {text1!r}")
        print(f"Prompt 1 inference time: {inference_time:.2f}s")
    if text2 is not None:
        print(f"Prompt 2 raw output: {text2!r}")
        print(f"Prompt 2 inference time: {inference_time2:.2f}s")
    if peak_mem is not None:
        print(f"Max VRAM: {peak_mem:.1f} MB")
    else:
        print("Max VRAM: N/A (CPU)")
    print(f"Total time: {total_time:.2f}s")
    print(f"Torch: {torch.__version__}")
    print("="*60)


if __name__ == "__main__":
    main()