#!/usr/bin/env python3
"""
DeepSeek-OCR-2 smoke test on real bubble crop.

Fixes:
- similarity calculation (difflib.SequenceMatcher, uppercase+whitespace normalize)
- Prompt 2 fallback runs whenever Prompt 1 fails SANITY (not just when empty)
- raw OCR output, similarity, time, peak VRAM for both prompts
"""
import re
import sys
import time
import traceback
from difflib import SequenceMatcher
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from PIL import Image

CROP_PATH = r"test_data\output\region_0_crop.png"
MODEL_ID = "deepseek-community/DeepSeek-OCR-2"

REFERENCE = (
    "JUDGING BY LUO TIAN'S PERFORMANCE JUST NOW, "
    "HE'S ALMOST ON PAR WITH A LEVEL 1 ABILITY USER "
    "WHO SPECIALIZES IN ARCHERY."
)

PROMPT_1 = "<image>\nFree OCR."
PROMPT_2 = "<image>\nTranscribe all visible English text exactly as written."

SANITY_THRESHOLD = 0.85

# Key phrases that must appear for meaningful English OCR
KEY_PHRASES = ["JUDGING", "LUO TIAN", "ABILITY USER", "ARCHERY"]


def normalize(text: str) -> str:
    """Uppercase + whitespace normalize only (no punctuation stripping)."""
    return re.sub(r"\s+", " ", text.upper()).strip()


def similarity(a: str, b: str) -> float:
    """Deterministic similarity via difflib.SequenceMatcher."""
    return SequenceMatcher(None, normalize(a), normalize(b)).ratio()


def contains_key_phrases(text: str) -> bool:
    """Check that key phrases appear in the normalized text."""
    norm = normalize(text)
    return all(phrase in norm for phrase in KEY_PHRASES)


def sanity_pass(text: str) -> bool:
    """Sanity PASS: similarity >= 0.85 AND meaningful English OCR."""
    if not text or not text.strip():
        return False
    sim = similarity(text, REFERENCE)
    if sim < SANITY_THRESHOLD:
        return False
    return contains_key_phrases(text)


def run_prompt(processor, model, crop, prompt: str, device: str):
    """Run a single prompt, return (text, inference_time, peak_vram_mb)."""
    inference_start = time.perf_counter()
    text = None
    try:
        inputs = processor(
            images=crop,
            text=prompt,
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
        text = processor.decode(new_ids[0], skip_special_tokens=True)
    except Exception as e:
        print("INFERENCE_FAIL:")
        traceback.print_exc()
    inference_time = time.perf_counter() - inference_start

    peak_mem = None
    if device == "cuda":
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2
        torch.cuda.reset_peak_memory_stats()

    return text, inference_time, peak_mem


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

    model_load_vram = None
    if device == "cuda":
        model_load_vram = torch.cuda.memory_allocated() / 1024**2
        print(f"VRAM allocated after load: {model_load_vram:.1f} MB")
        torch.cuda.reset_peak_memory_stats()

    # ── Prompt 1 ─────────────────────────────────────────────────────────────
    print("\nRunning OCR Prompt 1 …")
    text1, time1, peak1 = run_prompt(processor, model, crop, PROMPT_1, device)
    sim1 = similarity(text1, REFERENCE) if text1 else 0.0
    print(f"Prompt 1 raw: {text1!r}")
    print(f"Prompt 1 similarity: {sim1:.4f}")
    print(f"Prompt 1 inference time: {time1:.2f}s")
    if peak1 is not None:
        print(f"Prompt 1 peak VRAM: {peak1:.1f} MB")

    # ── Prompt 2 (fallback whenever Prompt 1 fails SANITY) ───────────────────
    text2, time2, peak2 = None, None, None
    sim2 = 0.0
    pass1 = sanity_pass(text1) if text1 else False
    if not pass1:
        print("\nPrompt 1 failed SANITY, trying Prompt 2 …")
        text2, time2, peak2 = run_prompt(processor, model, crop, PROMPT_2, device)
        sim2 = similarity(text2, REFERENCE) if text2 else 0.0
        print(f"Prompt 2 raw: {text2!r}")
        print(f"Prompt 2 similarity: {sim2:.4f}")
        print(f"Prompt 2 inference time: {time2:.2f}s")
        if peak2 is not None:
            print(f"Prompt 2 peak VRAM: {peak2:.1f} MB")
    else:
        print("\nPrompt 1 passed SANITY, Prompt 2 not needed.")

    # ── Select winner ─────────────────────────────────────────────────────────
    candidates = []
    if text1:
        candidates.append((sim1, "Prompt 1", text1, time1, peak1))
    if text2:
        candidates.append((sim2, "Prompt 2", text2, time2, peak2))

    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        win_sim, win_name, win_text, win_time, win_peak = candidates[0]
    else:
        win_sim, win_name, win_text, win_time, win_peak = 0.0, "None", None, None, None

    print(f"\nWinning prompt: {win_name} (similarity {win_sim:.4f})")

    # ── Sanity verdict ────────────────────────────────────────────────────────
    sanity_ok = sanity_pass(win_text) if win_text else False
    print(f"Sanity: {'PASS' if sanity_ok else 'FAIL'}")

    total_time = time.perf_counter() - t0

    # ── Report ────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("SMOKE TEST RESULT (DeepSeek-OCR-2)")
    print("="*60)
    print(f"Load success: YES")
    print(f"Model-load VRAM: {model_load_vram:.1f} MB" if model_load_vram else "Model-load VRAM: N/A (CPU)")
    print(f"Prompt 1 raw: {text1!r}")
    print(f"Similarity 1: {sim1:.4f}")
    if text2 is not None:
        print(f"Prompt 2 raw: {text2!r}")
        print(f"Similarity 2: {sim2:.4f}")
    print(f"Winning prompt: {win_name}")
    print(f"Sanity: {'PASS' if sanity_ok else 'FAIL'}")
    if win_peak is not None:
        print(f"Region 0 peak VRAM: {win_peak:.1f} MB")
    if win_time is not None:
        print(f"Region 0 inference time: {win_time:.2f}s")
    print(f"Total time: {total_time:.2f}s")
    print(f"Torch: {torch.__version__}")
    print("="*60)

    sys.exit(0 if sanity_ok else 4)


if __name__ == "__main__":
    main()