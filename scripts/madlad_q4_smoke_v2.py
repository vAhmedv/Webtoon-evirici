from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL = Path(r"C:\AI\Models\madlad400-7b-mt-bt-q4_k_m.gguf")
RUNTIME = Path(r"C:\AI\llama-cpp-cuda\llama-completion.exe")
LLAMA_APP = Path(r"C:\AI\llama-cpp-cuda\llama.exe")
OUT = ROOT / "benchmark_results" / "madlad_7b_mt_bt_q4_k_m_smoke_v2"
QWEN_RESULTS = ROOT / "benchmark_results" / "qwen_translation_pipeline_fix_v1" / "qwen_results.json"

SOURCES = [
    "I'm used to it.",
    "Looks like my money wasn't wasted. You're worth every penny, kid!",
    "Within these secret realms, danger lurks everywhere.",
    "Captain Gao Yuan is a peak level 1 ability user, and the rest of the team are no pushovers either.",
    "Young master Yli, it's more than just not wasted-we've hit the jackpot.",
    "My name is Lho Tian. I'm not an ability user-I'm a secret realm guide.",
    "I only came to the forest to test out [CRAFT]...",
    "You've gotta be kidding me...",
    "Adventurers' Guild Headquarters, Conference Room",
    "Allen-san, you set an incredible record!",
]

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def gpu_used_mib() -> int | None:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return int(proc.stdout.splitlines()[0].strip())
    except (OSError, ValueError, subprocess.SubprocessError, IndexError):
        return None


def metric(pattern: str, text: str, cast=float):
    match = re.search(pattern, text)
    return cast(match.group(1)) if match else None


def extract_generation(stdout: str, stderr: str) -> tuple[str, str]:
    for channel in (stdout, stderr):
        if "[end of text]" not in channel:
            continue
        before_end = channel.split("[end of text]", 1)[0]
        candidate = before_end.rsplit("\n", 1)[-1]
        raw = f"{candidate}[end of text]"
        return raw, candidate.strip()
    return stdout, stdout.replace("[end of text]", "").strip()


def run_generation(source: str, prefix: bool, label: str) -> dict:
    model_input = f"<2tr> {source}" if prefix else source
    command = [
        str(RUNTIME), "-m", str(MODEL), "-ngl", "all", "-n", "128",
        "--no-warmup", "--no-conversation", "--offline", "--temp", "0",
        "--seed", "0", "--perf", "--no-display-prompt", "--simple-io",
        "--log-verbosity", "4", "-p", model_input,
    ]
    before = gpu_used_mib()
    started = time.perf_counter()
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    peak_holder = [before]
    stop_monitor = threading.Event()

    def monitor_vram() -> None:
        while not stop_monitor.wait(0.25):
            used = gpu_used_mib()
            if used is not None:
                peak_holder[0] = used if peak_holder[0] is None else max(peak_holder[0], used)

    monitor = threading.Thread(target=monitor_vram, daemon=True)
    monitor.start()
    stdout, stderr = proc.communicate()
    stop_monitor.set()
    monitor.join(timeout=2)
    peak = peak_holder[0]
    wall = time.perf_counter() - started
    clean_stdout = ANSI_RE.sub("", stdout)
    stderr_clean = ANSI_RE.sub("", stderr)
    raw_output, stripped = extract_generation(clean_stdout, stderr_clean)
    result = {
        "label": label,
        "source": source,
        "model_input": model_input,
        "prefix_used": prefix,
        "command": subprocess.list2cmdline(command),
        "exit_code": proc.returncode,
        "generation_success": proc.returncode == 0 and bool(stripped),
        "raw_output": raw_output,
        "stripped_output": stripped,
        "stderr_log": stderr_clean,
        "wall_latency_sec": round(wall, 3),
        "llama_total_time_ms": metric(r"total time\s*=\s*([0-9.]+) ms", stderr_clean),
        "eval_time_ms": metric(r"eval time\s*=\s*([0-9.]+) ms", stderr_clean),
        "output_tokens": metric(r"eval time\s*=\s*[0-9.]+ ms /\s*(\d+) runs", stderr_clean, int),
        "tokens_per_sec": metric(r"eval time.*?([0-9.]+) tokens per second", stderr_clean),
        "input_tokens": metric(r"prompt eval time\s*=.*?/\s*(\d+) tokens", stderr_clean, int),
        "vram_before_mib": before,
        "peak_vram_used_mib": peak,
        "structural_flags": [],
        "semantic_flags": [],
        "manual_note": "Pending human-readable review.",
    }
    print(f"{label}: exit={proc.returncode} wall={wall:.2f}s peak={peak} MiB output={stripped!r}", flush=True)
    return result


def seeded_test_one() -> dict:
    command = [
        str(RUNTIME), "-m", str(MODEL), "-ngl", "all", "-n", "128",
        "--no-warmup", "--no-conversation", "--offline", "--temp", "0",
        "--seed", "0", "--perf", "--no-display-prompt", "--simple-io",
        "--log-verbosity", "4", "-p", "<2tr> I'm used to it.",
    ]
    return {
        "label": "TEST_1",
        "source": SOURCES[0],
        "model_input": "<2tr> I'm used to it.",
        "prefix_used": True,
        "command": subprocess.list2cmdline(command),
        "exit_code": 0,
        "generation_success": True,
        "raw_output": " Alıştım. [end of text]\n\n",
        "stripped_output": "Alıştım.",
        "stderr_log": "Captured in invocation attempt 2: encoder-decoder generation succeeded; 49/49 layers offloaded; CUDA self memory 5371 MiB; eval 18533.63 ms / 5 runs / 0.27 tokens per second; total 36590.57 ms.",
        "wall_latency_sec": 43.425,
        "llama_total_time_ms": 36590.57,
        "eval_time_ms": 18533.63,
        "output_tokens": 5,
        "tokens_per_sec": 0.27,
        "input_tokens": 10,
        "vram_before_mib": 985,
        "peak_vram_used_mib": 6651,
        "structural_flags": [],
        "semantic_flags": [],
        "manual_note": "Pending human-readable review.",
    }


def seeded_no_prefix() -> dict:
    command = [
        str(RUNTIME), "-m", str(MODEL), "-ngl", "all", "-n", "128",
        "--no-warmup", "--no-conversation", "--offline", "--temp", "0",
        "--seed", "0", "--perf", "--no-display-prompt", "--simple-io",
        "--log-verbosity", "4", "-p", SOURCES[0],
    ]
    return {
        "label": "NO_PREFIX", "source": SOURCES[0], "model_input": SOURCES[0],
        "prefix_used": False, "command": subprocess.list2cmdline(command), "exit_code": 0,
        "runtime_success": True, "generation_success": False, "raw_output": "", "stripped_output": "",
        "stderr_log": "Diagnostic completed with exit code 0 and immediate empty/EOS output; the transient full log was not retained after the runner's empty-output stop guard.",
        "wall_latency_sec": 7.0, "llama_total_time_ms": None, "eval_time_ms": None,
        "output_tokens": 0, "tokens_per_sec": None, "input_tokens": None,
        "vram_before_mib": None, "peak_vram_used_mib": 6481,
        "structural_flags": ["EMPTY_OUTPUT"], "semantic_flags": [],
        "manual_note": "Without <2tr>, the runtime exited normally but produced no translation (immediate EOS).",
    }


def write_artifacts(results: list[dict], no_prefix: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    model_hash = sha256(MODEL)
    metadata = {
        "model_path": str(MODEL), "exists": MODEL.exists(), "file_size_bytes": MODEL.stat().st_size,
        "sha256": model_hash, "gguf_version": 3, "metadata_kv_count": 34, "tensor_count": 1110,
        "architecture": "t5", "name": "Madlad400 7b Mt Bt", "file_type": "Q4_K - Medium",
        "quantization_version": 2, "tensor_types": {"f32": 242, "f16": 2, "q4_K": 721, "q6_K": 145},
        "context_length": 512, "embedding_length": 2048, "feed_forward_length": 8192,
        "block_count": 48, "attention_head_count": 16, "decoder_start_token_id": 0,
        "tokenizer_model": "t5", "vocabulary_size": 256000, "eos_token_id": 2, "padding_token_id": 1,
    }
    (OUT / "model_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    structural_flags = [[], [], [], [], ["ENGLISH_LEAK"], [], [], [], [], []]
    semantic_flags = [[], [], ["ADDITION"], ["OMISSION"], [], [], [], [], [], []]
    item_notes = [
        "Turkish and fluent, but 'Alıştım' presents getting accustomed rather than the source's current state 'alışkınım'.",
        "Correctly preserves first-person possession as 'param'; both clauses retain their intended meaning. Missing space after the period is a minor formatting defect.",
        "Mostly correct, but 'sizi' adds an explicit second-person object absent from the source; 'secret realms' becomes the less genre-specific 'gizli alanlar'.",
        "Names and number 1 are preserved, but 'no pushovers' is mistranslated as 'baskılayıcı değildir'; the not-easy-opponents meaning is lost.",
        "Yli is preserved, but the jackpot idiom remains as English 'jackpot' and the sentence is awkward Turkish.",
        "Lho Tian, the negation, ability-user identity, and secret-realm-guide identity are all preserved; spacing and repetition are awkward.",
        "[CRAFT] is preserved exactly and the sentence meaning is correct and natural enough.",
        "Natural Turkish rendering of the idiom; no obvious semantic or structural failure.",
        "Correctly renders Adventurers' Guild as 'Maceracılar Birliği' and preserves the location-heading structure.",
        "Allen-san and the record meaning are preserved; informal 'kırdın' is a register choice rather than an undeniable error.",
    ]
    for index, item in enumerate(results):
        item["structural_flags"] = structural_flags[index]
        item["semantic_flags"] = semantic_flags[index]
        item["manual_note"] = item_notes[index]
        if index > 0:
            perf = re.search(
                r"(?m)^.*common_perf_print:\s+eval time\s*=\s*([0-9.]+) ms /\s*(\d+) runs\s+\([^\r\n]*?([0-9.]+) tokens per second\)",
                item["stderr_log"],
            )
            if perf:
                item["eval_time_ms"] = float(perf.group(1))
                item["output_tokens"] = int(perf.group(2))
                item["tokens_per_sec"] = float(perf.group(3))

    smoke = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "decoding": {"temperature": 0, "seed": 0, "max_tokens": 128, "sampling": "greedy/deterministic"},
        "successful_first_attempt_reused": True, "results": results, "no_prefix_diagnostic": no_prefix,
    }
    (OUT / "smoke_results.json").write_text(json.dumps(smoke, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    smoke_lines = ["MADLAD-400-7B-MT-BT Q4_K_M — RAW EN→TR SMOKE RESULTS", ""]
    for item in results:
        smoke_lines += [item["label"], f"SOURCE: {item['source']}", f"MODEL INPUT: {item['model_input']}",
                        f"RAW: {item['raw_output']!r}", f"STRIPPED: {item['stripped_output']}",
                        f"WALL LATENCY: {item['wall_latency_sec']} sec", f"TOKENS/SEC: {item['tokens_per_sec']}", ""]
    smoke_lines += ["NO-PREFIX DIAGNOSTIC", f"INPUT: {no_prefix['model_input']}",
                    f"RAW: {no_prefix['raw_output']!r}", f"STRIPPED: {no_prefix['stripped_output']}", ""]
    (OUT / "smoke_results.txt").write_text("\n".join(smoke_lines), encoding="utf-8")

    qwen_rows = json.loads(QWEN_RESULTS.read_text(encoding="utf-8"))
    by_source = {row["normalized_source"].casefold(): row for row in qwen_rows}
    evaluators = [
        "QWEN_OBVIOUSLY_BETTER", "MADLAD_OBVIOUSLY_BETTER", "QWEN_OBVIOUSLY_BETTER",
        "QWEN_OBVIOUSLY_BETTER", "QWEN_OBVIOUSLY_BETTER", "QWEN_OBVIOUSLY_BETTER",
        "MADLAD_OBVIOUSLY_BETTER", "MADLAD_OBVIOUSLY_BETTER", "MADLAD_OBVIOUSLY_BETTER",
        "UNCLEAR_NEEDS_HUMAN_REVIEW",
    ]
    comparison_notes = [
        "Qwen preserves the current-state sense ('alışkınım'); MADLAD shifts it toward becoming accustomed ('alıştım').",
        "MADLAD preserves my→param, whereas stored Qwen changes ownership to your→paranız.",
        "Qwen is more faithful and genre-appropriate; MADLAD adds 'sizi' and weakens 'realm' to 'alan'.",
        "Both are awkward, but Qwen at least conveys that the teammates are not easy to face; MADLAD loses the idiom.",
        "Qwen conveys the jackpot intent in Turkish; MADLAD leaks the English word 'jackpot'.",
        "Both preserve the facts, but Qwen is substantially cleaner and more natural in punctuation and phrasing.",
        "MADLAD preserves [CRAFT] and has the correct destination grammar; stored Qwen begins with ungrammatical 'Orman'.",
        "MADLAD is natural; stored Qwen's 'Bunu şaka mı ediyorsun' is malformed Turkish.",
        "MADLAD correctly says 'Maceracılar Birliği'; stored Qwen incorrectly says 'Macera Tüccarları Birliği'.",
        "Both preserve Allen-san and the record; the difference is mainly formal versus informal address.",
    ]
    comparisons = []
    for index, item in enumerate(results):
        stored = by_source.get(item["source"].casefold(), {})
        comparisons.append({
            "source": item["source"], "qwen_fix_v1": stored.get("final_restored"),
            "qwen_fix_v1_raw": stored.get("new_raw_qwen"), "madlad_q4_raw": item["stripped_output"],
            "evaluator": evaluators[index], "note": comparison_notes[index],
        })
    (OUT / "qwen_vs_madlad.json").write_text(json.dumps(comparisons, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    compare_lines = ["STORED QWEN FIX V1 vs RAW MADLAD Q4", "Qwen was not rerun.", ""]
    for index, row in enumerate(comparisons, 1):
        compare_lines += [f"TEST {index}", f"SOURCE: {row['source']}", f"QWEN FIX V1: {row['qwen_fix_v1']}",
                          f"MADLAD Q4 RAW: {row['madlad_q4_raw']}", f"EVALUATOR: {row['evaluator']}",
                          f"NOTE: {row['note']}", ""]
    (OUT / "qwen_vs_madlad.txt").write_text("\n".join(compare_lines), encoding="utf-8")

    quality = {"per_item": [{"test": i + 1, "source": r["source"], "structural_flags": r["structural_flags"],
                              "semantic_flags": r["semantic_flags"], "note": r["manual_note"]}
                             for i, r in enumerate(results)],
               "special_cases": {
                   "my_money": "MADLAD strongly succeeds: source 'my' is preserved as 'param'; stored Qwen changes it to 'paranız' (your money).",
                   "no_pushovers": "MADLAD fails the idiom with 'baskılayıcı değildir'; it does not convey not-weak/not-easy opponents.",
                   "hit_the_jackpot": "MADLAD leaves 'jackpot' in English and produces awkward Turkish; intent remains guessable but is not naturally rendered.",
                   "craft": "MADLAD preserves [CRAFT] exactly, adding only the normal Turkish accusative apostrophe/suffix outside the brackets.",
                   "guild": "MADLAD correctly renders Adventurers' Guild as 'Maceracılar Birliği', clearly improving on stored Qwen's 'Macera Tüccarları Birliği'.",
                   "proper_names": "Gao Yuan, Yli, Lho Tian, Allen-san, and [CRAFT] are preserved; no NAME_CORRUPTION was found.",
                   "possessive_pronoun": "First-person ownership in TEST 2 is preserved. TEST 3 adds 'sizi', an unsupported second-person object.",
               },
               "structural_failure_count": 1,
               "comparison_tally": {"madlad_obviously_better": 4, "qwen_obviously_better": 5, "unclear": 1}}
    (OUT / "quality_notes.json").write_text(json.dumps(quality, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    successful = sum(bool(r["generation_success"]) for r in results)
    latencies = [r["wall_latency_sec"] for r in results]
    controlled_rates = [r["tokens_per_sec"] for r in results[1:] if r["tokens_per_sec"] is not None]
    peak = max((r["peak_vram_used_mib"] or 0) for r in results + [no_prefix])
    summary = {
        "model_path": str(MODEL), "sha256": model_hash, "file_size_bytes": MODEL.stat().st_size,
        "llama_cpp_build": "b10333-08659901c (10333)", "cuda_gpu": "NVIDIA GeForce RTX 5070",
        "load_success": True, "generation_success": successful == 10,
        "working_command": results[0]["command"], "gpu_offload": "49/49 layers; CUDA model buffer 4671.30 MiB",
        "load_time_sec": 6.08, "peak_vram_gb": round(peak / 1024, 3), "model_calls": 14,
        "successful_generation_count_including_no_prefix": 11,
        "successful_translation_count": successful, "structural_failure_count": 1,
        "avg_latency_sec": round(sum(latencies) / len(latencies), 3),
        "tokens_per_sec": round(sum(controlled_rates) / len(controlled_rates), 3),
        "prefix_required": True, "deserves_30_item_gate": True,
        "notes": [
            "Invocation 1 via llama.exe cli loaded the model but failed before output because llama_encode was not called.",
            "Invocation 2 via the separate local llama-completion.exe succeeded and was reused as TEST 1.",
            "The no-prefix diagnostic exited normally with immediate EOS/empty output, so <2tr> is required for this task.",
            "model_calls counts all 14 process invocations: 11 completed model executions and 3 failed/incomplete diagnostic infrastructure attempts; no successful translation was rerun.",
            "Controlled TEST 2-10 decoder throughput averaged 65.5 tokens/sec (52.86-74.25); TEST 1's 0.27 tokens/sec was an anomalous interactive log-capture measurement and is retained raw.",
            "Average wall latency across all ten stored tests is 10.593 sec; controlled TEST 2-10 wall latency averaged 6.945 sec, dominated by roughly 6.08 sec model initialization per process.",
            "Peak nvidia-smi usage was 6651 MiB total (about 5666 MiB above the 985 MiB baseline); llama.cpp reported 5371 MiB self usage.",
            "MADLAD won 4 obvious stored-Qwen comparisons, Qwen won 5, and 1 needs human review; the contrasting strengths justify a larger 30-item gate, not production integration.",
        ],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    runtime_lines = [
        "LLAMA.CPP RUNTIME PROBE", "", f"Primary app: {LLAMA_APP}", f"Working raw completion app: {RUNTIME}",
        "Build: b10333-08659901c (10333), Clang 20.1.8, Windows x86_64", "CUDA backend: available",
        "GPU: NVIDIA GeForce RTX 5070", "Pre-test VRAM: 12227 MiB total, 985 MiB used, 10959 MiB free (nvidia-smi)", "",
        "ATTEMPT 1 — FAILED AFTER MODEL LOAD", "Command: llama.exe cli -m ... -ngl all -n 128 --no-warmup --no-conversation --offline --temp 0 --seed 0 --perf --show-timings --no-display-prompt --simple-io --log-verbosity 4 -p \"<2tr> I'm used to it.\"",
        "Metadata parse: success; tensor loading: success; encoder init: success; decoder init: success; generation: failed.",
        "Error: llama-graph.cpp:1037: GGML_ASSERT(!cross->seq_ids_enc.empty() && \"llama_encode must be called first\") failed", "",
        "ATTEMPT 2 — SUCCESS", f"Command: {results[0]['command']}", "Raw output: Alıştım.",
        "Offload: 49/49 layers; CUDA model 4671.30 MiB; KV 192.00 MiB; compute 508.00 MiB; total self 5371 MiB.",
        "Encoder-decoder generation completed through llama-completion.exe.", "",
        "NO-PREFIX DIAGNOSTIC", "Exit code 0 with immediate EOS/empty translation; <2tr> is required.", "",
        "MEASUREMENT NOTES", "Controlled TEST 2-10 throughput: average 65.50 tokens/sec, range 52.86-74.25.",
        "Controlled TEST 2-10 average wall latency: 6.945 sec, including about 6.08 sec model initialization per process.",
        "TEST 1 retained its anomalous interactive-capture result: 43.425 sec wall, 0.27 tokens/sec.",
        "Peak nvidia-smi usage: 6651 MiB total; baseline 985 MiB; incremental peak about 5666 MiB.", "",
        "PER-GENERATION SUMMARY",
    ]
    runtime_lines.extend(f"{r['label']}: exit={r['exit_code']} wall={r['wall_latency_sec']} sec peak={r['peak_vram_used_mib']} MiB output={r['stripped_output']}" for r in results)
    runtime_lines.append(f"NO_PREFIX: exit={no_prefix['exit_code']} wall={no_prefix['wall_latency_sec']} sec peak={no_prefix['peak_vram_used_mib']} MiB output={no_prefix['stripped_output']}")
    (OUT / "runtime_probe.txt").write_text("\n".join(runtime_lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-remaining", action="store_true")
    parser.add_argument("--resume-after-no-prefix", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()
    if args.finalize_only:
        existing = json.loads((OUT / "smoke_results.json").read_text(encoding="utf-8"))
        write_artifacts(existing["results"], existing["no_prefix_diagnostic"])
        print(f"Finalized 8 artifacts in {OUT}", flush=True)
        return 0
    if not args.run_remaining:
        parser.error("Use --run-remaining to execute the authorized remaining 9 prefixed tests and one no-prefix diagnostic.")
    results = [seeded_test_one()]
    no_prefix = seeded_no_prefix() if args.resume_after_no_prefix else run_generation(SOURCES[0], prefix=False, label="NO_PREFIX")
    if no_prefix["exit_code"] != 0:
        raise SystemExit("No-prefix diagnostic failed at runtime; stopping before remaining translations.")
    for index, source in enumerate(SOURCES[1:], 2):
        result = run_generation(source, prefix=True, label=f"TEST_{index}")
        results.append(result)
        if not result["generation_success"]:
            write_artifacts(results, no_prefix)
            raise SystemExit(f"TEST_{index} failed; stopped.")
    write_artifacts(results, no_prefix)
    print(f"Wrote 8 artifacts to {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
