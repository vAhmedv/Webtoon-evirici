"""MADLAD-400-7B-MT-BT Q6_K compatibility diagnostic (no integration)."""
from __future__ import annotations

import hashlib
import json
import re
import struct
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, BinaryIO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(r"C:\AI\Models\model-q6k.gguf")
LLAMA_EXE = Path(r"C:\AI\llama-cpp-cuda\llama.exe")
OUTPUT_DIR = PROJECT_ROOT / "benchmark_results" / "madlad_7b_mt_bt_q6k_smoke_v1"
OUTPUT_FILES = {
    "model_metadata.json",
    "runtime_probe.txt",
    "smoke_results.json",
    "smoke_results.txt",
    "qwen_vs_madlad_comparison.json",
    "qwen_vs_madlad_comparison.txt",
    "summary.json",
}
OFFICIAL_MODEL_URL = "https://huggingface.co/google/madlad400-7b-mt-bt/blob/main/model-q6k.gguf"
OFFICIAL_SHA256 = "9FA27654A20C1FAC5EE21743F29FDD3CA80EF80DB918C4F3D624DEB35DC95998"
OFFICIAL_SIZE_BYTES = 6807667008

TEST_INPUTS = [
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


def _read_exact(handle: BinaryIO, size: int) -> bytes:
    data = handle.read(size)
    if len(data) != size:
        raise EOFError(f"Expected {size} bytes, got {len(data)}")
    return data


def _unpack(handle: BinaryIO, fmt: str) -> Any:
    return struct.unpack("<" + fmt, _read_exact(handle, struct.calcsize("<" + fmt)))[0]


def _read_string(handle: BinaryIO) -> str:
    length = _unpack(handle, "Q")
    return _read_exact(handle, length).decode("utf-8", errors="replace")


def _read_value(handle: BinaryIO, value_type: int) -> Any:
    scalar_formats = {
        0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i",
        6: "f", 7: "?", 10: "Q", 11: "q", 12: "d",
    }
    if value_type in scalar_formats:
        return _unpack(handle, scalar_formats[value_type])
    if value_type == 8:
        return _read_string(handle)
    if value_type == 9:
        element_type = _unpack(handle, "I")
        count = _unpack(handle, "Q")
        values = [_read_value(handle, element_type) for _ in range(count)]
        if count > 64:
            return {
                "element_type": element_type,
                "count": count,
                "first_values": values[:8],
                "last_values": values[-3:],
            }
        return values
    raise ValueError(f"Unsupported GGUF value type: {value_type}")


def inspect_gguf(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        magic = _read_exact(handle, 4)
        if magic != b"GGUF":
            raise ValueError(f"Invalid GGUF magic: {magic!r}")
        version = _unpack(handle, "I")
        tensor_count = _unpack(handle, "Q")
        metadata_count = _unpack(handle, "Q")
        metadata: dict[str, Any] = {}
        for _ in range(metadata_count):
            key = _read_string(handle)
            metadata[key] = _read_value(handle, _unpack(handle, "I"))

        tensors = []
        for index in range(tensor_count):
            name = _read_string(handle)
            dimensions = _unpack(handle, "I")
            shape = [_unpack(handle, "Q") for _ in range(dimensions)]
            tensor_type = _unpack(handle, "I")
            offset = _unpack(handle, "Q")
            tensors.append(
                {
                    "index": index,
                    "name": name,
                    "name_length_bytes": len(name.encode("utf-8")),
                    "shape": shape,
                    "ggml_type": tensor_type,
                    "offset": offset,
                }
            )

    longest = sorted(tensors, key=lambda item: item["name_length_bytes"], reverse=True)[:10]
    tensor_type_counts = Counter(str(item["ggml_type"]) for item in tensors)
    encoder_tensor_count = sum(item["name"].startswith("encoder.") for item in tensors)
    decoder_tensor_count = sum(item["name"].startswith("decoder.") for item in tensors)
    inferred_architecture = (
        "T5-family encoder-decoder"
        if encoder_tensor_count and decoder_tensor_count
        else None
    )
    return {
        "gguf_magic": magic.decode("ascii"),
        "gguf_version": version,
        "tensor_count": tensor_count,
        "metadata_kv_count": metadata_count,
        "metadata": metadata,
        "architecture": metadata.get("general.architecture"),
        "model_name": metadata.get("general.name"),
        "model_type": metadata.get("general.type"),
        "file_type": metadata.get("general.file_type"),
        "quantization_version": metadata.get("general.quantization_version"),
        "architecture_reported_by_metadata": metadata.get("general.architecture"),
        "architecture_inferred_from_tensor_names": inferred_architecture,
        "architecture_inference_evidence": {
            "encoder_tensor_count": encoder_tensor_count,
            "decoder_tensor_count": decoder_tensor_count,
            "relative_attention_bias_tensors": [
                item["name"] for item in tensors if "relative_attention_bias" in item["name"]
            ],
        },
        "ggml_tensor_type_counts": dict(sorted(tensor_type_counts.items())),
        "q6_k_tensor_type_14_count": tensor_type_counts.get("14", 0),
        "longest_tensor_names": longest,
        "tensor_338": tensors[338] if len(tensors) > 338 else None,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _run(command: list[str], timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )


def _stored_qwen_by_source() -> dict[str, str | None]:
    path = PROJECT_ROOT / "benchmark_results" / "qwen_translation_pipeline_fix_v1" / "qwen_results.json"
    if not path.exists():
        return {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    mapping: dict[str, str | None] = {}
    for row in rows:
        for field in ("v3_selected_english", "normalized_source", "original_accepted_english"):
            value = row.get(field)
            if value:
                mapping[str(value).strip().casefold()] = row.get("final_restored")
    return mapping


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    unexpected = {path.name for path in OUTPUT_DIR.iterdir() if path.is_file()} - OUTPUT_FILES
    if unexpected:
        raise RuntimeError(f"Unexpected existing output files: {sorted(unexpected)}")

    file_size = MODEL_PATH.stat().st_size
    started = time.perf_counter()
    metadata = inspect_gguf(MODEL_PATH)
    metadata.update(
        {
            "model_file_path": str(MODEL_PATH),
            "file_size_bytes": file_size,
            "file_size_gib": round(file_size / (1024 ** 3), 6),
            "file_size_decimal_gb": round(file_size / 1_000_000_000, 6),
            "sha256": _sha256(MODEL_PATH),
            "official_google_model_identity_verified": False,
            "official_model_url": OFFICIAL_MODEL_URL,
        }
    )
    metadata["official_google_model_identity_verified"] = (
        metadata["sha256"] == OFFICIAL_SHA256 and file_size == OFFICIAL_SIZE_BYTES
    )

    version = _run([str(LLAMA_EXE), "--version"], timeout=30)
    devices = _run([str(LLAMA_EXE), "cli", "--list-devices"], timeout=30)
    preflight_command = [
        str(LLAMA_EXE), "cli", "-m", str(MODEL_PATH), "-ngl", "all",
        "-n", "0", "--no-warmup", "--no-conversation", "--offline",
        "--log-verbosity", "4", "--no-display-prompt", "--simple-io",
        "-p", "<2tr> I'm used to it.",
    ]
    probe_started = time.perf_counter()
    probe = _run(preflight_command, timeout=180)
    probe_seconds = round(time.perf_counter() - probe_started, 3)
    combined_probe = "\n".join(
        [
            "MADLAD-400-7B-MT-BT Q6_K — LLAMA.CPP COMPATIBILITY PREFLIGHT",
            "=" * 78,
            f"COMMAND: {subprocess.list2cmdline(preflight_command)}",
            f"EXIT_CODE: {probe.returncode}",
            f"ELAPSED_SECONDS: {probe_seconds}",
            "",
            "LLAMA.CPP VERSION:",
            (version.stdout + version.stderr).strip(),
            "",
            "DEVICES:",
            (devices.stdout + devices.stderr).strip(),
            "",
            "PREFLIGHT STDOUT:",
            probe.stdout.rstrip(),
            "",
            "PREFLIGHT STDERR:",
            probe.stderr.rstrip(),
        ]
    )

    loaded = probe.returncode == 0
    parser_error = re.search(r"tensor name 338 is too long: 68 >= 64", combined_probe) is not None
    blocked_reason = (
        "GGUF_READER_TENSOR_NAME_LIMIT_64_BYTES" if parser_error
        else "UNKNOWN_LLAMA_CPP_MODEL_LOAD_FAILURE"
    ) if not loaded else None

    smoke_rows = [
        {
            "id": index,
            "source": source,
            "exact_model_input": f"<2tr> {source}",
            "raw_model_output": None,
            "final_stripped_output": None,
            "latency_seconds": None,
            "input_token_count": None,
            "generated_token_count": None,
            "tokens_per_second": None,
            "structural_flags": ["NOT_RUN_RUNTIME_BLOCKED"],
        }
        for index, source in enumerate(TEST_INPUTS, 1)
    ]
    no_prefix = {
        "source": "I'm used to it.",
        "exact_model_input": "I'm used to it.",
        "raw_model_output": None,
        "final_stripped_output": None,
        "status": "NOT_RUN_RUNTIME_BLOCKED",
    }
    smoke_payload = {
        "model_loaded_successfully": loaded,
        "model_call_count": 0,
        "maximum_allowed_model_calls": 11,
        "blocked_reason": blocked_reason,
        "items": smoke_rows,
        "no_prefix_diagnostic": no_prefix,
    }

    qwen = _stored_qwen_by_source()
    comparisons = []
    for index, source in enumerate(TEST_INPUTS, 1):
        comparisons.append(
            {
                "id": index,
                "source": source,
                "qwen_fix_v1": qwen.get(source.casefold()),
                "madlad_raw": None,
                "madlad_status": "NOT_RUN_RUNTIME_BLOCKED",
            }
        )

    summary = {
        "decision": "MADLAD_GGUF_SMOKE_BLOCKED",
        "model_file_path": str(MODEL_PATH),
        "file_size_bytes": file_size,
        "file_size_gib": metadata["file_size_gib"],
        "file_size_decimal_gb": metadata["file_size_decimal_gb"],
        "sha256": metadata["sha256"],
        "official_google_model_identity_verified": metadata["official_google_model_identity_verified"],
        "official_model_url": OFFICIAL_MODEL_URL,
        "gguf_architecture": metadata["architecture"],
        "gguf_architecture_status": "NOT_REPORTED_IN_FILE_METADATA",
        "architecture_inferred_from_tensor_names": metadata["architecture_inferred_from_tensor_names"],
        "architecture_inference_evidence": metadata["architecture_inference_evidence"],
        "metadata_kv_count": metadata["metadata_kv_count"],
        "gguf_model_name": metadata["model_name"],
        "llama_cpp_build": (version.stdout + version.stderr).strip(),
        "model_loaded_successfully": loaded,
        "exact_attempted_invocation": subprocess.list2cmdline(preflight_command),
        "working_invocation": None,
        "gpu_device": (devices.stdout + devices.stderr).strip(),
        "gpu_offload_requested": "all layers",
        "gpu_offload_achieved": False,
        "model_load_time_seconds": None,
        "preflight_failure_time_seconds": probe_seconds,
        "peak_vram_mib": None,
        "ram_usage_mib": None,
        "translation_generation_speed_tokens_per_second": None,
        "english_to_turkish_results_completed": 0,
        "english_to_turkish_results": smoke_rows,
        "no_prefix_diagnostic": no_prefix,
        "language_prefix_required": None,
        "language_prefix_works": None,
        "structural_failure_count": None,
        "structural_evaluation_status": "NOT_RUN_RUNTIME_BLOCKED",
        "name_preservation_observations": "Not observable; generation did not start.",
        "terminology_preservation_observations": "Not observable; generation did not start.",
        "pronoun_possessive_observations": "Not observable; generation did not start.",
        "idiom_observations": "Not observable; generation did not start.",
        "stored_qwen_comparison_count": sum(row["qwen_fix_v1"] is not None for row in comparisons),
        "llama_cpp_viable_runtime_for_this_gguf": False,
        "runtime_blocker": blocked_reason,
        "runtime_blocker_detail": "gguf_init_from_reader rejected tensor 338 because its 68-byte name exceeds the build's 64-byte tensor-name limit.",
        "madlad_deserves_larger_30_item_test": False,
        "larger_test_reason": "Resolve GGUF parser/runtime compatibility first, then rerun this 10-item smoke test.",
        "local_alternate_gguf_runtime_found": False,
        "transformers_installed_but_not_used": True,
        "project_integration_performed": False,
        "production_code_changed_by_this_task": False,
        "commit_performed": False,
        "push_performed": False,
        "model_downloads_performed": False,
        "model_calls_performed": 0,
        "total_diagnostic_seconds": round(time.perf_counter() - started, 3),
    }

    (OUTPUT_DIR / "model_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUTPUT_DIR / "runtime_probe.txt").write_text(combined_probe, encoding="utf-8")
    (OUTPUT_DIR / "smoke_results.json").write_text(
        json.dumps(smoke_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    smoke_lines = [
        "MADLAD RAW TRANSLATION SMOKE RESULTS",
        "=" * 78,
        f"STATUS: {blocked_reason}",
        "MODEL CALLS: 0 / 11",
        "All 10 translation items and the no-prefix diagnostic were skipped because model loading failed.",
    ]
    (OUTPUT_DIR / "smoke_results.txt").write_text("\n".join(smoke_lines), encoding="utf-8")
    (OUTPUT_DIR / "qwen_vs_madlad_comparison.json").write_text(
        json.dumps(comparisons, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    comparison_lines = ["STORED QWEN FIX V1 vs MADLAD RAW", "=" * 78]
    for row in comparisons:
        comparison_lines.extend(
            [
                f"[{row['id']}] SOURCE: {row['source']}",
                f"QWEN FIX V1: {row['qwen_fix_v1'] or '(no exact stored overlap)'}",
                "MADLAD RAW: (not run — runtime blocked)",
                "-" * 78,
            ]
        )
    (OUTPUT_DIR / "qwen_vs_madlad_comparison.txt").write_text(
        "\n".join(comparison_lines), encoding="utf-8"
    )
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    actual = {path.name for path in OUTPUT_DIR.iterdir() if path.is_file()}
    if actual != OUTPUT_FILES:
        raise RuntimeError(f"Output contract mismatch: {sorted(actual)}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.exit(main())
