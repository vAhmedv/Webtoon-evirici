from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import re
import statistics
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "benchmark_results" / "hy_mt2_vs_karga_vs_qwen_real30_v1"
DATASET_PATH = OUTPUT_DIR / "dataset.json"
STORED_QWEN_PATH = OUTPUT_DIR / "stored_qwen_results.json"
ORIGINAL_QWEN_PATH = (
    BASE_DIR / "benchmark_results" / "qwen_translation_pipeline_fix_v1" / "qwen_results.json"
)
RUNTIME_PROBE_PATH = OUTPUT_DIR / "runtime_probe.txt"
LLAMA_SERVER_EXE = Path(r"C:\AI\llama-cpp-cuda\llama-server.exe")
HY_MODEL_PATH = Path(r"C:\AI\Models\HY-MT2-7B-Q8_0.gguf")
KARGA_MODEL_PATH = Path(r"C:\AI\Models\Karga-DPO-v0.1.Q6_K.gguf")

HY_PROMPT = (
    "<|startoftext|>Translate the following text into Turkish. Note that you "
    "should only output the translated result without any additional explanation:\n"
    "{source}<|extra_0|>"
)
KARGA_PROMPT = (
    "<|im_start|>system\nTask: Translation.<|im_end|>\n"
    "<|im_start|>user\n{source}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

SERVER_COMMON_ARGS = [
    "-ngl",
    "99",
    "-c",
    "2048",
    "-np",
    "1",
    "--host",
    "127.0.0.1",
    "--no-webui",
    "--metrics",
]
DECODING_SETTINGS = {
    "temperature": 0.0,
    "top_k": 1,
    "top_p": 1.0,
    "seed": 0,
    "n_predict": 128,
    "stream": False,
    "cache_prompt": False,
}

GGUF_VALUE_TYPES = {
    0: ("B", 1),
    1: ("b", 1),
    2: ("H", 2),
    3: ("h", 2),
    4: ("I", 4),
    5: ("i", 4),
    6: ("f", 4),
    7: ("?", 1),
    10: ("Q", 8),
    11: ("q", 8),
    12: ("d", 8),
}
GGUF_FILE_TYPES = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    7: "Q8_0",
    8: "Q5_0",
    9: "Q5_1",
    10: "Q2_K",
    11: "Q3_K_S",
    12: "Q3_K_M",
    13: "Q3_K_L",
    14: "Q4_K_S",
    15: "Q4_K_M",
    16: "Q5_K_S",
    17: "Q5_K_M",
    18: "Q6_K",
}
GGML_TYPES = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    6: "Q5_0",
    7: "Q5_1",
    8: "Q8_0",
    9: "Q8_1",
    10: "Q2_K",
    11: "Q3_K",
    12: "Q4_K",
    13: "Q5_K",
    14: "Q6_K",
    15: "Q8_K",
    16: "IQ2_XXS",
    17: "IQ2_XS",
    18: "IQ3_XXS",
    19: "IQ1_S",
    20: "IQ4_NL",
    21: "IQ3_S",
    22: "IQ2_S",
    23: "IQ4_XS",
    24: "I8",
    25: "I16",
    26: "I32",
    27: "I64",
    28: "F64",
    30: "BF16",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def json_dump(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def read_exact(handle: Any, size: int) -> bytes:
    data = handle.read(size)
    if len(data) != size:
        raise EOFError(f"Expected {size} bytes, got {len(data)}")
    return data


def read_u32(handle: Any) -> int:
    return struct.unpack("<I", read_exact(handle, 4))[0]


def read_u64(handle: Any) -> int:
    return struct.unpack("<Q", read_exact(handle, 8))[0]


def read_gguf_string(handle: Any) -> str:
    size = read_u64(handle)
    return read_exact(handle, size).decode("utf-8", errors="replace")


def read_gguf_value(handle: Any, value_type: int, keep_array_values: bool = False) -> Any:
    if value_type in GGUF_VALUE_TYPES:
        fmt, size = GGUF_VALUE_TYPES[value_type]
        return struct.unpack("<" + fmt, read_exact(handle, size))[0]
    if value_type == 8:
        return read_gguf_string(handle)
    if value_type == 9:
        element_type = read_u32(handle)
        length = read_u64(handle)
        if keep_array_values and length <= 256:
            return [read_gguf_value(handle, element_type, True) for _ in range(length)]
        for _ in range(length):
            read_gguf_value(handle, element_type, False)
        return {
            "array_element_type": element_type,
            "length": length,
            "values_omitted": True,
        }
    raise ValueError(f"Unsupported GGUF metadata value type: {value_type}")


def inspect_gguf(path: Path) -> dict[str, Any]:
    tensor_type_counts: Counter[str] = Counter()
    metadata: dict[str, Any] = {}
    with path.open("rb") as handle:
        magic = read_exact(handle, 4)
        if magic != b"GGUF":
            raise ValueError(f"Not a GGUF file: {path}")
        version = read_u32(handle)
        tensor_count = read_u64(handle)
        kv_count = read_u64(handle)
        for _ in range(kv_count):
            key = read_gguf_string(handle)
            value_type = read_u32(handle)
            metadata[key] = read_gguf_value(handle, value_type)

        parameter_count = 0
        for _ in range(tensor_count):
            read_gguf_string(handle)
            dimension_count = read_u32(handle)
            dimensions = [read_u64(handle) for _ in range(dimension_count)]
            tensor_type = read_u32(handle)
            read_u64(handle)  # tensor data offset
            parameter_count += math.prod(dimensions)
            tensor_type_counts[GGML_TYPES.get(tensor_type, f"TYPE_{tensor_type}")] += 1

    tokenizer_metadata = {
        key: value for key, value in metadata.items() if key.startswith("tokenizer.")
    }
    architecture = metadata.get("general.architecture")
    block_count = metadata.get(f"{architecture}.block_count") if architecture else None
    context_length = metadata.get(f"{architecture}.context_length") if architecture else None
    file_type = metadata.get("general.file_type")
    chat_template = metadata.get("tokenizer.chat_template")
    if chat_template is None:
        template_keys = sorted(
            key for key in metadata if key.startswith("tokenizer.chat_template")
        )
        chat_template = {key: metadata[key] for key in template_keys} or None
    return {
        "model_path": str(path),
        "byte_size": path.stat().st_size,
        "sha256": sha256_file(path),
        "gguf_version": version,
        "gguf_architecture": architecture,
        "model_name": metadata.get("general.name"),
        "quantization": GGUF_FILE_TYPES.get(file_type, f"FILE_TYPE_{file_type}"),
        "general_file_type": file_type,
        "quantization_version": metadata.get("general.quantization_version"),
        "parameter_count": parameter_count,
        "parameter_count_billions": round(parameter_count / 1_000_000_000, 4),
        "tensor_count": tensor_count,
        "metadata_kv_count": kv_count,
        "block_count": block_count,
        "context_length": context_length,
        "tensor_type_counts": dict(sorted(tensor_type_counts.items())),
        "tokenizer_metadata": tokenizer_metadata,
        "embedded_chat_template": chat_template,
    }


def query_vram_mib() -> float | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
            timeout=10,
        )
        return float(result.stdout.strip().splitlines()[0])
    except Exception:
        return None


class VramMonitor:
    def __init__(self) -> None:
        self.initial_mib = query_vram_mib()
        self.peak_mib = self.initial_mib
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(0.20):
            current = query_vram_mib()
            if current is not None and (self.peak_mib is None or current > self.peak_mib):
                self.peak_mib = current

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def summary(self) -> dict[str, float | None]:
        delta = None
        if self.initial_mib is not None and self.peak_mib is not None:
            delta = self.peak_mib - self.initial_mib
        return {
            "initial_vram_mib": self.initial_mib,
            "peak_vram_mib": self.peak_mib,
            "delta_vram_mib": delta,
            "initial_vram_gib": round(self.initial_mib / 1024, 3)
            if self.initial_mib is not None
            else None,
            "peak_vram_gib": round(self.peak_mib / 1024, 3)
            if self.peak_mib is not None
            else None,
            "delta_vram_gib": round(delta / 1024, 3) if delta is not None else None,
        }


class ServerProcess:
    def __init__(self, model_path: Path, port: int, log_path: Path) -> None:
        self.model_path = model_path
        self.port = port
        self.log_path = log_path
        self.process: subprocess.Popen[str] | None = None
        self.lines: list[str] = []
        self._reader: threading.Thread | None = None
        self._log_handle: Any = None
        self.load_time_sec: float | None = None
        self.command = [
            str(LLAMA_SERVER_EXE),
            "-m",
            str(model_path),
            *SERVER_COMMON_ARGS,
            "--port",
            str(port),
        ]

    def start(self, timeout_sec: float = 180.0) -> None:
        self._log_handle = self.log_path.open("w", encoding="utf-8")
        started = time.perf_counter()
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self.process = subprocess.Popen(
            self.command,
            cwd=str(LLAMA_SERVER_EXE.parent),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=creationflags,
        )
        self._reader = threading.Thread(target=self._read_output, daemon=True)
        self._reader.start()
        deadline = time.perf_counter() + timeout_sec
        last_error = "server did not become ready"
        while time.perf_counter() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"llama-server exited with code {self.process.returncode}: "
                    + "\n".join(self.lines[-40:])
                )
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{self.port}/health", timeout=1.0
                ) as response:
                    payload = response.read().decode("utf-8", errors="replace")
                    if response.status == 200 and "ok" in payload.lower():
                        self.load_time_sec = time.perf_counter() - started
                        return
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = str(exc)
            time.sleep(0.25)
        raise TimeoutError(f"llama-server readiness timeout: {last_error}")

    def _read_output(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        assert self._log_handle is not None
        for line in self.process.stdout:
            clean = line.rstrip("\r\n")
            self.lines.append(clean)
            self._log_handle.write(line)
            self._log_handle.flush()

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        if self._reader is not None:
            self._reader.join(timeout=5)
        if self._log_handle is not None:
            self._log_handle.close()

    def runtime_buffers(self) -> dict[str, Any]:
        relevant = [
            line
            for line in self.lines
            if any(
                marker in line
                for marker in (
                    "model buffer size",
                    "KV buffer size",
                    "compute buffer size",
                    "offloaded",
                    "offloading",
                    "CUDA0",
                )
            )
        ]
        return {"diagnostic_lines": relevant}


def completion_request(port: int, prompt: str) -> tuple[dict[str, Any], float]:
    payload = {"prompt": prompt, **DECODING_SETTINGS}
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=180) as response:
        raw_body = response.read().decode("utf-8", errors="replace")
    latency = time.perf_counter() - started
    data = json.loads(raw_body)
    if "content" not in data:
        raise RuntimeError(f"Completion response has no content field: {raw_body[:1000]}")
    return data, latency


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def extract_numbers(text: str) -> list[str]:
    return re.findall(r"(?<!\w)\d+(?!\w)", text)


def extract_bracket_tokens(text: str) -> list[str]:
    return re.findall(r"\[[^\]\r\n]+\]", text)


def detect_structural_flags(source: str, translation: str) -> tuple[list[str], list[str]]:
    flags: list[str] = []
    hints: list[str] = []
    stripped = translation.strip()
    if not stripped:
        return ["EMPTY_OUTPUT"], hints

    source_normalized = re.sub(r"\s+", " ", source).strip().casefold()
    output_normalized = re.sub(r"\s+", " ", stripped).strip().casefold()
    if output_normalized == source_normalized:
        flags.append("SOURCE_COPY")

    prompt_markers = (
        "translate the following text",
        "output the translated result",
        "<|im_start|>",
        "<|im_end|>",
        "<|extra_0|>",
        "task: translation",
    )
    if any(marker in output_normalized for marker in prompt_markers):
        flags.append("PROMPT_ECHO")

    if re.search(
        r"^(?:işte\b|here(?:'s| is)\b|türkçe çeviri\s*:|çeviri\s*:|translation\s*:)",
        stripped,
        re.IGNORECASE,
    ):
        flags.append("EXPLANATION_WRAPPER")

    if re.search(
        r"(?:^|\n)\s*(?:1[.)]|2[.)]|alternatif\s*:|alternative\s*:)",
        stripped,
        re.IGNORECASE,
    ):
        flags.append("MULTIPLE_ALTERNATIVES")

    analysis_text = stripped.replace("\\n", "\n")
    analysis_normalized = analysis_text.casefold()
    english_function_words = set(
        re.findall(
            r"\b(?:the|and|that|this|with|from|have|has|was|were|will|would|"
            r"could|should|into|your|you|they|their|what|when|where|which|for|not)\b",
            output_normalized,
        )
    )
    turkish_markers = set(
        re.findall(
            r"\b(?:bir|ve|bu|şu|için|ile|değil|ama|çok|daha|gibi|olan|olarak|"
            r"sen|siz|ben|biz|onlar|mı|mi|mu|mü)\b",
            output_normalized,
        )
    )
    if len(english_function_words) >= 3 and not turkish_markers:
        flags.append("WRONG_LANGUAGE")
    elif len(english_function_words) >= 2:
        flags.append("ENGLISH_PROSE_LEAK")

    source_words = set(re.findall(r"\b[a-z]{5,}\b", source.casefold()))
    output_words = set(re.findall(r"\b[a-z]{5,}\b", output_normalized))
    protected_words = {
        "allen",
        "craft",
        "emma",
        "hahaha",
        "lucas",
        "tian",
        "vrmmo",
        "yuan",
    }
    if (source_words & output_words) - protected_words:
        flags.append("ENGLISH_PROSE_LEAK")

    if Counter(extract_numbers(source)) != Counter(extract_numbers(stripped)):
        flags.append("NUMBER_CHANGE")
    if Counter(extract_bracket_tokens(source)) != Counter(extract_bracket_tokens(stripped)):
        flags.append("BRACKET_TOKEN_LOSS")

    known_names = re.findall(
        r"\b(?:Yli|Gao\s+Yuan|Lho\s+Tian|Allen-san|Emma|Lucas|VRMMO)\b",
        source,
        re.IGNORECASE,
    )
    for name in known_names:
        if name.casefold() not in output_normalized:
            flags.append("NAME_CORRUPTION")
            break

    words_for_repetition = re.findall(r"\w+", analysis_normalized)
    repeated_four_grams = Counter(
        tuple(words_for_repetition[index : index + 4])
        for index in range(max(0, len(words_for_repetition) - 3))
    )
    has_repeated_phrase = any(count >= 3 for count in repeated_four_grams.values())
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", analysis_text) if part.strip()]
    if len(paragraphs) >= 2:
        flags.append("MULTIPLE_ALTERNATIVES")
    if (
        "\ufffd" in stripped
        or re.search(r"(.)\1{8,}", stripped)
        or (len(stripped) > max(120, len(source) * 2.5) and has_repeated_phrase)
    ):
        flags.append("GARBLED_OUTPUT")

    source_lower = source.casefold()
    output_lower = stripped.casefold()
    if re.search(r"\b(?:my|mine)\b", source_lower) and not re.search(
        r"\b(?:benim|bana|adım|param|alanım|dileğim|istediğim)\b|\w+(?:ım|im|um|üm)\b",
        output_lower,
    ):
        hints.append("POSSESSIVE_CHANGE")
    source_has_negation = bool(
        re.search(r"\b(?:not|never|no|nothing|nobody|isn't|wasn't|weren't|can't|won't)\b", source_lower)
        or "n't" in source_lower
        or re.search(r"\b(?:impossible|useless|fearless|without|\w+less)\b", source_lower)
    )
    output_has_negation = bool(
        re.search(
            r"\b(?:değil\w*|yok|asla|hiç|ölümsüz|imkansız|imkânsız|korkusuz|yaramaz\w*)\b",
            output_lower,
        )
        or re.search(r"\w+(?:sız|siz|suz|süz)\b", output_lower)
        or re.search(r"\w+(?:ma|me|maz|mez|madı|medi|mamış|memiş|mıyor|miyor|muyor|müyor)\b", output_lower)
    )
    if source_has_negation and not output_has_negation:
        hints.append("NEGATION_CHANGE")
    return sorted(set(flags)), sorted(set(hints))


def parse_existing_probe(path: Path) -> dict[str, dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    result: dict[str, dict[str, Any]] = {}
    patterns = {
        "hy_mt2": r"--- HY-MT2 ITEM #1 PROBE ---\s*(.*?)(?=\n---|\Z)",
        "karga": r"--- KARGA-DPO ITEM #1 PROBE ---\s*(.*?)(?=\n---|\Z)",
    }
    for model, pattern in patterns.items():
        match = re.search(pattern, text, re.DOTALL)
        if not match:
            raise ValueError(f"Existing ITEM 001 probe missing for {model}")
        block = match.group(1)
        raw_match = re.search(r"^Raw Output:\s*(.+)$", block, re.MULTILINE)
        if not raw_match:
            raise ValueError(f"Existing ITEM 001 raw output missing for {model}")
        raw_output = ast.literal_eval(raw_match.group(1).strip())

        def number(label: str) -> float | int | None:
            value_match = re.search(rf"^{re.escape(label)}:\s*([\d.]+)", block, re.MULTILINE)
            if not value_match:
                return None
            value = value_match.group(1)
            return float(value) if "." in value else int(value)

        result[model] = {
            "raw_output": raw_output,
            "stripped": raw_output.strip(),
            "latency_sec": number("Latency"),
            "prompt_tokens": number("Prompt Tokens"),
            "completion_tokens": number("Completion Tokens"),
            "tokens_per_sec": None,
            "probe_load_time_sec": number("Load Time"),
            "probe_peak_vram_gib": number("Peak VRAM"),
            "reused_existing_compatibility_probe": True,
        }
    return result


def validate_dataset() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], str]:
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    qwen_records = json.loads(STORED_QWEN_PATH.read_text(encoding="utf-8"))
    if len(dataset) != 30:
        raise ValueError(f"dataset.json has {len(dataset)} items; expected 30")
    item_ids = [item.get("item_id") for item in dataset]
    if len(set(item_ids)) != 30 or None in item_ids:
        raise ValueError("dataset.json item IDs are missing or duplicated")
    if any(not str(item.get("translation_input", "")).strip() for item in dataset):
        raise ValueError("dataset.json contains an empty translation_input")
    qwen_by_id = {item["item_id"]: item for item in qwen_records}
    if len(qwen_records) != 30 or len(qwen_by_id) != 30:
        raise ValueError("stored_qwen_results.json does not contain 30 unique items")
    if set(item_ids) != set(qwen_by_id):
        raise ValueError("Stored Qwen IDs do not map 1:1 to the benchmark dataset")
    for item in dataset:
        qwen = qwen_by_id[item["item_id"]]
        if qwen.get("translation_input") != item.get("translation_input"):
            raise ValueError(f"Qwen translation_input mismatch: {item['item_id']}")
        item["qwen_fix_v1"] = qwen["output"]
    return dataset, qwen_by_id, sha256_file(DATASET_PATH)


def result_record(
    item: dict[str, Any],
    raw_output: str,
    latency_sec: float | None,
    response: dict[str, Any] | None,
    reused_probe: bool,
    technical_error: str | None = None,
) -> dict[str, Any]:
    stripped = raw_output.strip()
    flags, hints = detect_structural_flags(item["translation_input"], stripped)
    if technical_error and "EMPTY_OUTPUT" not in flags:
        flags.append("EMPTY_OUTPUT")
    timings = (response or {}).get("timings") or {}
    completion_tokens = (response or {}).get("tokens_predicted")
    if completion_tokens is None:
        completion_tokens = timings.get("predicted_n")
    prompt_tokens = (response or {}).get("tokens_evaluated")
    if prompt_tokens is None:
        prompt_tokens = timings.get("prompt_n")
    tokens_per_sec = timings.get("predicted_per_second")
    return {
        "item_id": item["item_id"],
        "series": item["series"],
        "chapter": item["chapter"],
        "original_source": item["original_source"],
        "translation_input": item["translation_input"],
        "raw_output": raw_output,
        "stripped_translation": stripped,
        "latency_sec": round(latency_sec, 6) if latency_sec is not None else None,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "tokens_per_sec": round(tokens_per_sec, 4) if tokens_per_sec is not None else None,
        "flags": sorted(set(flags)),
        "review_hints": hints,
        "reused_existing_compatibility_probe": reused_probe,
        "technical_error": technical_error,
        "api_stop_type": (response or {}).get("stop_type"),
        "api_stopped_eos": (response or {}).get("stopped_eos"),
    }


def run_model(
    model_key: str,
    display_name: str,
    model_path: Path,
    prompt_template: str,
    port: int,
    dataset: list[dict[str, Any]],
    probe: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    print(f"\n=== {display_name}: loading one persistent llama-server process ===", flush=True)
    existing_flags, existing_hints = detect_structural_flags(
        dataset[0]["translation_input"], probe["stripped"]
    )
    item_one = {
        "item_id": dataset[0]["item_id"],
        "series": dataset[0]["series"],
        "chapter": dataset[0]["chapter"],
        "original_source": dataset[0]["original_source"],
        "translation_input": dataset[0]["translation_input"],
        "raw_output": probe["raw_output"],
        "stripped_translation": probe["stripped"],
        "latency_sec": probe["latency_sec"],
        "prompt_tokens": probe["prompt_tokens"],
        "completion_tokens": probe["completion_tokens"],
        "tokens_per_sec": probe["tokens_per_sec"],
        "flags": existing_flags,
        "review_hints": existing_hints,
        "reused_existing_compatibility_probe": True,
        "technical_error": None,
        "api_stop_type": None,
        "api_stopped_eos": None,
    }
    results = [item_one]
    attempts = 0
    technical_failures = 0
    server_start_attempts = 0
    server_errors: list[str] = []
    server: ServerProcess | None = None
    monitor: VramMonitor | None = None
    model_wall_started = time.perf_counter()
    runtime_buffers: dict[str, Any] = {"diagnostic_lines": []}
    current_load_time: float | None = None

    for startup_attempt in range(1, 4):
        server_start_attempts = startup_attempt
        monitor = VramMonitor()
        monitor.start()
        server = ServerProcess(
            model_path,
            port,
            OUTPUT_DIR / f"{model_key}_llama_server.log",
        )
        try:
            server.start()
            current_load_time = server.load_time_sec
            break
        except Exception as exc:
            server_errors.append(f"attempt {startup_attempt}: {exc}")
            server.stop()
            monitor.stop()
            server = None
            monitor = None
            if startup_attempt < 3:
                time.sleep(1)

    if server is not None and monitor is not None:
        try:
            for index, item in enumerate(dataset[1:], start=2):
                prompt = prompt_template.format(source=item["translation_input"])
                response: dict[str, Any] | None = None
                latency: float | None = None
                error: str | None = None
                for call_attempt in range(1, 4):
                    attempts += 1
                    try:
                        response, latency = completion_request(port, prompt)
                        error = None
                        break
                    except Exception as exc:
                        technical_failures += 1
                        error = f"attempt {call_attempt}: {type(exc).__name__}: {exc}"
                        if call_attempt < 3:
                            time.sleep(0.5)
                raw_output = str((response or {}).get("content", ""))
                record = result_record(
                    item,
                    raw_output,
                    latency,
                    response,
                    reused_probe=False,
                    technical_error=error,
                )
                results.append(record)
                print(
                    f"[{index:02d}/30] {display_name} {record['latency_sec']}s: "
                    f"{record['stripped_translation']}",
                    flush=True,
                )
        finally:
            server.stop()
            monitor.stop()
            runtime_buffers = server.runtime_buffers()

    if len(results) < 30:
        existing_ids = {record["item_id"] for record in results}
        error_text = "; ".join(server_errors) or "llama-server runtime unavailable"
        for item in dataset:
            if item["item_id"] not in existing_ids:
                results.append(
                    result_record(
                        item,
                        "",
                        None,
                        None,
                        reused_probe=False,
                        technical_error=error_text,
                    )
                )
        results.sort(key=lambda record: [item["item_id"] for item in dataset].index(record["item_id"]))

    model_wall_time = time.perf_counter() - model_wall_started
    vram = monitor.summary() if monitor is not None else {
        "initial_vram_mib": None,
        "peak_vram_mib": None,
        "delta_vram_mib": None,
        "initial_vram_gib": None,
        "peak_vram_gib": None,
        "delta_vram_gib": None,
    }
    successful = [record for record in results if record["stripped_translation"] and not record["technical_error"]]
    live_successful = [record for record in successful if not record["reused_existing_compatibility_probe"]]
    steady_latencies = [
        record["latency_sec"]
        for record in live_successful
        if record["latency_sec"] is not None
    ]
    all_latencies = [
        record["latency_sec"]
        for record in successful
        if record["latency_sec"] is not None
    ]
    completion_tokens = [
        record["completion_tokens"]
        for record in successful
        if isinstance(record["completion_tokens"], int)
    ]
    timed_tps = [
        record["tokens_per_sec"]
        for record in live_successful
        if isinstance(record["tokens_per_sec"], (int, float))
    ]
    objective_flags = sum(len(record["flags"]) for record in results)
    server_command = server.command if server is not None else [
        str(LLAMA_SERVER_EXE),
        "-m",
        str(model_path),
        *SERVER_COMMON_ARGS,
        "--port",
        str(port),
    ]
    performance = {
        "model_name": display_name,
        "load_success": server is not None,
        "generation_success": len(successful) == 30,
        "benchmark_model_calls": 30,
        "existing_probe_calls_reused": 1,
        "new_http_generation_attempts_this_run": attempts,
        "technical_retry_attempts": technical_failures,
        "successful_outputs": len(successful),
        "failed_outputs": 30 - len(successful),
        "empty_outputs": sum(not record["stripped_translation"] for record in results),
        "structural_failure_count": objective_flags,
        "current_server_load_time_sec": round(current_load_time, 4)
        if current_load_time is not None
        else None,
        "existing_item1_probe_load_time_sec": probe["probe_load_time_sec"],
        "first_call_warmup_latency_sec": probe["latency_sec"],
        "translation_wall_time_sec": round(sum(steady_latencies), 4)
        if steady_latencies
        else None,
        "total_model_wall_time_sec": round(model_wall_time, 4),
        "avg_item_latency_excluding_item1_sec": round(statistics.mean(steady_latencies), 4)
        if steady_latencies
        else None,
        "avg_item_latency_all_sec": round(statistics.mean(all_latencies), 4)
        if all_latencies
        else None,
        "median_item_latency_sec": round(statistics.median(all_latencies), 4)
        if all_latencies
        else None,
        "p95_item_latency_sec": round(percentile(all_latencies, 0.95), 4)
        if all_latencies
        else None,
        "total_output_tokens": sum(completion_tokens) if completion_tokens else None,
        "tokens_per_sec": round(statistics.mean(timed_tps), 4) if timed_tps else None,
        **vram,
        "existing_item1_probe_peak_vram_gib": probe["probe_peak_vram_gib"],
        "gpu_offload": "-ngl 99 (maximum requested; exact offload in server log)",
        "runtime_executable": str(LLAMA_SERVER_EXE),
        "runtime_version": "llama.cpp b10333 (08659901c)",
        "server_command": server_command,
        "server_start_attempts": server_start_attempts,
        "server_errors": server_errors,
        "server_runtime_buffers": runtime_buffers,
        "endpoint": "/completion (raw native prompt)",
        "prompt_template": prompt_template,
        "decoding_settings": DECODING_SETTINGS,
    }
    return results, refresh_performance_metrics(performance, results)


def refresh_performance_metrics(
    performance: dict[str, Any], results: list[dict[str, Any]]
) -> dict[str, Any]:
    successful = [
        record
        for record in results
        if record["stripped_translation"] and not record["technical_error"]
    ]
    live_successful = [
        record
        for record in successful
        if not record["reused_existing_compatibility_probe"]
    ]
    steady_successful = live_successful[1:] if live_successful else []
    all_latencies = [
        record["latency_sec"]
        for record in successful
        if record["latency_sec"] is not None
    ]
    steady_latencies = [
        record["latency_sec"]
        for record in steady_successful
        if record["latency_sec"] is not None
    ]
    steady_tps = [
        record["tokens_per_sec"]
        for record in steady_successful
        if isinstance(record["tokens_per_sec"], (int, float))
    ]
    completion_tokens = [
        record["completion_tokens"]
        for record in successful
        if isinstance(record["completion_tokens"], int)
    ]
    performance.update(
        {
            "model_calls": 30,
            "successful_outputs": len(successful),
            "failed_outputs": 30 - len(successful),
            "structural_failure_count": sum(len(record["flags"]) for record in results),
            "load_time_sec": performance.get("current_server_load_time_sec"),
            "current_server_first_call_warmup_latency_sec": live_successful[0]["latency_sec"]
            if live_successful
            else None,
            "avg_latency_sec": round(statistics.mean(steady_latencies), 4)
            if steady_latencies
            else None,
            "avg_item_latency_excluding_warmup_sec": round(
                statistics.mean(steady_latencies), 4
            )
            if steady_latencies
            else None,
            "median_latency_sec": round(statistics.median(all_latencies), 4)
            if all_latencies
            else None,
            "p95_latency_sec": round(percentile(all_latencies, 0.95), 4)
            if all_latencies
            else None,
            "total_output_tokens": sum(completion_tokens) if completion_tokens else None,
            "tokens_per_sec": round(statistics.mean(steady_tps), 4) if steady_tps else None,
            "peak_vram_gib": performance.get("peak_vram_gib"),
        }
    )
    return performance


def write_individual_results(model_key: str, results: list[dict[str, Any]]) -> None:
    json_dump(OUTPUT_DIR / f"{model_key}_results.json", results)
    text_lines = []
    for result in results:
        text_lines.extend(
            [
                "=" * 60,
                f"ITEM ID: {result['item_id']}",
                f"RAW: {result['raw_output']}",
                f"STRIPPED: {result['stripped_translation']}",
                f"FLAGS: {', '.join(result['flags']) or 'NONE'}",
                f"REVIEW HINTS: {', '.join(result['review_hints']) or 'NONE'}",
                "",
            ]
        )
    (OUTPUT_DIR / f"{model_key}_results.txt").write_text(
        "\n".join(text_lines), encoding="utf-8"
    )


def build_review_artifacts(
    dataset: list[dict[str, Any]],
    hy_results: list[dict[str, Any]],
    karga_results: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    hy_by_id = {record["item_id"]: record for record in hy_results}
    karga_by_id = {record["item_id"]: record for record in karga_results}
    three_way: list[dict[str, Any]] = []
    review_blocks: list[str] = []
    three_way_text: list[str] = []
    for index, item in enumerate(dataset, start=1):
        hy = hy_by_id[item["item_id"]]
        karga = karga_by_id[item["item_id"]]
        entry = {
            "item_id": item["item_id"],
            "series": item["series"],
            "chapter": item["chapter"],
            "original_source": item["original_source"],
            "translation_input": item["translation_input"],
            "qwen_fix_v1": item["qwen_fix_v1"],
            "hy_mt2": {
                "raw": hy["raw_output"],
                "stripped": hy["stripped_translation"],
                "flags": hy["flags"],
                "review_hints": hy["review_hints"],
                "latency_sec": hy["latency_sec"],
            },
            "karga": {
                "raw": karga["raw_output"],
                "stripped": karga["stripped_translation"],
                "flags": karga["flags"],
                "review_hints": karga["review_hints"],
                "latency_sec": karga["latency_sec"],
            },
        }
        three_way.append(entry)
        hy_flags = ", ".join(hy["flags"] + hy["review_hints"]) or "NONE"
        karga_flags = ", ".join(karga["flags"] + karga["review_hints"]) or "NONE"
        review_blocks.append(
            "\n".join(
                [
                    "=" * 60,
                    f"ITEM {index:03d}",
                    "",
                    "Series:",
                    item["series"],
                    "Chapter:",
                    item["chapter"],
                    "",
                    "ORIGINAL SOURCE:",
                    item["original_source"],
                    "",
                    "TRANSLATION INPUT:",
                    item["translation_input"],
                    "",
                    "QWEN FIX V1:",
                    item["qwen_fix_v1"],
                    "",
                    "HY-MT2 Q8_0:",
                    hy["stripped_translation"],
                    "",
                    "KARGA-DPO Q6_K:",
                    karga["stripped_translation"],
                    "",
                    "HY-MT2 FLAGS:",
                    hy_flags,
                    "",
                    "KARGA FLAGS:",
                    karga_flags,
                    "=" * 60,
                ]
            )
        )
        three_way_text.extend(
            [
                "=" * 60,
                f"ITEM {index:03d} — {item['item_id']}",
                f"QWEN FIX V1: {item['qwen_fix_v1']}",
                f"HY-MT2 Q8_0: {hy['stripped_translation']}",
                f"KARGA-DPO Q6_K: {karga['stripped_translation']}",
                "",
            ]
        )

    json_dump(OUTPUT_DIR / "three_way_results.json", three_way)
    (OUTPUT_DIR / "three_way_results.txt").write_text(
        "\n".join(three_way_text), encoding="utf-8"
    )
    (OUTPUT_DIR / "assistant_review_pack.txt").write_text(
        "\n\n".join(review_blocks) + "\n", encoding="utf-8"
    )

    structural_flags = {
        "hy_mt2": {
            "total_structural_failures": sum(len(item["hy_mt2"]["flags"]) for item in three_way),
            "objective_flag_breakdown": dict(
                sorted(Counter(flag for item in three_way for flag in item["hy_mt2"]["flags"]).items())
            ),
            "review_hint_breakdown": dict(
                sorted(
                    Counter(
                        flag for item in three_way for flag in item["hy_mt2"]["review_hints"]
                    ).items()
                )
            ),
        },
        "karga": {
            "total_structural_failures": sum(len(item["karga"]["flags"]) for item in three_way),
            "objective_flag_breakdown": dict(
                sorted(Counter(flag for item in three_way for flag in item["karga"]["flags"]).items())
            ),
            "review_hint_breakdown": dict(
                sorted(
                    Counter(
                        flag for item in three_way for flag in item["karga"]["review_hints"]
                    ).items()
                )
            ),
        },
        "note": "Review hints are automatic suspicions, not definitive semantic judgments.",
    }
    json_dump(OUTPUT_DIR / "structural_flags.json", structural_flags)
    return three_way, structural_flags


def write_runtime_probe(
    original_probe_text: str,
    hy_performance: dict[str, Any],
    karga_performance: dict[str, Any],
) -> None:
    text = "\n".join(
        [
            "=== LLAMA.CPP RUNTIME COMPATIBILITY AND RESUME RECORD ===",
            f"Runtime: {LLAMA_SERVER_EXE}",
            "Runtime version: llama.cpp b10333 (08659901c)",
            "Execution: one persistent server per model; models loaded sequentially",
            "ITEM 001 outputs below were reused from the existing successful compatibility probes.",
            "Items 002-030 were generated through the persistent /completion server endpoint.",
            "",
            "--- CURRENT HY-MT2 SERVER RUN ---",
            json.dumps(hy_performance, ensure_ascii=False, indent=2),
            "",
            "--- CURRENT KARGA SERVER RUN ---",
            json.dumps(karga_performance, ensure_ascii=False, indent=2),
            "",
            "--- ORIGINAL ITEM 001 COMPATIBILITY PROBE (PRESERVED VERBATIM) ---",
            original_probe_text.rstrip(),
            "",
        ]
    )
    RUNTIME_PROBE_PATH.write_text(text, encoding="utf-8")


def parse_verbose_load_diagnostics(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")

    def capture(pattern: str) -> float | str | None:
        match = re.search(pattern, text)
        if not match:
            return None
        value = match.group(1)
        try:
            return float(value)
        except ValueError:
            return value

    diagnostic_lines = [
        line
        for line in text.splitlines()
        if any(
            marker in line
            for marker in (
                "offloading output layer",
                "offloading ",
                "offloaded ",
                "model buffer size",
                "KV buffer size",
                "compute buffer size",
            )
        )
    ]
    return {
        "log_path": str(path),
        "offloaded_layers": capture(r"offloaded\s+(\d+/\d+)\s+layers to GPU"),
        "cpu_mapped_model_buffer_mib": capture(
            r"CPU_Mapped model buffer size\s*=\s*([\d.]+)\s*MiB"
        ),
        "cuda_model_buffer_mib": capture(
            r"CUDA0 model buffer size\s*=\s*([\d.]+)\s*MiB"
        ),
        "cuda_kv_buffer_mib": capture(r"CUDA0 KV buffer size\s*=\s*([\d.]+)\s*MiB"),
        "cuda_compute_buffer_mib": capture(
            r"CUDA0 compute buffer size\s*=\s*([\d.]+)\s*MiB"
        ),
        "cuda_host_compute_buffer_mib": capture(
            r"CUDA_Host compute buffer size\s*=\s*([\d.]+)\s*MiB"
        ),
        "diagnostic_lines": diagnostic_lines,
    }


def write_summary_file(
    dataset_sha256: str,
    model_metadata: dict[str, Any],
    hy_performance: dict[str, Any],
    karga_performance: dict[str, Any],
    structural_flags: dict[str, Any],
) -> None:
    summary = {
        "dataset_size": 30,
        "dataset_sha256": dataset_sha256,
        "dataset_path": str(DATASET_PATH),
        "hy_mt2": {
            "model_path": str(HY_MODEL_PATH),
            "model_sha256": model_metadata["hy_mt2"]["sha256"],
            "architecture": model_metadata["hy_mt2"]["gguf_architecture"],
            "quantization": model_metadata["hy_mt2"]["quantization"],
            **hy_performance,
            "structural_failure_count": structural_flags["hy_mt2"]["total_structural_failures"],
        },
        "karga": {
            "model_path": str(KARGA_MODEL_PATH),
            "model_sha256": model_metadata["karga"]["sha256"],
            "architecture": model_metadata["karga"]["gguf_architecture"],
            "quantization": model_metadata["karga"]["quantization"],
            **karga_performance,
            "structural_failure_count": structural_flags["karga"]["total_structural_failures"],
        },
        "qwen": {
            "rerun": False,
            "new_model_calls": 0,
            "source_artifact_path": str(STORED_QWEN_PATH),
            "original_source_artifact_path": str(ORIGINAL_QWEN_PATH),
        },
        "ocr_rerun": False,
        "semantic_resolver_rerun": False,
        "madlad_rerun": False,
        "translategemma_rerun": False,
        "production_integration": False,
        "commit": False,
        "push": False,
        "model_outputs_manually_corrected": False,
        "models_run_sequentially": True,
    }
    json_dump(OUTPUT_DIR / "summary.json", summary)


def rebuild_saved_artifacts() -> int:
    dataset, _, dataset_sha256 = validate_dataset()
    model_metadata = json.loads(
        (OUTPUT_DIR / "model_metadata.json").read_text(encoding="utf-8")
    )
    hy_results = json.loads((OUTPUT_DIR / "hy_mt2_results.json").read_text(encoding="utf-8"))
    karga_results = json.loads((OUTPUT_DIR / "karga_results.json").read_text(encoding="utf-8"))
    if len(hy_results) != 30 or len(karga_results) != 30:
        raise ValueError("Cannot rebuild: both saved model result files must have 30 records")
    for results in (hy_results, karga_results):
        for record in results:
            flags, hints = detect_structural_flags(
                record["translation_input"], record["stripped_translation"]
            )
            record["flags"] = flags
            record["review_hints"] = hints
    write_individual_results("hy_mt2", hy_results)
    write_individual_results("karga", karga_results)
    _, structural_flags = build_review_artifacts(dataset, hy_results, karga_results)
    performance_summary = json.loads(
        (OUTPUT_DIR / "performance_summary.json").read_text(encoding="utf-8")
    )
    hy_performance = refresh_performance_metrics(
        performance_summary["hy_mt2"], hy_results
    )
    karga_performance = refresh_performance_metrics(
        performance_summary["karga"], karga_results
    )
    hy_performance["verified_load_diagnostics"] = parse_verbose_load_diagnostics(
        OUTPUT_DIR / "hy_mt2_verbose_load.log"
    )
    karga_performance["verified_load_diagnostics"] = parse_verbose_load_diagnostics(
        OUTPUT_DIR / "karga_verbose_load.log"
    )
    json_dump(
        OUTPUT_DIR / "performance_summary.json",
        {
            "hy_mt2": hy_performance,
            "karga": karga_performance,
            "models_run_sequentially": True,
            "qwen_new_model_calls": 0,
        },
    )
    write_summary_file(
        dataset_sha256,
        model_metadata,
        hy_performance,
        karga_performance,
        structural_flags,
    )
    print("Saved outputs rebuilt without any model calls.", flush=True)
    return 0


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if "--rebuild-only" in sys.argv[1:]:
        return rebuild_saved_artifacts()
    print("=== REAL30 HY-MT2 / KARGA / STORED QWEN SHOOTOUT ===", flush=True)
    dataset, _, dataset_sha256 = validate_dataset()
    print(
        f"Dataset validated: 30 items, 30 unique IDs, SHA256={dataset_sha256}",
        flush=True,
    )
    if not LLAMA_SERVER_EXE.is_file():
        raise FileNotFoundError(LLAMA_SERVER_EXE)
    if not HY_MODEL_PATH.is_file():
        raise FileNotFoundError(HY_MODEL_PATH)
    if not KARGA_MODEL_PATH.is_file():
        raise FileNotFoundError(KARGA_MODEL_PATH)

    original_probe_text = RUNTIME_PROBE_PATH.read_text(encoding="utf-8")
    existing_probe = parse_existing_probe(RUNTIME_PROBE_PATH)

    print("Inspecting exact GGUF headers and hashing model files...", flush=True)
    model_metadata = {
        "hy_mt2": inspect_gguf(HY_MODEL_PATH),
        "karga": inspect_gguf(KARGA_MODEL_PATH),
    }
    model_metadata["hy_mt2"]["execution_prompt_template"] = HY_PROMPT
    model_metadata["karga"]["execution_prompt_template"] = KARGA_PROMPT
    json_dump(OUTPUT_DIR / "model_metadata.json", model_metadata)
    print(
        "HY-MT2: "
        f"{model_metadata['hy_mt2']['gguf_architecture']} "
        f"{model_metadata['hy_mt2']['quantization']}, "
        f"{model_metadata['hy_mt2']['parameter_count_billions']}B params",
        flush=True,
    )
    print(
        "Karga: "
        f"{model_metadata['karga']['gguf_architecture']} "
        f"{model_metadata['karga']['quantization']}, "
        f"{model_metadata['karga']['parameter_count_billions']}B params",
        flush=True,
    )

    hy_results, hy_performance = run_model(
        "hy_mt2",
        "HY-MT2-7B Q8_0",
        HY_MODEL_PATH,
        HY_PROMPT,
        8137,
        dataset,
        existing_probe["hy_mt2"],
    )
    write_individual_results("hy_mt2", hy_results)
    print("HY-MT2 server fully terminated before Karga load.", flush=True)

    karga_results, karga_performance = run_model(
        "karga",
        "Karga-DPO-v0.1 Q6_K",
        KARGA_MODEL_PATH,
        KARGA_PROMPT,
        8138,
        dataset,
        existing_probe["karga"],
    )
    write_individual_results("karga", karga_results)

    _, structural_flags = build_review_artifacts(dataset, hy_results, karga_results)
    performance_summary = {
        "hy_mt2": hy_performance,
        "karga": karga_performance,
        "models_run_sequentially": True,
        "qwen_new_model_calls": 0,
    }
    json_dump(OUTPUT_DIR / "performance_summary.json", performance_summary)

    write_summary_file(
        dataset_sha256,
        model_metadata,
        hy_performance,
        karga_performance,
        structural_flags,
    )
    write_runtime_probe(original_probe_text, hy_performance, karga_performance)

    both_complete = (
        hy_performance["successful_outputs"] == 30
        and karga_performance["successful_outputs"] == 30
    )
    print(
        f"Completed: HY={hy_performance['successful_outputs']}/30, "
        f"Karga={karga_performance['successful_outputs']}/30, Qwen new calls=0",
        flush=True,
    )
    return 0 if both_complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
