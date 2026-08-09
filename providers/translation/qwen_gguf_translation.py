"""Qwen3.5-9B GGUF llama-server translation provider.

Uses llama.exe serve (or llama-server) running with CUDA offload for fast text translation.
Preserves existing prompt structure, structured JSON output, fidelity flags,
term_id mapping, validation pipeline, and token-aware batching.

The CandidateStore is NEVER mutated directly inside this provider.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from loguru import logger

from providers.translation.base import (
    TranslationInput,
    TranslationItem,
    TranslationOutput,
    TranslationOutputItem,
    TranslationProvider,
)
from providers.translation.qwen_prompt import (
    _SYSTEM_PROMPT,
    build_qwen_translation_user_prompt,
)
from providers.translation.qwen_translation import (
    _extract_translations_from_natural_language,
    _strip_thinking,
    _try_extract_json,
    _validate_output,
)

DEFAULT_GGUF_MODEL_PATH = r"C:\AI\Models\Qwen3.5-9B-GGUF\Qwen3.5-9B-Q5_K_M.gguf"
DEFAULT_LLAMA_EXE_PATH = r"C:\AI\llama-cpp-cuda\llama.exe"
DEFAULT_SERVER_URL = "http://127.0.0.1:8080"


@dataclass
class QwenGGUFMetrics:
    model_load_seconds: float = 0.0
    peak_vram_gb: float | None = None
    translation_model: str = "Qwen3.5-9B-Q5_K_M-GGUF"
    input_token_count: int = 0
    generated_token_count: int = 0
    generation_seconds: float = 0.0
    tokens_per_sec: float = 0.0
    generation_call_count: int = 0
    retries: int = 0
    cuda_active: bool = True
    gpu_offload: str = "configured: 36/36 layers (100%)"


def _parse_gguf_output(
    item_term_maps: dict[int, dict[str, str]],
    model_name: str,
    inp: TranslationInput,
    raw: str,
) -> TranslationOutput:
    json_obj = _try_extract_json(raw)
    results: list[TranslationOutputItem] = []

    if json_obj is None:
        translations = _extract_translations_from_natural_language(raw, inp.items)
        if translations:
            logger.warning("Translation JSON parse failed; using NL fallback extraction")
            json_obj = {"translations": translations}
        else:
            logger.warning("Translation output JSON parse failed")
            for item in inp.items:
                results.append(
                    TranslationOutputItem(
                        region_id=item.region_id,
                        source=item.source,
                        translation=None,
                        raw_model_response=raw[:500],
                        validation_warnings=["json_parse_failure"],
                        requires_review=True,
                    )
                )
            return TranslationOutput(
                inputs=inp,
                results=results,
                raw_response=raw[:2000],
                repair_model=model_name,
            )

    translations: list[dict[str, Any]] = json_obj.get("translations", [])
    warnings_by_id: dict[int, list[str]] = _validate_output(
        inp.items, translations, json_obj, profile=inp.profile
    )

    for item in inp.items:
        match = next((t for t in translations if t.get("id") == item.region_id), None)
        if match is None:
            results.append(
                TranslationOutputItem(
                    region_id=item.region_id,
                    source=item.source,
                    translation=None,
                    raw_model_response=raw[:500],
                    validation_warnings=warnings_by_id.get(item.region_id, ["missing_id"]),
                    requires_review=True,
                )
            )
        else:
            translation = str(match.get("translation", "")).strip()
            warns = warnings_by_id.get(item.region_id, [])
            resolved = translation if translation and translation.lower() != "null" else None
            term_usages = match.get("term_usages", [])
            if not isinstance(term_usages, list):
                term_usages = []
            fidelity_flags = match.get("fidelity_flags", [])
            if not isinstance(fidelity_flags, list):
                fidelity_flags = []
            item_term_map = item_term_maps.get(item.region_id, {})
            results.append(
                TranslationOutputItem(
                    region_id=item.region_id,
                    source=item.source,
                    translation=resolved,
                    raw_model_response=raw[:500],
                    validation_warnings=warns,
                    requires_review=bool(warns),
                    term_usages=term_usages,
                    fidelity_flags=fidelity_flags,
                    term_id_map=item_term_map,
                )
            )

    return TranslationOutput(
        inputs=inp,
        results=results,
        raw_response=raw[:2000],
        repair_model=model_name,
    )


class QwenGGUFTranslationProvider(TranslationProvider):
    """Qwen3.5-9B GGUF translation provider backed by llama-server (CUDA)."""

    def __init__(
        self,
        model_path: str = DEFAULT_GGUF_MODEL_PATH,
        executable_path: str = DEFAULT_LLAMA_EXE_PATH,
        server_url: str = DEFAULT_SERVER_URL,
        n_gpu_layers: int = 99,
        auto_start_server: bool = True,
        timeout_seconds: float = 120.0,
    ) -> None:
        self._model_path = model_path
        self._executable_path = executable_path
        self._server_url = server_url.rstrip("/")
        self._n_gpu_layers = n_gpu_layers
        self._auto_start_server = auto_start_server
        self._timeout_seconds = timeout_seconds

        self._process: subprocess.Popen | None = None
        self._owns_server = False
        self._loaded = False
        self.metrics = QwenGGUFMetrics()
        self._item_term_maps: dict[int, dict[str, str]] = {}

    @property
    def name(self) -> str:
        return "Qwen3.5-9B-GGUF-Translation"

    @property
    def is_loaded(self) -> bool:
        return self._loaded and self._check_health()

    def _check_health(self) -> bool:
        """Check if llama-server is alive and serving HTTP requests."""
        url = f"{self._server_url}/health"
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        return False

    def load(self) -> None:
        if self.is_loaded:
            logger.info(f"QwenGGUFTranslationProvider: Server at {self._server_url} is already running and healthy.")
            return

        t0 = time.perf_counter()
        if self._check_health():
            self._loaded = True
            self._owns_server = False
            self.metrics.model_load_seconds = time.perf_counter() - t0
            logger.info(f"Connected to pre-existing llama-server at {self._server_url}")
            return

        if not self._auto_start_server:
            raise RuntimeError(
                f"llama-server unavailable at {self._server_url} and auto_start_server is False."
            )

        if not os.path.exists(self._model_path):
            raise RuntimeError(f"GGUF model file not found: {self._model_path}")
        if not os.path.exists(self._executable_path):
            raise RuntimeError(f"llama executable not found: {self._executable_path}")

        # Parse port and host from server_url
        port = "8080"
        host = "127.0.0.1"
        url_part = self._server_url.split("//")[-1]
        if ":" in url_part:
            host, port = url_part.split(":", 1)

        cmd = [self._executable_path]
        if self._executable_path.lower().endswith("llama.exe"):
            cmd.append("serve")

        cmd.extend([
            "-m", self._model_path,
            "-ngl", str(self._n_gpu_layers),
            "--host", host,
            "--port", port,
            "--reasoning", "off",
            "-c", "4096",
        ])

        logger.info(f"Starting managed llama-server process: {' '.join(cmd)}")
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        self._owns_server = True

        # Wait for server to report ready
        start_time = time.perf_counter()
        ready = False
        while time.perf_counter() - start_time < 45.0:
            if self._process.poll() is not None:
                raise RuntimeError(
                    f"llama-server process exited unexpectedly with code {self._process.returncode}"
                )
            if self._check_health():
                ready = True
                break
            time.sleep(0.5)

        if not ready:
            self.unload()
            raise RuntimeError(f"Timeout waiting for llama-server to initialize at {self._server_url} (45s)")

        self._loaded = True
        self.metrics.model_load_seconds = time.perf_counter() - t0
        logger.info(
            f"QwenGGUFTranslationProvider loaded successfully, load time: {self.metrics.model_load_seconds:.2f}s"
        )

    def unload(self) -> None:
        if self._owns_server and self._process is not None:
            logger.info("Terminating managed llama-server process...")
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except Exception:
                try:
                    self._process.kill()
                except Exception:
                    pass
            self._process = None
            self._owns_server = False

        self._loaded = False
        logger.info("QwenGGUFTranslationProvider unloaded.")

    def _build_prompt(self, inp: TranslationInput) -> tuple[str, str]:
        user_prompt, item_term_maps = build_qwen_translation_user_prompt(inp)
        self._item_term_maps = item_term_maps
        return _SYSTEM_PROMPT, user_prompt

    def _query_llm(self, system_prompt: str, user_prompt: str) -> tuple[str, int, int]:
        endpoint = f"{self._server_url}/v1/chat/completions"
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 1024,
            "stream": False,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self._timeout_seconds) as resp:
                res_json = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace") if e.fp else str(e)
            raise RuntimeError(f"llama-server HTTP error {e.code}: {err_body}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to communicate with llama-server at {self._server_url}: {e}") from e

        choices = res_json.get("choices", [])
        if not choices:
            raise RuntimeError(f"llama-server returned empty choices: {res_json}")

        message = choices[0].get("message", {})
        raw_text = message.get("content", "") or ""

        usage = res_json.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        return raw_text, prompt_tokens, completion_tokens

    def translate(self, inp: TranslationInput) -> TranslationOutput:
        if not self.is_loaded:
            raise RuntimeError("QwenGGUFTranslationProvider not loaded; call load() first")

        from core.translation.batcher import TranslationBatcher
        batcher = TranslationBatcher()
        sub_inputs = batcher.create_batches(inp)

        if len(sub_inputs) == 1:
            return self._translate_single_batch(sub_inputs[0])

        logger.info(f"Executing GGUF translation across {len(sub_inputs)} sub-batches")
        sub_outputs: list[TranslationOutput] = []
        for sub_inp in sub_inputs:
            sub_outputs.append(self._translate_single_batch(sub_inp))

        return batcher.merge_outputs(inp, sub_outputs)

    def _translate_single_batch(
        self, inp: TranslationInput, retry_count: int = 0
    ) -> TranslationOutput:
        system_prompt, user_prompt = self._build_prompt(inp)
        t0 = time.perf_counter()

        raw_output, ilen, gen_len = self._query_llm(system_prompt, user_prompt)
        gen_time = time.perf_counter() - t0

        raw_output = _strip_thinking(raw_output)

        self.metrics.input_token_count += ilen
        self.metrics.generated_token_count += gen_len
        self.metrics.generation_seconds += gen_time
        if self.metrics.generation_seconds > 0:
            self.metrics.tokens_per_sec = (
                self.metrics.generated_token_count / self.metrics.generation_seconds
            )
        self.metrics.generation_call_count += 1

        output = _parse_gguf_output(
            self._item_term_maps,
            self.metrics.translation_model,
            inp,
            raw_output,
        )

        missing_ids = [r.region_id for r in output.results if r.translation is None]
        if missing_ids and gen_len >= 900 and retry_count < 2 and len(inp.items) > 1:
            logger.warning(
                f"GGUF batch truncated at {gen_len} tokens with missing IDs {missing_ids}. Splitting batch to recover missing items."
            )
            self.metrics.retries += 1
            half = len(inp.items) // 2
            sub_a = TranslationInput(
                items=inp.items[:half],
                glossary=inp.glossary,
                chapter_context=inp.chapter_context,
                profile=inp.profile,
                context_items=inp.context_items,
                candidate_store=inp.candidate_store,
                chapter_id=inp.chapter_id,
            )
            sub_b = TranslationInput(
                items=inp.items[half:],
                glossary=inp.glossary,
                chapter_context=inp.chapter_context,
                profile=inp.profile,
                context_items=inp.items[:half][-2:],
                candidate_store=inp.candidate_store,
                chapter_id=inp.chapter_id,
            )
            out_a = self._translate_single_batch(sub_a, retry_count + 1)
            out_b = self._translate_single_batch(sub_b, retry_count + 1)
            from core.translation.batcher import TranslationBatcher
            return TranslationBatcher().merge_outputs(inp, [out_a, out_b])

        logger.info(
            f"Qwen GGUF batch complete: {ilen} in, {gen_len} gen, {gen_time:.2f}s "
            f"({gen_len / gen_time:.2f} tok/s if gen_time > 0 else 0)"
        )
        return output
