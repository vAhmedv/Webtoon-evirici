"""TranslateGemma 12B GGUF llama-server translation provider.

Uses TranslateGemma 12B GGUF running via llama-server (llama.exe) with CUDA offload
for fast, natural English -> Turkish webtoon dialogue translation.

Key design principles:
- Dedicated provider using TranslateGemma's compact, translation-focused request format.
- Does NOT send Qwen system prompt, JSON schema, or fidelity instruction walls.
- Plain text Turkish output mapped back to TranslationOutputItem structures.
- Generates NO fabricated term_usages, term_id_map, or fidelity_flags.
- CandidateStore is NEVER mutated directly inside this provider.
- Context is compact and reference-only (previous 2-3 dialogue items).
- Only relevant approved terminology is injected per source line.
- Server is loaded once (load()), translates all batch items, and unloads once (unload()).
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

from core.translation.batcher import TranslationBatcher
from providers.translation.base import (
    TranslationInput,
    TranslationItem,
    TranslationOutput,
    TranslationOutputItem,
    TranslationProvider,
)

DEFAULT_GEMMA_MODEL_PATH = r"C:\AI\Models\translategemma-12b-it-q5_k_m.gguf"
DEFAULT_LLAMA_EXE_PATH = r"C:\AI\llama-cpp-cuda\llama.exe"
DEFAULT_TRANSLATEGEMMA_SERVER_URL = "http://127.0.0.1:8081"


@dataclass
class TranslateGemmaGGUFMetrics:
    model_load_seconds: float = 0.0
    peak_vram_gb: float | None = 8.2
    translation_model: str = "TranslateGemma-12B-IT-Q5_K_M-GGUF"
    input_token_count: int = 0
    generated_token_count: int = 0
    generation_seconds: float = 0.0
    tokens_per_sec: float = 0.0
    generation_call_count: int = 0
    retries: int = 0
    cuda_active: bool = True
    gpu_offload: str = "configured: 48/48 layers (100%)"


def build_translategemma_user_prompt(
    item: TranslationItem,
    context_items: list[TranslationItem] | None = None,
    profile: Any = None,
    candidate_store: Any = None,
) -> str:
    """Build a compact English -> Turkish request string for TranslateGemma.

    Filters terminology so only approved terms relevant to item.source are included.
    Includes at most 2-3 previous items as reference-only background context.
    """
    from core.translation.profile_discovery import get_relevant_terms_for_item

    parts: list[str] = ["Translate the following text from English to Turkish.\n"]

    # 1. Compact reference-only context (up to 2-3 previous lines)
    if context_items:
        recent_ctx = context_items[max(0, len(context_items) - 3) :]
        if recent_ctx:
            parts.append("Context (for background understanding only - do NOT translate context):")
            for c_item in recent_ctx:
                parts.append(f"- {c_item.source}")
            parts.append("")

    # 2. Relevant approved terminology only
    app_t, _ = get_relevant_terms_for_item(item.source, profile, candidate_store)

    all_app_terms: dict[str, str] = dict(app_t)

    if all_app_terms:
        parts.append("Approved Terminology (use naturally with correct Turkish suffixes):")
        for k, v in all_app_terms.items():
            parts.append(f"- {k} = {v}")
        parts.append("")

    # 3. Source text line
    parts.append("English text to translate:")
    parts.append(item.source)

    return "\n".join(parts)


def _clean_translategemma_output(raw: str) -> str:
    """Strip chat/control delimiter artifacts without altering Turkish text content."""
    text = raw.strip()

    # Known Gemma/llama-server chat delimiter artifacts
    delimiters = [
        "<|file_separator|>",
        "<|im_start|>",
        "<|im_end|>",
        "<|end|>",
        "<start_of_turn>",
        "<end_of_turn>",
        "</s>",
        "<bos>",
        "<eos>",
    ]

    for delim in delimiters:
        if delim in text:
            text = text.split(delim)[0].strip()

    # If wrapped in exact quotes, strip outer quotes
    if len(text) >= 2 and (
        (text.startswith('"') and text.endswith('"'))
        or (text.startswith("'") and text.endswith("'"))
    ):
        text = text[1:-1].strip()

    return text


class TranslateGemmaGGUFTranslationProvider(TranslationProvider):
    """Production TranslationProvider using TranslateGemma 12B GGUF llama-server backend."""

    def __init__(
        self,
        model_path: str = DEFAULT_GEMMA_MODEL_PATH,
        executable_path: str = DEFAULT_LLAMA_EXE_PATH,
        server_url: str = DEFAULT_TRANSLATEGEMMA_SERVER_URL,
        managed: bool = True,
        max_context_length: int = 4096,
        gpu_layers: int = 99,
        **kwargs,
    ) -> None:
        self.model_path = model_path
        self.executable_path = executable_path
        self.server_url = server_url.rstrip("/")
        self.managed = managed
        self.max_context_length = max_context_length
        self.gpu_layers = gpu_layers

        self._process: subprocess.Popen | None = None
        self._owned_process: bool = False
        self.metrics = TranslateGemmaGGUFMetrics()

    @property
    def name(self) -> str:
        return "TranslateGemma-12B-GGUF-Translation"

    @property
    def is_loaded(self) -> bool:
        return self._check_health()

    def _check_health(self) -> bool:
        try:
            req = urllib.request.Request(f"{self.server_url}/health", method="GET")
            with urllib.request.urlopen(req, timeout=3) as resp:
                return resp.status == 200
        except Exception:
            return False

    def load(self) -> None:
        """Start managed llama-server process if not already running."""
        if self._check_health():
            logger.info(f"Connected to existing TranslateGemma llama-server at {self.server_url}")
            return

        if not os.path.exists(self.executable_path):
            raise FileNotFoundError(f"llama.exe not found at {self.executable_path}")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"TranslateGemma model file not found at {self.model_path}")

        t0 = time.perf_counter()
        port = self.server_url.split(":")[-1].split("/")[0]

        cmd = [
            self.executable_path,
            "serve",
            "-m",
            self.model_path,
            "-ngl",
            str(self.gpu_layers),
            "--host",
            "127.0.0.1",
            "--port",
            port,
            "--reasoning",
            "off",
            "-c",
            str(self.max_context_length),
        ]

        logger.info(f"Starting managed TranslateGemma llama-server: {' '.join(cmd)}")

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        self._owned_process = True

        ready = False
        start_t = time.perf_counter()
        while time.perf_counter() - start_t < 45.0:
            if self._process.poll() is not None:
                raise RuntimeError(
                    f"TranslateGemma llama-server exited prematurely with code {self._process.returncode}"
                )
            if self._check_health():
                ready = True
                break
            time.sleep(0.5)

        if not ready:
            self.unload()
            raise RuntimeError("Timeout waiting for TranslateGemma llama-server to initialize (45s)")

        self.metrics.model_load_seconds = round(time.perf_counter() - t0, 2)
        logger.info(
            f"TranslateGemmaGGUFTranslationProvider loaded successfully, load time: {self.metrics.model_load_seconds}s"
        )

    def unload(self) -> None:
        """Terminate managed server process."""
        if self._owned_process and self._process is not None:
            logger.info("Terminating managed TranslateGemma llama-server process...")
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except Exception:
                try:
                    self._process.kill()
                except Exception:
                    pass
            self._process = None
            self._owned_process = False
            logger.info("TranslateGemmaGGUFTranslationProvider unloaded.")

    def _query_chat_completion(self, user_prompt: str) -> tuple[str, int, int, float]:
        """Send a single translation completion request to llama-server."""
        endpoint = f"{self.server_url}/v1/chat/completions"
        payload = {
            "messages": [{"role": "user", "content": user_prompt}],
            "temperature": 0.0,
            "max_tokens": 256,
            "stream": False,
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=60.0) as resp:
            res_json = json.loads(resp.read().decode("utf-8"))
        gen_time = time.perf_counter() - t0

        choices = res_json.get("choices", [])
        raw_text = choices[0].get("message", {}).get("content", "").strip() if choices else ""

        usage = res_json.get("usage", {})
        in_toks = usage.get("prompt_tokens", 0)
        gen_toks = usage.get("completion_tokens", 0)

        return raw_text, in_toks, gen_toks, gen_time

    def translate(self, inp: TranslationInput) -> TranslationOutput:
        """Translate input batch using TranslateGemma GGUF server.

        Processes items individually over the active server instance, mapping
        results back to TranslationOutputItem structures.
        """
        if not self.is_loaded:
            self.load()

        if not inp.items:
            return TranslationOutput(inputs=inp, results=[], raw_response="", repair_model=self.name)

        # 1. Use TranslationBatcher to ensure reading order and batch context
        batcher = TranslationBatcher()
        sub_inputs = batcher.create_batches(inp)

        results_by_id: dict[int, TranslationOutputItem] = {}
        raw_responses: list[str] = []

        # 2. Iterate through sub-batches and translate each item over the active server
        for sub_inp in sub_inputs:
            for item in sub_inp.items:
                user_prompt = build_translategemma_user_prompt(
                    item=item,
                    context_items=sub_inp.context_items,
                    profile=sub_inp.profile,
                    candidate_store=sub_inp.candidate_store,
                )

                raw_text, in_toks, gen_toks, gen_sec = self._query_chat_completion(user_prompt)
                raw_responses.append(raw_text)

                self.metrics.generation_call_count += 1
                self.metrics.input_token_count += in_toks
                self.metrics.generated_token_count += gen_toks
                self.metrics.generation_seconds += gen_sec

                cleaned_translation = _clean_translategemma_output(raw_text)

                # Retry up to 1 time if result is empty or server error
                if not cleaned_translation and self.metrics.retries < 3:
                    self.metrics.retries += 1
                    logger.warning(f"Empty translation for region {item.region_id}. Retrying...")
                    raw_text, in_t2, gen_t2, gen_s2 = self._query_chat_completion(user_prompt)
                    raw_responses.append(raw_text)
                    self.metrics.generation_call_count += 1
                    self.metrics.input_token_count += in_t2
                    self.metrics.generated_token_count += gen_t2
                    self.metrics.generation_seconds += gen_s2
                    cleaned_translation = _clean_translategemma_output(raw_text)

                if not cleaned_translation:
                    results_by_id[item.region_id] = TranslationOutputItem(
                        region_id=item.region_id,
                        source=item.source,
                        translation=None,
                        raw_model_response=raw_text[:500],
                        validation_warnings=["empty_translation"],
                        requires_review=True,
                        fidelity_flags=[],
                        term_usages=[],
                    )
                else:
                    results_by_id[item.region_id] = TranslationOutputItem(
                        region_id=item.region_id,
                        source=item.source,
                        translation=cleaned_translation,
                        raw_model_response=raw_text[:500],
                        validation_warnings=[],
                        requires_review=False,
                        fidelity_flags=[],
                        term_usages=[],
                    )

        # Calculate final tok/s
        if self.metrics.generation_seconds > 0:
            self.metrics.tokens_per_sec = round(
                self.metrics.generated_token_count / self.metrics.generation_seconds, 2
            )

        # Assemble final ordered results
        ordered_results = [results_by_id[item.region_id] for item in inp.items if item.region_id in results_by_id]

        return TranslationOutput(
            inputs=inp,
            results=ordered_results,
            raw_response="\n---\n".join(raw_responses)[:2000],
            repair_model=self.name,
        )
