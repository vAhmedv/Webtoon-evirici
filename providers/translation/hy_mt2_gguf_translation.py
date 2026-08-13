"""Production Hy-MT2 GGUF translator using a persistent local llama-server.

The provider deliberately reuses the hardened Qwen-v2 preparation/finalization
path.  Model-specific responsibilities in this module are limited to server
lifecycle, native prompt rendering, deterministic inference, and trace capture.

The Q8 model nearly fills a 12 GiB GPU.  Call :meth:`unload` before starting a
different heavyweight GPU model (notably visual OCR).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import http.client
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

from core.translation.protection import restore_protected_translation
from providers.translation.base import (
    TranslationInput,
    TranslationOutput,
    TranslationOutputItem,
)
from providers.translation.qwen_gguf_translation_v2 import (
    QwenGGUFMetricsV2,
    QwenGGUFTranslationProviderV2,
)


logger = logging.getLogger(__name__)

DEFAULT_HY_MT2_MODEL_PATH = r"C:\AI\Models\HY-MT2-7B-Q8_0.gguf"
DEFAULT_LLAMA_SERVER_PATH = r"C:\AI\llama-cpp-cuda\llama-server.exe"
DEFAULT_HY_MT2_SERVER_URL = "http://127.0.0.1:8085"
DEFAULT_HY_MT2_SERVER_ALIAS = "hy-mt2-7b-q8_0-translator"
DEFAULT_HY_MT2_SERVER_LOG = "logs/hy_mt2_llama_server.log"

HY_MT2_TRANSLATION_INSTRUCTION = (
    "Translate the following text into Turkish. Note that you should only output "
    "the translated result without any additional explanation:\n"
)
HY_MT2_NATIVE_PROMPT_TEMPLATE = (
    "<|startoftext|>" + HY_MT2_TRANSLATION_INSTRUCTION + "{source}<|extra_0|>"
)
_RUNTIME_EOS_SUFFIX = re.compile(
    r"(?:<\|eos\|>|<\|endoftext\|>|<\|im_end\|>)\s*$",
    re.IGNORECASE,
)


def render_hy_mt2_prompt(prepared_source_text: str) -> str:
    """Render the proven native Hy-MT2 English→Turkish translation prompt."""
    return HY_MT2_NATIVE_PROMPT_TEMPLATE.format(source=prepared_source_text)


def clean_hy_mt2_output(raw_text: str, rendered_prompt: str) -> str:
    """Strip transport-only artifacts without editing translation content."""
    cleaned = raw_text.strip()
    if cleaned.startswith(rendered_prompt):
        cleaned = cleaned[len(rendered_prompt) :].lstrip()
    while _RUNTIME_EOS_SUFFIX.search(cleaned):
        cleaned = _RUNTIME_EOS_SUFFIX.sub("", cleaned).rstrip()
    return cleaned


@dataclass
class HyMT2GGUFMetrics(QwenGGUFMetricsV2):
    translation_model: str = "Tencent-Hy-MT2-7B-Q8_0-GGUF"


@dataclass(frozen=True)
class HyMT2ProductionTrace:
    region_id: int
    original_source: str
    normalized_input: str
    protected_input: str
    protected_terms: list[dict[str, Any]]
    raw_hy_output: str
    stripped_output: str
    restored_output: str | None
    final_output: str | None
    guard_flags: list[str]
    requires_review: bool
    model_call_performed: bool
    latency_sec: float | None
    pipeline_diagnosis: str


class HyMT2GGUFTranslationProvider(QwenGGUFTranslationProviderV2):
    """Hy-MT2 production provider with shared normalization/protection/guards."""

    def __init__(
        self,
        model_path: str = DEFAULT_HY_MT2_MODEL_PATH,
        executable_path: str = DEFAULT_LLAMA_SERVER_PATH,
        server_url: str = DEFAULT_HY_MT2_SERVER_URL,
        *,
        managed: bool = True,
        max_context_length: int = 2048,
        gpu_layers: int = 99,
        max_output_tokens: int = 128,
        startup_timeout_sec: float = 90.0,
        request_timeout_sec: float = 90.0,
        server_alias: str = DEFAULT_HY_MT2_SERVER_ALIAS,
        server_log_path: str = DEFAULT_HY_MT2_SERVER_LOG,
    ) -> None:
        super().__init__(
            model_path=model_path,
            executable_path=executable_path,
            server_url=server_url,
            max_context_length=max_context_length,
            gpu_layers=gpu_layers,
            system_prompt="",
        )
        self.managed = managed
        self.max_output_tokens = max_output_tokens
        self.startup_timeout_sec = startup_timeout_sec
        self.request_timeout_sec = request_timeout_sec
        self.server_alias = server_alias
        self.server_log_path = server_log_path
        self.metrics = HyMT2GGUFMetrics(translation_model=Path(self.model_path).name)
        self.last_traces: list[HyMT2ProductionTrace] = []
        self.last_server_command: list[str] = []
        self._server_log_handle: Any | None = None

    @property
    def name(self) -> str:
        return "Tencent-Hy-MT2-7B-GGUF-Translator"

    @property
    def version(self) -> str:
        return "production-v1"

    @property
    def is_loaded(self) -> bool:
        return (
            self._loaded
            and self._check_health()
            and self._server_identity_compatible()
        )

    def _server_identity(self) -> dict[str, Any] | None:
        # Prefer llama.cpp's native route because model_path identifies the
        # configured GGUF directly instead of relying only on a public alias.
        try:
            request = urllib.request.Request(f"{self.server_url}/props", method="GET")
            with urllib.request.urlopen(request, timeout=3.0) as response:
                payload = json.loads(response.read().decode("utf-8"))
            loaded = str(payload.get("model_path") or payload.get("model_alias") or "")
            if loaded:
                return {"id": loaded}
        except Exception:
            pass
        endpoint = f"{self.server_url}/v1/models"
        try:
            request = urllib.request.Request(endpoint, method="GET")
            with urllib.request.urlopen(request, timeout=3.0) as response:
                payload = json.loads(response.read().decode("utf-8"))
            models = payload.get("data") or []
            return models[0] if models else None
        except Exception:
            return None

    def _server_identity_compatible(self) -> bool:
        identity = self._server_identity()
        if not identity:
            return False
        model_id = str(identity.get("id") or "").casefold()
        expected_alias = self.server_alias.casefold()
        expected_filename = Path(self.model_path).name.casefold()
        if model_id not in {expected_alias, expected_filename} and expected_filename not in model_id:
            return False
        return True

    def _wait_for_compatible_identity(self, timeout_sec: float = 5.0) -> bool:
        """Allow llama-server's OpenAI metadata route to settle after health."""
        deadline = time.perf_counter() + timeout_sec
        while time.perf_counter() < deadline:
            if self._server_identity_compatible():
                return True
            time.sleep(0.1)
        return self._server_identity_compatible()

    def _server_log_tail(self, limit: int = 40) -> str:
        try:
            if self._server_log_handle is not None:
                self._server_log_handle.flush()
            lines = Path(self.server_log_path).read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
            return "\n".join(lines[-limit:])
        except Exception:
            return ""

    def load(self) -> None:
        if self.is_loaded:
            return

        started = time.perf_counter()
        if self._check_health():
            if not self._wait_for_compatible_identity():
                raise RuntimeError(
                    f"Port at {self.server_url} is occupied by an incompatible llama-server; "
                    "the process was left untouched"
                )
            self._loaded = True
            self._owned_process = False
            self.metrics.model_load_seconds = round(time.perf_counter() - started, 4)
            logger.info("Connected to compatible existing Hy-MT2 server at %s", self.server_url)
            return

        if not self.managed:
            raise RuntimeError(
                f"No compatible Hy-MT2 server is ready at {self.server_url} and managed=False"
            )
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"Hy-MT2 GGUF model file not found: {self.model_path}")
        if not os.path.isfile(self.executable_path):
            raise FileNotFoundError(f"llama-server executable not found: {self.executable_path}")

        parsed = urllib.parse.urlparse(self.server_url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 8085
        command = [
            self.executable_path,
            "-m",
            self.model_path,
            "-ngl",
            str(self.gpu_layers),
            "-c",
            str(self.max_context_length),
            "-np",
            "1",
            "--host",
            host,
            "--port",
            str(port),
            "--alias",
            self.server_alias,
            "--no-webui",
        ]
        self.last_server_command = command
        log_path = Path(self.server_log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._server_log_handle = log_path.open("a", encoding="utf-8")
        self._server_log_handle.write("\n=== Hy-MT2 managed server start ===\n")
        self._server_log_handle.write("COMMAND: " + " ".join(command) + "\n")
        self._server_log_handle.flush()

        logger.info("Starting managed Hy-MT2 llama-server: %s", " ".join(command))
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=self._server_log_handle,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        self._owned_process = True

        deadline = time.perf_counter() + self.startup_timeout_sec
        while time.perf_counter() < deadline:
            if self._process.poll() is not None:
                exit_code = self._process.returncode
                tail = self._server_log_tail()
                self.unload()
                raise RuntimeError(
                    f"Hy-MT2 llama-server exited during startup with code {exit_code}.\n{tail}"
                )
            if self._check_health():
                if not self._wait_for_compatible_identity():
                    self.unload()
                    raise RuntimeError("Started Hy-MT2 server reported an incompatible model identity")
                self._loaded = True
                self.metrics.model_load_seconds = round(time.perf_counter() - started, 4)
                return
            time.sleep(0.25)

        tail = self._server_log_tail()
        self.unload()
        raise RuntimeError(
            f"Timeout waiting for Hy-MT2 llama-server after {self.startup_timeout_sec:.1f}s.\n{tail}"
        )

    def unload(self) -> None:
        if self._owned_process and self._process is not None:
            logger.info(
                "Terminating owned Hy-MT2 llama-server process PID=%s",
                self._process.pid,
            )
            try:
                self._process.terminate()
                self._process.wait(timeout=15)
            except Exception:
                try:
                    self._process.kill()
                    self._process.wait(timeout=5)
                except Exception:
                    pass
        self._process = None
        self._owned_process = False
        self._loaded = False
        if self._server_log_handle is not None:
            try:
                self._server_log_handle.close()
            finally:
                self._server_log_handle = None

    def _query_chat_completion(self, prepared_text: str) -> tuple[str, int, int, float]:
        rendered_prompt = render_hy_mt2_prompt(prepared_text)
        payload = {
            "prompt": rendered_prompt,
            "temperature": 0.0,
            "top_k": 1,
            "top_p": 1.0,
            "min_p": 0.0,
            "seed": 0,
            "n_predict": self.max_output_tokens,
            "stream": False,
            "cache_prompt": False,
        }
        request = urllib.request.Request(
            f"{self.server_url}/completion",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        request_started = time.perf_counter()
        with urllib.request.urlopen(request, timeout=self.request_timeout_sec) as response:
            response_json = json.loads(response.read().decode("utf-8"))

        request_latency = time.perf_counter() - request_started
        if "content" not in response_json:
            raise RuntimeError("Hy-MT2 llama-server completion response has no content field")

        timings = response_json.get("timings") or {}
        prompt_tokens = response_json.get("tokens_evaluated", timings.get("prompt_n", 0))
        generated_tokens = response_json.get(
            "tokens_predicted", timings.get("predicted_n", 0)
        )
        predicted_ms = timings.get("predicted_ms")
        generation_seconds = (
            float(predicted_ms) / 1000.0 if predicted_ms is not None else request_latency
        )
        return (
            str(response_json["content"]),
            int(prompt_tokens or 0),
            int(generated_tokens or 0),
            generation_seconds,
        )

    def _request_translation(self, prepared_text: str, label: str) -> tuple[str, str, bool]:
        raw_text = ""
        rendered_prompt = render_hy_mt2_prompt(prepared_text)
        for attempt in range(2):
            if attempt:
                self.metrics.retries += 1
            self.metrics.generation_call_count += 1
            try:
                raw_text, input_tokens, generated_tokens, generation_seconds = (
                    self._query_chat_completion(prepared_text)
                )
            except (
                http.client.RemoteDisconnected,
                urllib.error.URLError,
                ConnectionAbortedError,
                ConnectionRefusedError,
                ConnectionResetError,
                TimeoutError,
            ) as exc:
                owned_process = self._process if self._owned_process else None
                pid = owned_process.pid if owned_process is not None else "external"
                alive = owned_process.poll() is None if owned_process is not None else "unknown"
                retry_reason = f"transient_connection_failure:{type(exc).__name__}"
                if attempt == 0 and alive is not False:
                    # /completion is stateless and deterministic here. Repeating it
                    # can duplicate compute, but cannot mutate application state.
                    logger.warning(
                        "Hy-MT2 retrying once for %s; reason=%s server_pid=%s "
                        "process_alive=%s owned_by_provider=%s error=%s",
                        label,
                        retry_reason,
                        pid,
                        alive,
                        owned_process is not None,
                        exc,
                    )
                    time.sleep(0.25)
                    continue
                logger.error(
                    "Hy-MT2 request failed for %s; reason=%s server_pid=%s "
                    "process_alive=%s owned_by_provider=%s error=%s",
                    label,
                    retry_reason,
                    pid,
                    alive,
                    owned_process is not None,
                    exc,
                )
                return "", "", True
            except Exception as exc:
                logger.error(
                    "Hy-MT2 request failed for %s without retry; reason=non_transient:%s error=%s",
                    label,
                    type(exc).__name__,
                    exc,
                )
                return "", "", True

            cleaned = clean_hy_mt2_output(raw_text, rendered_prompt)
            self.metrics.input_token_count += input_tokens
            self.metrics.generated_token_count += generated_tokens
            self.metrics.generation_seconds += generation_seconds
            if self.metrics.generation_seconds > 0:
                self.metrics.tokens_per_sec = (
                    self.metrics.generated_token_count / self.metrics.generation_seconds
                )
            if cleaned:
                return raw_text, cleaned, False
            if attempt == 0:
                logger.warning("Hy-MT2 returned an empty translation for %s; retrying once", label)
        return raw_text, "", False

    @staticmethod
    def _protected_terms(prepared: Any) -> list[dict[str, Any]]:
        return [asdict(meta) for meta in prepared.placeholder_map.values()]

    def translate(self, inp: TranslationInput) -> TranslationOutput:
        if not self.is_loaded:
            self.load()

        self.last_traces = []
        results: list[TranslationOutputItem] = []
        raw_responses: list[str] = []

        for item in inp.items:
            prepared = self._prepare_item(item, inp)
            protected_terms = self._protected_terms(prepared)

            if prepared.system_ui_translation is not None:
                self.metrics.system_ui_bypass_count += 1
                result = TranslationOutputItem(
                    region_id=item.region_id,
                    source=item.source,
                    translation=prepared.system_ui_translation,
                    raw_model_response="[SYSTEM_UI_BYPASS]",
                )
                results.append(result)
                raw_responses.append("[SYSTEM_UI_BYPASS]")
                self.last_traces.append(
                    HyMT2ProductionTrace(
                        region_id=item.region_id,
                        original_source=item.source,
                        normalized_input=prepared.normalized_source,
                        protected_input=prepared.prepared_text,
                        protected_terms=protected_terms,
                        raw_hy_output="[SYSTEM_UI_BYPASS]",
                        stripped_output=prepared.system_ui_translation,
                        restored_output=prepared.system_ui_translation,
                        final_output=prepared.system_ui_translation,
                        guard_flags=[],
                        requires_review=False,
                        model_call_performed=False,
                        latency_sec=None,
                        pipeline_diagnosis="SYSTEM_UI_BYPASS",
                    )
                )
                continue

            if prepared.term_only_translation is not None:
                self.metrics.term_only_bypass_count += 1
                result = TranslationOutputItem(
                    region_id=item.region_id,
                    source=item.source,
                    translation=prepared.term_only_translation,
                    raw_model_response="[TERM_ONLY_BYPASS]",
                )
                results.append(result)
                raw_responses.append("[TERM_ONLY_BYPASS]")
                self.last_traces.append(
                    HyMT2ProductionTrace(
                        region_id=item.region_id,
                        original_source=item.source,
                        normalized_input=prepared.normalized_source,
                        protected_input=prepared.prepared_text,
                        protected_terms=protected_terms,
                        raw_hy_output="[TERM_ONLY_BYPASS]",
                        stripped_output=prepared.term_only_translation,
                        restored_output=prepared.term_only_translation,
                        final_output=prepared.term_only_translation,
                        guard_flags=[],
                        requires_review=False,
                        model_call_performed=False,
                        latency_sec=None,
                        pipeline_diagnosis="TERM_ONLY_BYPASS",
                    )
                )
                continue

            request_started = time.perf_counter()
            raw_text, cleaned, error_occurred = self._request_translation(
                prepared.prepared_text, label=f"item {item.region_id}"
            )
            latency = time.perf_counter() - request_started
            raw_responses.append(raw_text)

            if error_occurred or not cleaned:
                warning = "translation_server_error" if error_occurred else "empty_translation"
                result = TranslationOutputItem(
                    region_id=item.region_id,
                    source=item.source,
                    translation=None,
                    raw_model_response=raw_text[:500] if raw_text else "[Server Error]",
                    validation_warnings=[warning],
                    requires_review=True,
                )
                restored = None
                diagnosis = "PROVIDER_FAILURE" if error_occurred else "EMPTY_MODEL_OUTPUT"
            else:
                restored = restore_protected_translation(cleaned, prepared.placeholder_map)
                result = self._finalize_prepared_item(prepared, cleaned, raw_text)
                diagnosis = (
                    "GUARD_REVIEW:" + ",".join(result.validation_warnings)
                    if result.validation_warnings
                    else "MODEL_OUTPUT_ACCEPTED"
                )

            results.append(result)
            self.last_traces.append(
                HyMT2ProductionTrace(
                    region_id=item.region_id,
                    original_source=item.source,
                    normalized_input=prepared.normalized_source,
                    protected_input=prepared.prepared_text,
                    protected_terms=protected_terms,
                    raw_hy_output=raw_text,
                    stripped_output=cleaned,
                    restored_output=restored,
                    final_output=result.translation,
                    guard_flags=list(result.validation_warnings),
                    requires_review=result.requires_review,
                    model_call_performed=True,
                    latency_sec=round(latency, 6),
                    pipeline_diagnosis=diagnosis,
                )
            )

        return TranslationOutput(
            inputs=inp,
            results=results,
            raw_response="\n---\n".join(raw_responses)[:2000],
            repair_model=self.name,
        )


__all__ = [
    "DEFAULT_HY_MT2_MODEL_PATH",
    "DEFAULT_LLAMA_SERVER_PATH",
    "DEFAULT_HY_MT2_SERVER_URL",
    "DEFAULT_HY_MT2_SERVER_ALIAS",
    "HY_MT2_NATIVE_PROMPT_TEMPLATE",
    "HyMT2GGUFMetrics",
    "HyMT2ProductionTrace",
    "HyMT2GGUFTranslationProvider",
    "render_hy_mt2_prompt",
    "clean_hy_mt2_output",
]
