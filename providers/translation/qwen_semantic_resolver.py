"""Text-only Qwen GGUF semantic resolver backed by managed llama.cpp."""
from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from loguru import logger

from core.translation.semantic_context import (
    SemanticContextRequest,
    render_semantic_resolver_prompt,
)


DEFAULT_QWEN_SEMANTIC_MODEL_PATH = (
    r"C:\AI\Models\Qwen3.5-9B-GGUF\Qwen3.5-9B-Q5_K_M.gguf"
)
DEFAULT_QWEN_SEMANTIC_LLAMA_EXE_PATH = r"C:\AI\llama-cpp-cuda\llama.exe"
DEFAULT_QWEN_SEMANTIC_SERVER_URL = "http://127.0.0.1:8082"
DEFAULT_QWEN_SEMANTIC_CONTEXT_SIZE = 4096
DEFAULT_QWEN_SEMANTIC_GPU_LAYERS = 99
DEFAULT_QWEN_SEMANTIC_TEMPERATURE = 0.0
DEFAULT_QWEN_SEMANTIC_ALIAS = "qwen3.5-9b-semantic-resolver"
MAX_RESOLVER_NEW_TOKENS = 512

_TRANSLATION_RISK_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "rewrite_needed": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "risk_types": {
            "type": "array",
            "items": {"type": "string", "pattern": "^[a-z][a-z0-9_]{0,63}$"},
        },
        "semantic_notes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "span": {"type": "string"},
                    "resolved_meaning": {"type": "string"},
                    "evidence": {"type": "string"},
                },
                "required": ["span", "resolved_meaning", "evidence"],
                "additionalProperties": False,
            },
        },
        "question_type": {"type": ["string", "null"]},
        "tense_aspect": {"type": ["string", "null"]},
        "referents": {"type": "array"},
        "clarified_target": {"type": "string"},
    },
    "required": [
        "rewrite_needed",
        "confidence",
        "risk_types",
        "semantic_notes",
        "question_type",
        "tense_aspect",
        "referents",
        "clarified_target",
    ],
    "additionalProperties": False,
}


_CONTROLLED_ENGLISH_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "rewrite_needed": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "risk_types": {
            "type": "array",
            "items": {"type": "string", "pattern": "^[a-z][a-z0-9_]{0,63}$"},
        },
        "semantic_notes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "span": {"type": "string"},
                    "resolved_meaning": {"type": "string"},
                    "evidence": {"type": "string"},
                },
                "required": ["span", "resolved_meaning", "evidence"],
                "additionalProperties": False,
            },
        },
        "question_word": {"type": ["string", "null"]},
        "tense_aspect": {"type": ["string", "null"]},
        "referents": {"type": "array"},
        "controlled_target": {"type": "string"},
    },
    "required": [
        "rewrite_needed",
        "confidence",
        "risk_types",
        "semantic_notes",
        "question_word",
        "tense_aspect",
        "referents",
        "controlled_target",
    ],
    "additionalProperties": False,
}


@dataclass
class QwenSemanticResolverMetrics:
    resolver_calls: int = 0
    resolver_failures: int = 0
    reasoning_contamination_count: int = 0
    model_load_seconds: float = 0.0
    model_load_vram_gb: float = 0.0
    peak_vram_gb: float = 0.0
    input_token_count: int = 0
    generated_token_count: int = 0
    generation_seconds: float = 0.0
    server_generation_seconds: float = 0.0

    @property
    def average_resolver_seconds(self) -> float:
        return (
            self.generation_seconds / self.resolver_calls
            if self.resolver_calls
            else 0.0
        )

    @property
    def tokens_per_second(self) -> float:
        return (
            self.generated_token_count / self.server_generation_seconds
            if self.server_generation_seconds
            else 0.0
        )


class QwenSemanticResolverProvider:
    """Resolve English semantics through Qwen GGUF; never translate to Turkish."""

    backend = "qwen3.5-9b-gguf-llamacpp"
    quantization = "Q5_K_M"
    chat_template_strategy = "embedded GGUF template via /v1/chat/completions"
    reasoning_mode = "off"

    def __init__(
        self,
        model_path: str = DEFAULT_QWEN_SEMANTIC_MODEL_PATH,
        executable_path: str = DEFAULT_QWEN_SEMANTIC_LLAMA_EXE_PATH,
        server_url: str = DEFAULT_QWEN_SEMANTIC_SERVER_URL,
        managed: bool = True,
        max_context_length: int = DEFAULT_QWEN_SEMANTIC_CONTEXT_SIZE,
        gpu_layers: int = DEFAULT_QWEN_SEMANTIC_GPU_LAYERS,
        max_new_tokens: int = MAX_RESOLVER_NEW_TOKENS,
        temperature: float = DEFAULT_QWEN_SEMANTIC_TEMPERATURE,
        model_alias: str = DEFAULT_QWEN_SEMANTIC_ALIAS,
        startup_timeout: float = 120.0,
        request_timeout: float = 300.0,
        prompt_renderer=render_semantic_resolver_prompt,
        json_schema: dict[str, Any] | None = None,
    ) -> None:
        parsed = urlparse(server_url.rstrip("/"))
        if parsed.scheme != "http" or not parsed.hostname or not parsed.port:
            raise ValueError("server_url must be an explicit http://host:port URL")
        if parsed.path not in {"", "/"}:
            raise ValueError("server_url must not contain an API path")
        if max_context_length <= 0 or gpu_layers < 0 or max_new_tokens <= 0:
            raise ValueError("context size, GPU layers, and max tokens must be positive")
        if temperature != 0.0:
            raise ValueError("Semantic resolver temperature must remain deterministic at 0.0")

        self.model_path = model_path
        self.executable_path = executable_path
        self.server_url = server_url.rstrip("/")
        self.managed = managed
        self.max_context_length = max_context_length
        self.gpu_layers = gpu_layers
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.model_alias = model_alias
        self.startup_timeout = startup_timeout
        self.request_timeout = request_timeout
        self.prompt_renderer = prompt_renderer
        self.json_schema = json_schema or _CONTROLLED_ENGLISH_JSON_SCHEMA
        self._host = parsed.hostname
        self._port = parsed.port

        self._process: subprocess.Popen | None = None
        self._owned_process = False
        self._loaded = False
        self.metrics = QwenSemanticResolverMetrics()
        self.last_prompt = ""
        self.last_props: dict[str, Any] = {}
        self.last_server_command: tuple[str, ...] = ()
        self.port_closed_after_unload: bool | None = None

    @property
    def name(self) -> str:
        return "Qwen3.5-9B-GGUF-Q5_K_M-llama.cpp"

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def owned_process(self) -> bool:
        return self._owned_process

    @property
    def chat_template_sha256(self) -> str | None:
        template = self.last_props.get("chat_template")
        if not isinstance(template, str) or not template:
            return None
        return hashlib.sha256(template.encode("utf-8")).hexdigest()

    @property
    def chat_template_preview(self) -> str | None:
        template = self.last_props.get("chat_template")
        if not isinstance(template, str) or not template:
            return None
        return " ".join(template.split())[:240]

    def _request_json(
        self,
        path: str,
        *,
        method: str = "GET",
        payload: dict[str, Any] | None = None,
        timeout: float = 5.0,
    ) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            f"{self.server_url}{path}",
            data=body,
            headers={"Content-Type": "application/json"} if body else {},
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                if response.status != 200:
                    raise RuntimeError(f"llama.cpp returned HTTP {response.status}")
                decoded = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"llama.cpp {path} returned HTTP {exc.code}: {detail[:500]}"
            ) from exc
        if not isinstance(decoded, dict):
            raise RuntimeError(f"llama.cpp {path} response must be a JSON object")
        return decoded

    def _probe_server(self) -> tuple[bool, dict[str, Any] | None]:
        try:
            self._request_json("/health", timeout=3.0)
        except Exception:
            return False, None
        try:
            return True, self._request_json("/props", timeout=3.0)
        except Exception:
            return True, None

    def _server_identity_matches(self, props: dict[str, Any]) -> bool:
        expected_basename = os.path.basename(self.model_path).casefold()
        loaded_path = str(props.get("model_path") or "")
        loaded_alias = str(props.get("model_alias") or "")
        if loaded_path:
            try:
                if os.path.normcase(os.path.abspath(loaded_path)) == os.path.normcase(
                    os.path.abspath(self.model_path)
                ):
                    return True
            except OSError:
                pass
            if os.path.basename(loaded_path).casefold() == expected_basename:
                return True
        return bool(loaded_alias and expected_basename in loaded_alias.casefold())

    def _check_health(self) -> bool:
        healthy, props = self._probe_server()
        return bool(healthy and props and self._server_identity_matches(props))

    def _port_is_open(self) -> bool:
        try:
            with socket.create_connection((self._host, self._port), timeout=0.5):
                return True
        except OSError:
            return False

    @staticmethod
    def _sample_gpu_memory_gb() -> float:
        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=3,
            )
            values = [
                float(line.strip())
                for line in completed.stdout.splitlines()
                if line.strip()
            ]
            return max(values, default=0.0) / 1024.0
        except Exception:
            return 0.0

    def _build_server_command(self) -> list[str]:
        return [
            self.executable_path,
            "serve",
            "-m",
            self.model_path,
            "-ngl",
            str(self.gpu_layers),
            "-c",
            str(self.max_context_length),
            "-np",
            "1",
            "--host",
            self._host,
            "--port",
            str(self._port),
            "--alias",
            self.model_alias,
            "--jinja",
            "--reasoning",
            "off",
            "--no-webui",
        ]

    def _wait_for_ready(self) -> dict[str, Any]:
        started = time.perf_counter()
        while time.perf_counter() - started < self.startup_timeout:
            if self._process is not None and self._process.poll() is not None:
                raise RuntimeError(
                    "Qwen llama.cpp server exited during startup with code "
                    f"{self._process.returncode}"
                )
            healthy, props = self._probe_server()
            if healthy and props is not None:
                if not self._server_identity_matches(props):
                    loaded = props.get("model_path") or props.get("model_alias")
                    raise RuntimeError(
                        f"Port {self._port} serves '{loaded}', not the expected Qwen GGUF"
                    )
                return props
            time.sleep(0.5)
        raise RuntimeError(
            f"Timeout waiting {self.startup_timeout:.0f}s for Qwen llama.cpp server"
        )

    def load(self) -> None:
        if self._loaded:
            return
        started = time.perf_counter()
        healthy, props = self._probe_server()
        if healthy:
            if props is None or not self._server_identity_matches(props):
                loaded = (props or {}).get("model_path") or (props or {}).get(
                    "model_alias"
                )
                raise RuntimeError(
                    f"Port {self._port} is occupied by unexpected model/server: {loaded}"
                )
            self.last_props = props
            self._loaded = True
            self.metrics.model_load_seconds = round(time.perf_counter() - started, 4)
            logger.info("Reusing expected Qwen llama.cpp server at {}", self.server_url)
            return
        if self._port_is_open():
            raise RuntimeError(
                f"Port {self._port} is occupied but does not expose expected llama.cpp health/props"
            )
        if not self.managed:
            raise RuntimeError(
                f"Expected unmanaged Qwen llama.cpp server is unavailable at {self.server_url}"
            )
        if not os.path.isfile(self.executable_path):
            raise FileNotFoundError(f"llama.exe not found at {self.executable_path}")
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"Qwen GGUF not found at {self.model_path}")

        command = self._build_server_command()
        self.last_server_command = tuple(command)
        logger.info("Starting managed Qwen llama.cpp server: {}", " ".join(command))
        self._process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=(
                getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
            ),
        )
        self._owned_process = True
        try:
            self.last_props = self._wait_for_ready()
        except Exception:
            self.unload()
            raise

        self._loaded = True
        self.metrics.model_load_seconds = round(time.perf_counter() - started, 4)
        sampled_vram = self._sample_gpu_memory_gb()
        self.metrics.model_load_vram_gb = sampled_vram
        self.metrics.peak_vram_gb = max(self.metrics.peak_vram_gb, sampled_vram)
        logger.info(
            "Qwen GGUF resolver loaded in {:.2f}s; sampled GPU use {:.2f} GiB",
            self.metrics.model_load_seconds,
            sampled_vram,
        )

    def unload(self) -> None:
        self._loaded = False
        if not self._owned_process or self._process is None:
            self._process = None
            self._owned_process = False
            self.port_closed_after_unload = None
            return

        process = self._process
        logger.info("Terminating owned Qwen llama.cpp server process PID {}", process.pid)
        try:
            process.terminate()
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        finally:
            self._process = None
            self._owned_process = False

        deadline = time.perf_counter() + 5.0
        while self._port_is_open() and time.perf_counter() < deadline:
            time.sleep(0.1)
        self.port_closed_after_unload = not self._port_is_open()
        if not self.port_closed_after_unload:
            logger.warning(
                "Owned Qwen process exited, but port {} remains open; "
                "no unrelated process was killed",
                self._port,
            )

    def _chat_payload(self, prompt: str) -> dict[str, Any]:
        return {
            "model": self.model_alias,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "seed": 0,
            "max_tokens": self.max_new_tokens,
            "stream": False,
            "cache_prompt": True,
            "timings_per_token": False,
            "json_schema": self.json_schema,
        }

    def _query_chat_completion(
        self,
        prompt: str,
    ) -> tuple[str, int, int, float, float]:
        response = self._request_json(
            "/v1/chat/completions",
            method="POST",
            payload=self._chat_payload(prompt),
            timeout=self.request_timeout,
        )
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("llama.cpp chat response has no choices")
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise RuntimeError("llama.cpp chat response has no text content")
        reasoning = message.get("reasoning_content")
        content = message["content"].strip()
        if reasoning or "<think" in content.casefold() or "<thinking" in content.casefold():
            self.metrics.reasoning_contamination_count += 1
            raise RuntimeError("Qwen reasoning content contaminated the JSON response")

        usage_value = response.get("usage")
        usage: dict[str, Any] = usage_value if isinstance(usage_value, dict) else {}
        timings_value = response.get("timings")
        timings: dict[str, Any] = (
            timings_value if isinstance(timings_value, dict) else {}
        )
        input_tokens = int(
            usage.get("prompt_tokens")
            or (int(timings.get("prompt_n") or 0) + int(timings.get("cache_n") or 0))
        )
        generated_tokens = int(
            usage.get("completion_tokens") or timings.get("predicted_n") or 0
        )
        server_generation_seconds = float(timings.get("predicted_ms") or 0.0) / 1000.0
        server_tokens_per_second = float(timings.get("predicted_per_second") or 0.0)
        return (
            content,
            input_tokens,
            generated_tokens,
            server_generation_seconds,
            server_tokens_per_second,
        )

    def resolve(self, request: SemanticContextRequest) -> str:
        """Return one raw strict-JSON resolver response through native Qwen chat."""
        if not self._loaded:
            self.load()
        prompt = self.prompt_renderer(request)
        self.last_prompt = prompt
        self.metrics.resolver_calls += 1
        started = time.perf_counter()
        try:
            (
                content,
                input_tokens,
                generated_tokens,
                server_generation_seconds,
                _,
            ) = self._query_chat_completion(prompt)
        except Exception:
            self.metrics.resolver_failures += 1
            raise
        wall_seconds = time.perf_counter() - started
        self.metrics.input_token_count += input_tokens
        self.metrics.generated_token_count += generated_tokens
        self.metrics.generation_seconds += wall_seconds
        self.metrics.server_generation_seconds += server_generation_seconds
        self.metrics.peak_vram_gb = max(
            self.metrics.peak_vram_gb,
            self._sample_gpu_memory_gb(),
        )
        return content
