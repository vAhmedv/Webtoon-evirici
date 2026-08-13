"""Qwen visual OCR repair / adjudication provider using llama-server multimodal REST API.

Qwen3.5-9B GGUF + mmproj vision model bubble crop görüntüsünü okur ve
OCR disagreement'ı çözmek için llama-server REST API (port 8086) üzerinden çalışır.

Görevi: Crop'taki gerçek İngilizce metni belirlemek.
Translator değildir. OCR adayları sadece ipucudur.

Model yalnızca ``verdict.needs_repair == True`` olduğunda çalışır.
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any
import urllib.parse
import urllib.request

from loguru import logger
from PIL import Image

from providers.ocr.repair import OCRRepairInput, OCRRepairProvider, OCRRepairResult

DEFAULT_QWEN_GGUF_MODEL_PATH = r"C:\AI\Models\Qwen3.5-9B-GGUF\Qwen3.5-9B-Q5_K_M.gguf"
DEFAULT_QWEN_MMPROJ_PATH = r"C:\AI\Models\Qwen3.5-9B-GGUF\mmproj-F16.gguf"
DEFAULT_LLAMA_SERVER_PATH = r"C:\AI\llama-cpp-cuda\llama-server.exe"
DEFAULT_QWEN_SERVER_URL = "http://127.0.0.1:8086"
DEFAULT_QWEN_SERVER_PORT = 8086
MAX_NEW_TOKENS = 512

# Kısa, odaklı prompt. JSON talep eder ama model doğal dil de çıkarabilir; parser buna göre esnek çalışır.
_USER_PROMPT = (
    "Read the English text in the provided bubble crop image.\n"
    "Two OCR candidates disagree:\n"
    "A: {primary}\n"
    "B: {verifier}\n"
    "Disagreement: {reason}\n"
    "{glossary}{context}\n"
    "Output JSON: {{\"status\": \"resolved\", \"text\": \"<text or null>\", \"reason\": \"<why>\"}}"
)


@dataclass(frozen=True)
class QwenRepairConfig:
    model_path: str = DEFAULT_QWEN_GGUF_MODEL_PATH
    mmproj_path: str = DEFAULT_QWEN_MMPROJ_PATH
    server_path: str = DEFAULT_LLAMA_SERVER_PATH
    server_url: str = DEFAULT_QWEN_SERVER_URL
    server_port: int = DEFAULT_QWEN_SERVER_PORT
    n_gpu_layers: int = 99
    max_new_tokens: int = MAX_NEW_TOKENS
    max_memory_gb: int = 12


@dataclass(frozen=True)
class QwenRepairInput:
    image: Image.Image
    primary_raw: str
    primary_normalized: str
    verifier_raw: str
    verifier_normalized: str
    reason: str
    known_names: list[str] = field(default_factory=list)
    nearby_context: str | None = None

    @classmethod
    def from_repair_input(cls, ri: OCRRepairInput, image: Image.Image) -> "QwenRepairInput":
        return cls(
            image=image,
            primary_raw=ri.primary_raw,
            primary_normalized=ri.primary_normalized,
            verifier_raw=ri.verifier_raw,
            verifier_normalized=ri.verifier_normalized,
            reason=ri.reason,
            known_names=list(ri.known_names),
            nearby_context=ri.nearby_context,
        )


@dataclass
class QwenRepairMetrics:
    model_load_vram_gb: float = 0.0
    peak_vram_gb: float = 0.0
    repair_model: str = ""


@dataclass
class OCRAdjudicatedResult:
    verdict: Any
    clean_source_text: str | None
    repair_result: OCRRepairResult | None = None
    repair_model: str | None = None
    requires_review: bool = False


class QwenRepairProvider(OCRRepairProvider):
    """Qwen3.5-9B GGUF visual OCR repair using llama-server multimodal API.

    llama-server.exe süreci load() sırasında başlatılır (--mmproj ile),
    unload() çağrıldığında sonlandırılır.
    """

    def __init__(self, config: QwenRepairConfig | None = None) -> None:
        self._config = config or QwenRepairConfig()
        self._loaded = False
        self._server_process: subprocess.Popen | None = None
        self.metrics = QwenRepairMetrics()

    @property
    def name(self) -> str:
        return "Qwen3.5-9B-OCR-Repair-GGUF"

    @property
    def version(self) -> str:
        return "3.5-9B-GGUF"

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def device(self) -> str:
        return "cuda"

    def _find_mmproj_file(self, model_path: Path) -> Path | None:
        """Sırasıyla mmproj yolunu ve model dizinindeki mmproj*.gguf dosyalarını arar."""
        configured = Path(self._config.mmproj_path)
        if configured.exists():
            return configured

        # Model klasöründe mmproj ara
        search_dir = model_path.parent if model_path.is_file() else model_path
        if search_dir.exists():
            matches = list(search_dir.glob("*mmproj*.gguf"))
            if matches:
                return matches[0]
            # Genel mmproj araması
            matches = list(search_dir.glob("mmproj*"))
            if matches:
                return matches[0]

        return None

    def load(self) -> None:
        if self._loaded:
            return

        model_path = Path(self._config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Qwen GGUF model path not found: {model_path}")

        mmproj_path = self._find_mmproj_file(model_path)
        if mmproj_path is None or not mmproj_path.exists():
            raise FileNotFoundError(
                f"Qwen multimodal vision projector (mmproj) GGUF file not found in {model_path.parent} or {self._config.mmproj_path}"
            )

        server_bin = Path(self._config.server_path)
        if not server_bin.exists():
            raise FileNotFoundError(f"llama-server binary not found: {server_bin}")

        # Server sunucu durumunu kontrol et
        if self._check_health():
            self._loaded = True
            self.metrics.repair_model = f"Qwen-GGUF-llama-server:{self._config.server_port}"
            logger.info(f"Existing Qwen llama-server reusable on port {self._config.server_port}")
            return

        # Server sürecini başlat
        cmd = [
            str(server_bin),
            "-m", str(model_path),
            "--mmproj", str(mmproj_path),
            "--port", str(self._config.server_port),
            "-ngl", str(self._config.n_gpu_layers),
            "--alias", "qwen-ocr-repair",
        ]

        logger.info(f"Starting Qwen llama-server: {' '.join(cmd)}")
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        out_f = open(log_dir / "qwen_llama_server.log", "w", encoding="utf-8")

        self._server_process = subprocess.Popen(
            cmd,
            stdout=out_f,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )

        # Sağlık kontörlü bekle
        started = False
        deadline = time.time() + 30.0
        while time.time() < deadline:
            if self._server_process.poll() is not None:
                out_f.close()
                raise RuntimeError(f"Qwen llama-server process exited unexpectedly with code {self._server_process.poll()}")
            if self._check_health():
                started = True
                break
            time.sleep(0.5)

        if not started:
            self.unload()
            raise TimeoutError(f"Qwen llama-server health check timed out on port {self._config.server_port}")

        self._loaded = True
        self.metrics.repair_model = f"Qwen-GGUF-llama-server:{self._config.server_port}"
        logger.info(f"Qwen llama-server started successfully on port {self._config.server_port}")

    def _check_health(self) -> bool:
        url = f"{self._config.server_url}/health"
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        return False

    def unload(self) -> None:
        if self._server_process is not None:
            try:
                self._server_process.terminate()
                self._server_process.wait(timeout=5.0)
            except Exception:
                try:
                    self._server_process.kill()
                except Exception:
                    pass
            self._server_process = None

        self._loaded = False
        logger.info("Qwen llama-server process terminated and unloaded")

    def _build_prompt(self, inp: QwenRepairInput) -> str:
        primary = inp.primary_normalized or inp.primary_raw
        verifier = inp.verifier_normalized or inp.verifier_raw
        return _USER_PROMPT.format(
            primary=primary,
            verifier=verifier,
            reason=inp.reason,
            glossary=(f"Glossary: {', '.join(inp.known_names)}\n" if inp.known_names else ""),
            context=(f"Context: {inp.nearby_context}\n" if inp.nearby_context else ""),
        )

    @staticmethod
    def _image_to_base64_url(image: Image.Image) -> str:
        pil_img = image.convert("RGB") if isinstance(image, Image.Image) else Image.fromarray(image).convert("RGB")
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=95)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"

    def repair(self, repair_input: OCRRepairInput, image: Image.Image) -> OCRRepairResult:
        if not self._loaded:
            raise RuntimeError("QwenRepairProvider not loaded; call load() first")

        inp = QwenRepairInput.from_repair_input(repair_input, image)
        prompt = self._build_prompt(inp)
        image_url = self._image_to_base64_url(image)

        payload = {
            "messages": [
                {"role": "system", "content": "You are a JSON-only OCR adjudicator. Output ONLY a JSON object."},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
            "temperature": 0.1,
            "max_tokens": self._config.max_new_tokens,
        }

        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self._config.server_url}/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=30.0) as resp:
                res_data = json.loads(resp.read().decode("utf-8"))
                raw_text = res_data["choices"][0]["message"]["content"].strip()
                logger.info(f"Qwen raw response ({len(raw_text)} chars): {raw_text[:200]}")
                return self._parse_output(raw_text)
        except Exception as e:
            logger.error(f"Qwen llama-server repair API call failed: {e}")
            return self._unresolved(f"api_error: {e}")

    # --- Parser: handles both JSON and natural language output ---
    def _parse_output(self, raw: str) -> OCRRepairResult:
        if not raw or len(raw) < 2:
            return self._unresolved("empty_output")

        # 1. Try JSON: find first { ... last }
        data = self._try_json(raw)
        if data is not None:
            status = str(data.get("status", "")).strip().lower()
            text = data.get("text")
            reason = str(data.get("reason", "")).strip()

            if status == "unresolved":
                return OCRRepairResult(
                    repaired_text=None, changed=False, unresolved=True,
                    metadata={"repair_model": self.metrics.repair_model,
                              "repair_reason": reason or "insufficient_visual_evidence",
                              "raw_output": raw[:500]},
                )

            if status == "resolved" and text is not None:
                repaired = str(text).strip()
                if not repaired or repaired == "null":
                    return self._unresolved("empty_text")
                if self._looks_like_placeholder(repaired):
                    return self._unresolved("placeholder_in_text")
                return OCRRepairResult(
                    repaired_text=repaired, changed=True, unresolved=False,
                    metadata={"repair_model": self.metrics.repair_model,
                              "repair_reason": reason or "visual_evidence",
                              "raw_output": raw[:500]},
                )

        # 2. Fallback: no valid JSON. Check for explicit "unresolved" verdict.
        if re.search(r'\bunresolved\b', raw, re.IGNORECASE):
            return self._unresolved("insufficient_visual_evidence")

        # 3. Fallback: extract text from model analysis output.
        extracted = self._extract_text_from_natural_language(raw)
        if extracted:
            return OCRRepairResult(
                repaired_text=extracted, changed=True, unresolved=False,
                metadata={"repair_model": self.metrics.repair_model,
                          "repair_reason": "visual_evidence",
                          "raw_output": raw[:500]},
            )

        return self._unresolved("no_structured_output")

    @staticmethod
    def _try_json(text: str) -> dict[str, Any] | None:
        first = text.find("{")
        last = text.rfind("}")
        if first < 0 or last <= first:
            return None
        for candidate in [text[first : last + 1], text.strip()]:
            try:
                obj = json.loads(candidate)
                if isinstance(obj, dict) and "status" in obj:
                    return obj
            except json.JSONDecodeError:
                pass
        return None

    @staticmethod
    def _looks_like_placeholder(text: str) -> bool:
        markers = ["<exact text>", "<why>", "<text>", "<reason>"]
        return any(m in text for m in markers)

    @staticmethod
    def _extract_text_from_natural_language(text: str) -> str | None:
        line_matches = re.findall(r'[Ll]ine\s*\d*:\s*"([^"]+)"', text)
        if line_matches:
            full = " ".join(m.strip() for m in line_matches)
            if len(full) >= 2:
                return full

        cand_matches = re.findall(r'Candidate\s+[AB]:\s*"([^"]+)"', text)
        if cand_matches:
            return cand_matches[-1].strip()

        reads_match = re.search(r'(?:text\s+)?reads:\s*["\']?(.+?)(?:["\']|$)', text, re.IGNORECASE | re.DOTALL)
        if reads_match:
            val = reads_match.group(1).strip().strip('"').strip("'")
            if val and len(val) >= 2:
                return val

        correct_match = re.search(r'(?:correct|right|should be):\s*["\']?([^"\']+)["\']?', text, re.IGNORECASE)
        if correct_match:
            val = correct_match.group(1).strip()
            if val and len(val) >= 2:
                return val

        quoted = re.findall(r'"([^"]{3,})"', text)
        if quoted:
            longest = max(quoted, key=len).strip()
            if len(longest) >= 2:
                return longest

        return None

    def _unresolved(self, reason: str) -> OCRRepairResult:
        return OCRRepairResult(
            repaired_text=None, changed=False, unresolved=True,
            metadata={"repair_model": self.metrics.repair_model,
                      "repair_reason": reason, "raw_output": ""},
        )

    def _parse_repair_output(self, raw_output: str) -> OCRRepairResult:
        """Public wrapper for _parse_output (used by tests)."""
        return self._parse_output(raw_output)


def adjudicate_ocr(
    verdict: "OCRVerdict",
    crop_image: Image.Image | None,
    repair_provider: QwenRepairProvider,
) -> OCRAdjudicatedResult:
    """OCR verdict'ı Qwen repair ile çözer.

    - needs_repair=False -> Qwen çağrılmaz, accepted_text = clean_source_text.
    - needs_repair=True -> Qwen repair çalıştırılır.
    """
    from providers.ocr.agreement import OCRVerdict

    if not verdict.needs_repair:
        logger.debug("Qwen skip: safe agreement")
        return OCRAdjudicatedResult(
            verdict=verdict,
            clean_source_text=verdict.accepted_text,
            repair_result=None,
            repair_model=None,
            requires_review=False,
        )

    if crop_image is None or not repair_provider.is_loaded:
        logger.warning("needs_repair=True ama Qwen skip (no crop/provider not loaded)")
        return OCRAdjudicatedResult(
            verdict=verdict,
            clean_source_text=None,
            repair_result=None,
            repair_model=None,
            requires_review=True,
        )

    ri = OCRRepairInput(
        primary_raw=verdict.primary_raw,
        primary_normalized=verdict.primary_normalized,
        verifier_raw=verdict.verifier_raw,
        verifier_normalized=verdict.verifier_normalized,
        reason=verdict.reason or "word_difference",
    )

    logger.info(
        f"Qwen repair: {verdict.reason}, "
        f"primary='{verdict.primary_normalized[:40]}', "
        f"verifier='{verdict.verifier_normalized[:40]}'"
    )

    result = repair_provider.repair(ri, crop_image)
    updated = replace(verdict, repaired_text=result.repaired_text)

    if result.unresolved:
        return OCRAdjudicatedResult(
            verdict=updated, clean_source_text=None,
            repair_result=result,
            repair_model=result.metadata.get("repair_model"),
            requires_review=True,
        )
    return OCRAdjudicatedResult(
        verdict=updated,
        clean_source_text=result.repaired_text,
        repair_result=result,
        repair_model=result.metadata.get("repair_model"),
        requires_review=False,
    )
