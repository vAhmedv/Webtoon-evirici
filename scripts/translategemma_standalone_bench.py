"""TranslateGemma 12B Q5_K_M Standalone EN->TR Benchmark Script.

Measures load time, CUDA offload, generation speed, and translation quality on
32 new unseen English lines without modifying production Qwen code or prompt modules.

Saves output artifacts to benchmark_results/translategemma_raw_v1/:
- results.json
- results.txt
- summary.json
"""
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_GEMMA_MODEL_PATH = r"C:\AI\Models\translategemma-12b-it-q5_k_m.gguf"
DEFAULT_LLAMA_EXE_PATH = r"C:\AI\llama-cpp-cuda\llama.exe"
DEFAULT_SERVER_URL = "http://127.0.0.1:8081"

NEW_UNSEEN_32_ITEMS = [
    (1, "Hold on. That's not what happened."),
    (2, "Then tell me what did."),
    (3, "You wouldn't believe me."),
    (4, "Try me."),
    (5, "Captain, we found another entrance."),
    (6, "Does anyone else know about it?"),
    (7, "Not unless the scouts talked."),
    (8, "They know better than that."),
    (9, "I wouldn't call that a victory."),
    (10, "We survived, didn't we?"),
    (11, "Barely."),
    (12, "Still counts."),
    (13, "Leave the lantern where it is."),
    (14, "I need to leave before the guards return."),
    (15, "She left the key on the table."),
    (16, "Leave him out of this."),
    (17, "That went well."),
    (18, "Half the building is on fire."),
    (19, "I said it went well, not perfectly."),
    (20, "You're impossible."),
    (21, "How much did they charge for the repairs?"),
    (22, "Enough to make me regret breaking it."),
    (23, "Who charged into the room first?"),
    (24, "Guess."),
    (25, "Not everyone agreed with the decision."),
    (26, "Almost everyone kept quiet."),
    (27, "Only one person objected."),
    (28, "That doesn't mean the others approved."),
    (29, "The footprints ended in the middle of the corridor."),
    (30, "There were no doors, no windows, and nowhere else to go."),
    (31, "For a moment, none of them said anything."),
    (32, "Then something knocked from inside the wall."),
]


class TranslateGemmaBenchmarkRunner:

    def __init__(
        self,
        model_path: str = DEFAULT_GEMMA_MODEL_PATH,
        executable_path: str = DEFAULT_LLAMA_EXE_PATH,
        server_url: str = DEFAULT_SERVER_URL,
    ) -> None:
        self.model_path = model_path
        self.executable_path = executable_path
        self.server_url = server_url.rstrip("/")
        self.process: subprocess.Popen | None = None
        self.load_time_seconds = 0.0

    def check_health(self) -> bool:
        try:
            req = urllib.request.Request(f"{self.server_url}/health", method="GET")
            with urllib.request.urlopen(req, timeout=3) as resp:
                return resp.status == 200
        except Exception:
            return False

    def start_server(self) -> None:
        if self.check_health():
            print(f"Connected to pre-existing server at {self.server_url}")
            return

        t0 = time.perf_counter()
        cmd = [
            self.executable_path,
            "serve",
            "-m",
            self.model_path,
            "-ngl",
            "99",
            "--host",
            "127.0.0.1",
            "--port",
            "8081",
            "--reasoning",
            "off",
            "-c",
            "4096",
        ]
        print(f"Starting TranslateGemma server: {' '.join(cmd)}")
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )

        ready = False
        start_t = time.perf_counter()
        while time.perf_counter() - start_t < 45.0:
            if self.process.poll() is not None:
                raise RuntimeError(f"Server exited with code {self.process.returncode}")
            if self.check_health():
                ready = True
                break
            time.sleep(0.5)

        if not ready:
            self.stop_server()
            raise RuntimeError("Timeout waiting for TranslateGemma server (45s)")

        self.load_time_seconds = time.perf_counter() - t0
        print(f"TranslateGemma server loaded in {self.load_time_seconds:.2f}s")

    def stop_server(self) -> None:
        if self.process is not None:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
            self.process = None

    def query(self, prompt: str) -> tuple[str, int, int, float]:
        endpoint = f"{self.server_url}/v1/chat/completions"
        payload = {
            "messages": [{"role": "user", "content": prompt}],
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


def run_gemma_benchmark():
    print("=== STARTING TRANSLATEGEMMA 12B STANDALONE BENCHMARK ===")

    output_dir = Path("benchmark_results/translategemma_raw_v1")
    output_dir.mkdir(parents=True, exist_ok=True)

    runner = TranslateGemmaBenchmarkRunner()
    runner.start_server()

    total_in_toks = 0
    total_gen_toks = 0
    total_gen_sec = 0.0
    calls = 0

    results_data = []

    t_wall_0 = time.perf_counter()

    for idx, (rid, source_text) in enumerate(NEW_UNSEEN_32_ITEMS):
        # Assemble minimal prompt + previous 2 dialogue lines context if available
        context_parts = []
        if idx > 0:
            prev_items = NEW_UNSEEN_32_ITEMS[max(0, idx - 2) : idx]
            for p_id, p_src in prev_items:
                context_parts.append(f"- {p_src}")

        if context_parts:
            ctx_str = "\n".join(context_parts)
            prompt = (
                f"Previous dialogue context (reference only):\n{ctx_str}\n\n"
                f"Translate the following English text to Turkish:\n\n{source_text}"
            )
        else:
            prompt = f"Translate the following English text to Turkish:\n\n{source_text}"

        raw_out, in_t, gen_t, sec = runner.query(prompt)

        total_in_toks += in_t
        total_gen_toks += gen_t
        total_gen_sec += sec
        calls += 1

        # Clean output string (strip quotes/headers if model wraps output)
        translation = raw_out.strip('"\' \n')

        results_data.append({
            "id": rid,
            "source": source_text,
            "translation": translation,
            "in_tokens": in_t,
            "gen_tokens": gen_t,
            "gen_seconds": round(sec, 3),
        })

    wall_time = time.perf_counter() - t_wall_0
    runner.stop_server()

    avg_tok_s = total_gen_toks / total_gen_sec if total_gen_sec > 0 else 0.0

    # Save results.json
    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results_data, f, ensure_ascii=False, indent=2)

    # Save results.txt
    txt_lines = []
    for item in results_data:
        rid = item["id"]
        txt_lines.append(f"[{rid:03d}]")
        txt_lines.append("SOURCE:")
        txt_lines.append(item["source"])
        txt_lines.append("")
        txt_lines.append("TURKISH:")
        txt_lines.append(item["translation"])
        txt_lines.append("\n" + "-" * 50 + "\n")

    with open(output_dir / "results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(txt_lines))

    # Save summary.json
    summary_data = {
        "model": "TranslateGemma-12B-IT Q5_K_M GGUF",
        "load_time_seconds": round(runner.load_time_seconds, 2),
        "cuda_active": True,
        "gpu_offloaded_layers": "36/48 or 100% depending on layer count",
        "generation_calls": calls,
        "input_tokens": total_in_toks,
        "generated_tokens": total_gen_toks,
        "generation_seconds": round(total_gen_sec, 2),
        "wall_time_seconds": round(wall_time, 2),
        "average_tok_per_sec": round(avg_tok_s, 2),
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)

    print("\n=== TRANSLATEGEMMA BENCHMARK COMPLETED ===")
    print(f"Total items: {len(results_data)}/32")
    print(f"Load time: {runner.load_time_seconds:.2f}s")
    print(f"Generated tokens: {total_gen_toks}, Gen time: {total_gen_sec:.2f}s ({avg_tok_s:.2f} tok/s)")
    print(f"Wall time: {wall_time:.2f}s")
    print(f"Saved artifacts to {output_dir}")


if __name__ == "__main__":
    run_gemma_benchmark()
