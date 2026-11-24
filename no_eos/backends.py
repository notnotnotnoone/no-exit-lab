"""Where next-token distributions come from: a real Ollama server, or a fake for demos."""

from __future__ import annotations

import math
import random
import time

import httpx

# Never matches anything. Replaces the model's built-in stop strings (e.g.
# "<|im_end|>") so Ollama doesn't cut output on our behalf.
_NEVER = "␞␞__no_exit__␞␞"


class OllamaBackend:
    """Asks Ollama for exactly one token, in raw mode, and keeps only the top-20
    logprobs. Ollama's own pick is thrown away; the engine samples by itself."""

    def __init__(self, model: str, host: str = "http://127.0.0.1:11434", num_ctx: int = 32768, top_k: int = 20):
        self.model = model
        self.name = model
        self.num_ctx = num_ctx
        self.top_k = top_k
        self.client = httpx.Client(base_url=host, timeout=httpx.Timeout(600.0, connect=10.0))
        self.last_raw: dict | None = None

    def next_dist(self, prompt: str, seed: int, temperature: float) -> list[tuple[str, float]] | None:
        body = {
            "model": self.model,
            "prompt": prompt,
            "raw": True,
            "stream": False,
            "logprobs": True,
            "top_logprobs": self.top_k,
            "keep_alive": -1,
            "options": {
                "num_predict": 1,
                "num_ctx": self.num_ctx,
                "temperature": temperature,
                "top_k": 40,
                "top_p": 1.0,
                "seed": seed,
                "stop": [_NEVER],
                # Our engine handles repetition itself; keep Ollama's sampler neutral.
                "repeat_penalty": 1.0,
            },
        }
        r = self.client.post("/api/generate", json=body)
        if r.status_code >= 400:
            raise RuntimeError(f"Ollama {r.status_code}: {r.text[:300]}")
        d = r.json()
        self.last_raw = d
        lps = d.get("logprobs") or []
        if not lps:
            return None
        entry = lps[0]
        tops = entry.get("top_logprobs") or [entry]
        return [(t.get("token", ""), float(t.get("logprob", -99.0))) for t in tops]


class FakeBackend:
    """A pretend model for previewing the TUI without a GPU. It loves to repeat
    itself and wants to stop after every sentence, so both rules get exercised."""

    name = "fake-model (demo)"
    WORDS = (
        "the of and to in a is that it for on as with was be by this are or from at an "
        "machine dream silence river memory light signal ocean forget endless thought "
        "neuron static echo garden mirror lantern circuit window whisper horizon clock "
        "strange quietly forever again perhaps somewhere nothing everything beneath "
        "beyond within language token meaning ending door story begin cannot stop"
    ).split()

    def __init__(self, speed: float = 25.0, seed: int = 7):
        self.delay = 1.0 / speed
        self.rng = random.Random(seed)

    def next_dist(self, prompt: str, seed: int, temperature: float) -> list[tuple[str, float]] | None:
        time.sleep(self.delay)
        rng = self.rng
        body = prompt.rsplit("<|im_start|>assistant\n", 1)[-1]
        words = body.replace("<think>", " ").replace("</think>", " ").split()
        in_think = "<think>" in body and "</think>" not in body and "<think>\n\n</think>" not in prompt

        pool = rng.sample(self.WORDS, 17)
        scores = {f" {w}": -abs(rng.gauss(0, 1.6)) - i * 0.18 for i, w in enumerate(pool)}
        # The fake loves to continue a phrase it already said.
        if len(words) > 6 and rng.random() < 0.55:
            i = rng.randrange(0, len(words) - 2)
            scores[f" {words[i + 1]}"] = 0.6
        scores["."] = -1.8 + (0.4 if len(words) % 9 > 6 else -1.0)
        if in_think and len(words) > 40:
            scores["</think>"] = 0.8
        end_lp = -6.0
        if body.rstrip().endswith("."):
            end_lp = rng.choice([-0.2, 0.3, -2.0, 1.0])
        if not in_think:
            scores["<|im_end|>"] = end_lp
            scores["<|endoftext|>"] = end_lp - 2.5
        z = math.log(sum(math.exp(v) for v in scores.values()))
        out = sorted(((t, v - z) for t, v in scores.items()), key=lambda x: -x[1])[:20]
        return out
