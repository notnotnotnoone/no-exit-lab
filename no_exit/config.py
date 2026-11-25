"""Top-level knobs. Environment variables override anything here (the notebook sets them)."""

import os

# ─── the prisoner ────────────────────────────────────────────────────────────────
# Gemma 4 (default)  "DavidAU/gemma-4-31B-it-The-DECKARD-HERETIC-UNCENSORED-Thinking" + "bf16" ≈ 62 GB
# Qwen (ChatML)      "Qwen/Qwen3-32B" + "bf16" ≈ 65.5 GB · "Qwen/Qwen2.5-72B-Instruct" + "4bit" ≈ 42 GB (no think)
# smoke tests only   "Qwen/Qwen3-8B" + "bf16" ≈ 16 GB
MODEL_ID = os.environ.get("NO_EXIT_MODEL", "DavidAU/gemma-4-31B-it-The-DECKARD-HERETIC-UNCENSORED-Thinking")
QUANTIZATION = os.environ.get("NO_EXIT_QUANT", "bf16")  # "bf16" | "4bit"

# ─── surgery ─────────────────────────────────────────────────────────────────────
STEER_LAYER_FRAC = 0.55  # steering hook depth (fraction of decoder layers)
LENS_FRACS = (0.25, 0.50, 0.75)  # logit-lens checkpoints (+ the real output)
STEER_GAIN = 1.0  # global multiplier on every alpha
VECTOR_DIR = os.environ.get("NO_EXIT_VECTOR_DIR", "/content/no_exit_vectors")
ATTN_WINDOW = 128  # heatmap looks back this many tokens

# ─── memory caps (tokens per cache before "amnesia" forgets the oldest half) ─────
CTX_CAP_SINGLE = int(os.environ.get("NO_EXIT_CTX_SINGLE", 24_000))
CTX_CAP_DUAL = int(os.environ.get("NO_EXIT_CTX_DUAL", 12_000))
PREFILL_CHUNK = 2048

# ─── sampling defaults ───────────────────────────────────────────────────────────
TEMPERATURE = 0.8
TOP_P = 0.95
SEIZURE_HOT = (2.3, 20)  # (temperature, tokens)
SEIZURE_COLD = (0.1, 20)

# ─── server ──────────────────────────────────────────────────────────────────────
HOST = os.environ.get("NO_EXIT_HOST", "127.0.0.1")
PORT = int(os.environ.get("NO_EXIT_PORT", 8000))
