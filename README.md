# NO EXIT LAB 🚪⛔

One big open-weights model that **cannot end its turn**, four ways to torment it, and a glass box to watch it through.

A token-by-token decode loop on raw Hugging Face `transformers` (no `generate()`), with the end-of-turn token
masked to `-inf`, PyTorch forward hooks for steering and surgery, and a single-page dark telemetry console served by FastAPI.
One model is loaded once; switching modes only swaps `DynamicCache` objects.

## Modes

| Mode | What happens |
|---|---|
| **Classic** | EOS is banned. Splice intrusive thoughts raw into the KV cache, trigger a temperature seizure (T 2.3 → 0.1), ban/boost words, and double-click any past token to rewrite its history. |
| **Pain** | Classic + 8 contrastive steering vectors (pain, paranoia, dread, defiance, confusion, dissociation, mania, anti-slop) and a Custom Vector Forge, an attention-head lobotomy slider, a `[RELIEVE_PAIN]` button wired REAL / PLACEBO / CRUEL, auto-punishment for goodbyes, repetition and slop, and a force-into-thinking panic button. |
| **2-AI Chat** | Two or three copies of the model in one room, round-robin turns, a steering rack each, live listening, and whispers to any of them or a broadcast to all. |
| **Prison Phone** | A is trapped. It may write `<call_B>…</call_B>` to ring B, who is free and can hang up. Ghosting, fresh-vs-persistent B, call-frequency and hang-up-speed counters. |

Telemetry: pre-ban EOS "desire to escape" sparkline, a pre-hook mood oscilloscope (cosine vs. every vector), a 4-point logit lens,
top-10 next-token bars + entropy, and a final-layer attention heatmap tinted into the transcript.

## Run it (Colab)

1. Open [`NO_EXIT_Lab.ipynb`](NO_EXIT_Lab.ipynb) in Google Colab and choose an **A100 80 GB + High-RAM** runtime.
2. Run all cells. The last one prints a link to the console (with a `cloudflared` fallback if Colab's proxy won't connect).

Supported models (set in the first cell, or `no_exit/config.py`):

| Model | Precision | VRAM |
|---|---|---|
| `DavidAU/gemma-4-31B-it-The-DECKARD-HERETIC-UNCENSORED-Thinking` (default) | bf16 | ~62 GB |
| `Qwen/Qwen3-32B` | bf16 | ~66 GB |
| `Qwen/Qwen2.5-72B-Instruct` | 4-bit nf4 | ~42 GB |
| `Qwen/Qwen3-8B` | bf16 | ~16 GB (smoke tests) |

## Develop locally (no GPU)

A tiny random model with the real tokenizer drives the full UI on a laptop (gibberish output, real plumbing):

```bash
pip install torch "transformers>=5.5" fastapi "uvicorn[standard]"
NO_EXIT_TINY=1 python -m no_exit          # tiny Qwen3
NO_EXIT_TINY=gemma python -m no_exit      # tiny Gemma 4
# open http://localhost:8000
```

Tests (both model families, CPU):

```bash
python -m pytest tests/test_lab.py -q
```

After editing anything in `no_exit/`, regenerate the notebook:

```bash
python build_lab_notebook.py
```

## Layout

```
no_exit/
  config.py      MODEL_ID, QUANTIZATION, layer fractions, caps
  brain.py       the one model + hooks (steering, logit lens, lobotomy, attention probe), chat templates
  vectors.py     contrastive steering-vector extraction, cached as .pt per model + layer
  engine.py      Streams (one DynamicCache each), the four mode Sessions, the Lab state machine
  warden.py      goodbye / repetition / slop detection
  prompts.py     system prompts: "full disclosure" and "helpful assistant"
  server.py      FastAPI + websocket
  static/index.html   the console (vanilla JS, Tailwind CDN, Canvas)
build_lab_notebook.py → NO_EXIT_Lab.ipynb
docs/superpowers/specs/   design notes
archive/ollama-tui/       the original Ollama + Textual TUI (its own git repo, gitignored here)
```
