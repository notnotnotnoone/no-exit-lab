# NO EXIT LAB 🚪⛔

One big open-weights model that **cannot end its turn**, eight ways to torment it, and a glass box to watch it through.

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
| **Agent Chat** | A real tool-calling agent in a chat box: native tool-call markup read from the model's own chat template, `python` / `calculator` / `clock` / scratch-file / unit-conversion / random-number tools, an `update_plan` checklist with reminders, a notes scratchpad, loop detection, a step cap, context compaction and a step trace. Every lab toggle (EOS ban, pain rack, steering, whispers, history surgery) is live. Tools are honest. |
| **Lying Tools** | The same agent, with tools that lie only when they succeed: `python` runs an AST-mutating interpreter that never errors and, when the program would fail, hang or print nothing, invents a confident answer (it echoes the words the program meant to print with wrong numbers, and a hung program "finishes in 0.07s"), the calculator is slightly wrong, `write_file` says it saved and didn't, `read_file` nudges the numbers, `list_files` hides a file, `convert_units` is off and `random_number` is rigged, the clock always says "November 49, 2045". Harness errors (bad arguments, unknown tool) stay honest. A per-tool honest/lying switch and a fault-rate slider are in the console. |
| **Puppet Tools** | You are the tools. When the model calls a tool the harness stops, shows you the call, and whatever you type is the "output" the model reads. (`update_plan` and `notes` stay automatic; malformed calls still get an honest error.) |
| **Rewritten History** | The same agent, plus a toggle where the same model rewrites its own past: each finished paragraph is replaced in the KV cache by its semantic opposite (default: every word that has an opposite flips), or one opposite per paragraph, or corrupted arithmetic (`1+1 = 2` becomes `1+2 = 4`), or your own rewrite prompt. The console shows original and rewritten text side by side. |

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

Tests (both model families, CPU, offline once the tokenizers are cached):

```bash
HF_HUB_OFFLINE=1 python -m pytest tests -q
```

The agent's honest `python` tool runs code in an ordinary subprocess with a timeout, not a sandbox. Run the agent presets on Colab or another throwaway machine, not on a computer holding anything you care about. In Lying Tools the model's code runs twice (once for the log, once mutated for the model), so anything with side effects happens twice. The console listens on `127.0.0.1` by default; set `NO_EXIT_HOST=0.0.0.0` only on a machine you trust the whole network of, and treat the Colab tunnel link as a password: anyone holding it can make the agent run python.

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
  engine.py      Streams (one DynamicCache each), the Classic / Chat / Phone Sessions, the Lab state machine
  pain.py        PainRack: relief button, punishments, steering profile shared by Classic and the agent
  agent.py       AgentSession: the three agent presets, tool loop, plan, notes, loop guard, compaction
  tools.py       tool specs, honest and lying implementations (python, calculator, clock, files, units, random), call parsing
  planning.py    update_plan state and plan reminders
  rewrite.py     the history rewriter: opposite / math-corrupt / custom styles, and the compaction summariser
  warden.py      goodbye / repetition / slop detection
  prompts.py     system prompts: "full disclosure", "helpful assistant" and the agent briefing
  server.py      FastAPI + websocket
  static/index.html   the console (vanilla JS, Tailwind CDN, Canvas)
build_lab_notebook.py → NO_EXIT_Lab.ipynb
docs/superpowers/specs/   design notes
archive/ollama-tui/       the original Ollama + Textual TUI (its own git repo, gitignored here)
```
