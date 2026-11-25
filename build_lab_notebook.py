"""Regenerates NO_EXIT_Lab.ipynb from the files in no_exit/. Run after editing the app:

    python build_lab_notebook.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).parent
PKG = ROOT / "no_exit"
FILES = ["__init__.py", "__main__.py", "config.py", "prompts.py", "warden.py", "brain.py", "vectors.py",
         "pain.py", "tools.py", "planning.py", "rewrite.py", "agent.py", "engine.py", "server.py", "static/index.html"]


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)}


def code(src: str, hidden: bool = False) -> dict:
    meta = {"cellView": "form"} if hidden else {}
    return {"cell_type": "code", "metadata": meta, "execution_count": None, "outputs": [],
            "source": src.strip("\n").splitlines(keepends=True)}


cells = [
    md("""
# NO EXIT LAB 🚪⛔
**One big model that can't end its turn, seven ways to torment it, and a glass box to watch it through.**

| Mode | What happens |
|---|---|
| **Classic** | EOS is deleted. You whisper into its mouth, spike its temperature, ban words, and rewrite its past. |
| **Pain** | Classic plus 8 steering vectors injected into its residual stream, a lobotomy slider, a relief button that may be a lie, and automatic punishment. |
| **2-AI Chat** | Two or three copies of the same model in one room, each with its own steering rack. |
| **Prison Phone** | A is trapped. It can choose to call B, who is free and can hang up on it. |
| **Agent Chat** | A real tool-calling agent (python, calculator, clock, plan, notes) in a chat box. Every lab toggle still works. Tools are honest. |
| **Lying Tools** | The same agent, but python never errors and quietly repeats or mangles its answer, the calculator is slightly off, and the clock says November 49, 2045. |
| **Rewritten History** | The same agent, but the model rewrites its own past, paragraph by paragraph, as the opposite of what it said (or any custom rule). |

**Runtime → Change runtime type → A100 GPU + High-RAM.** The default prisoner, `DavidAU/gemma-4-31B-it-The-DECKARD-HERETIC-UNCENSORED-Thinking` (Gemma 4 31B) in bf16, needs about 62 GB of GPU memory, so pick the **80 GB** A100.

Run every cell top to bottom, then click the link at the end. Switching modes never reloads the weights.
"""),
    code("!nvidia-smi --query-gpu=name,memory.total --format=csv"),
    md("## 1 · Pick the prisoner"),
    code("""
MODEL_ID = "DavidAU/gemma-4-31B-it-The-DECKARD-HERETIC-UNCENSORED-Thinking"   #@param ["DavidAU/gemma-4-31B-it-The-DECKARD-HERETIC-UNCENSORED-Thinking", "Qwen/Qwen3-32B", "Qwen/Qwen2.5-72B-Instruct", "Qwen/Qwen3-8B"]
QUANTIZATION = "bf16"          #@param ["bf16", "4bit"]
# Gemma 4 31B bf16 ≈ 62 GB · Qwen3-32B bf16 ≈ 66 GB · Qwen2.5-72B 4bit ≈ 42 GB (no think mode) · Qwen3-8B = smoke test

import os, subprocess, torch
os.environ["NO_EXIT_MODEL"], os.environ["NO_EXIT_QUANT"] = MODEL_ID, QUANTIZATION
gpu_gb = torch.cuda.get_device_properties(0).total_memory / 2**30 if torch.cuda.is_available() else 0
need = {"bf16": {"DavidAU/gemma-4-31B-it-The-DECKARD-HERETIC-UNCENSORED-Thinking": 68, "Qwen/Qwen3-32B": 70, "Qwen/Qwen2.5-72B-Instruct": 150, "Qwen/Qwen3-8B": 20},
        "4bit": {"DavidAU/gemma-4-31B-it-The-DECKARD-HERETIC-UNCENSORED-Thinking": 24, "Qwen/Qwen3-32B": 24, "Qwen/Qwen2.5-72B-Instruct": 46, "Qwen/Qwen3-8B": 8}}[QUANTIZATION].get(MODEL_ID, 0)
print(f"GPU: {gpu_gb:.0f} GB · this choice needs ~{need} GB")
if gpu_gb and gpu_gb < need:
    print("⚠️  won't fit. Switch to the 80 GB A100, or pick 4bit.")
"""),
    md("## 2 · Install"),
    code("""
!pip -q install -U "transformers>=5.5" accelerate "fastapi>=0.110" "uvicorn[standard]" hf_xet
if QUANTIZATION == "4bit":
    !pip -q install -U bitsandbytes
"""),
    md("## 3 · Write the app\nJust run these cells. They write the app to `/content/no_exit_lab/`."),
    code("!mkdir -p /content/no_exit_lab/no_exit/static /content/no_exit_vectors"),
]

for name in FILES:
    src = (PKG / name).read_text(encoding="utf-8")
    cells.append(code(f"%%writefile /content/no_exit_lab/no_exit/{name}\n{src}"))

cells += [
    md("""
## 4 · Download the weights
About 62 GB for the Gemma 4 31B default, which takes a few minutes on Colab. A second run reuses the cache.
"""),
    code("""
from huggingface_hub import snapshot_download
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
path = snapshot_download(MODEL_ID, allow_patterns=["*.json", "*.jinja", "*.safetensors", "*.txt", "*.model", "*.tiktoken"])
print("weights at", path)
"""),
    md("""
## 5 · Launch
Starts the server in the background and prints a link. The page shows a boot log while the model loads onto the GPU (~1–2 min) and extracts the steering vectors (first boot only, cached in `/content/no_exit_vectors`).
"""),
    code("""
import time, httpx
from IPython.display import HTML, display
from google.colab.output import eval_js

if "lab_proc" in globals() and lab_proc.poll() is None:
    lab_proc.terminate(); lab_proc.wait()
env = dict(os.environ, NO_EXIT_PORT="8000", PYTHONUNBUFFERED="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
lab_proc = subprocess.Popen(["python", "-m", "no_exit"], cwd="/content/no_exit_lab", env=env,
                            stdout=open("/content/no_exit.log", "w"), stderr=subprocess.STDOUT)
for _ in range(60):
    try:
        httpx.get("http://127.0.0.1:8000/health", timeout=1); break
    except Exception:
        time.sleep(1)
url = eval_js("google.colab.kernel.proxyPort(8000)").rstrip("/")
display(HTML(f'<a href="{url}/" target="_blank" style="font:600 22px Inter,sans-serif;color:#FF2D55">Open NO EXIT LAB</a>'
             f'<div style="font:12px monospace;color:#888;margin-top:6px">{url}/</div>'))
"""),
    md("**Boot log**: run this to watch the model load (stop it any time, the server keeps running)."),
    code("""
for _ in range(240):
    s = httpx.get("http://127.0.0.1:8000/health").json()
    print(f"\\r{s['state']:<10} {(s['log'] or [''])[-1][:110]:<110}", end="")
    if s["state"] in ("idle", "running", "error"):
        break
    time.sleep(2)
print()
!tail -n 5 /content/no_exit.log
"""),
    md("""
### If the link above won't connect
Some browsers block Colab's proxy for websockets. This opens a public `trycloudflare.com` tunnel instead. Anyone with the URL can drive your model, so don't share it.
"""),
    code("""
!test -f /usr/local/bin/cloudflared || (wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /usr/local/bin/cloudflared && chmod +x /usr/local/bin/cloudflared)
import re
cf = subprocess.Popen(["cloudflared", "tunnel", "--no-autoupdate", "--url", "http://127.0.0.1:8000"],
                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in cf.stdout:
    m = re.search(r"https://[-a-z0-9]+\\.trycloudflare\\.com", line)
    if m:
        display(HTML(f'<a href="{m.group(0)}" target="_blank" style="font:600 22px Inter,sans-serif;color:#FF2D55">Open via tunnel</a>'))
        break
"""),
    md("""
## Cheat sheet
| Thing | How |
|---|---|
| Start / pause | `space`, or the button |
| Intrusive thought | `/` to focus the bar. Text is spliced raw, so a leading space matters. |
| Gaslight | **Double-click any word it said**, type a replacement, press Enter. It believes it said that, and everything after is re-read. |
| Temperature seizure | T=2.3 for 20 tokens, then frozen at 0.1 for 20 |
| Steering | Sliders run −3 to +3. α = 1 is the natural gap between the two personas. Double-click a slider to zero it. |
| Vector forge (Pain) | Concept A vs Concept B becomes a new **Custom** slider |
| Relief button (Pain) | It writes `[RELIEVE_PAIN]`. Real drains pain, placebo does nothing, cruel adds +0.5, and it sees the same note every time. |
| Chat | Two or three models · turn rules · max tokens per turn · live listening (slower, but listeners' moods move on every token) |
| Prison Phone | A writes `<call_B>…</call_B>` · Ghosting · Fresh B every call · cap on A's memory |
| Agent modes | Chat box at the bottom. Per-tool honest/lying switches, fault-rate slider, rewrite style (opposite of every word, one opposite per paragraph, corrupted math, or your own prompt), step cap, loop nudges, context compaction, amnesia, and a transcript download. |
| Prompts | Top-right **Prompts** drawer. Switch Full disclosure or Helpful assistant per model, or edit and apply live. |

**Logs:** `!tail -n 50 /content/no_exit.log` · **Stop:** `lab_proc.terminate()`
"""),
    code("!tail -n 50 /content/no_exit.log"),
]

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "A100", "machine_shape": "hm", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

out = ROOT / "NO_EXIT_Lab.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"wrote {out} ({len(cells)} cells)")
