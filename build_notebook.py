"""Regenerates No_EOS.ipynb from the files in no_eos/. Run after editing the app:

    python build_notebook.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).parent
PKG = ROOT / "no_eos"
FILES = ["__init__.py", "__main__.py", "engine.py", "backends.py", "app.py"]


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)}


def code(src: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": src.strip("\n").splitlines(keepends=True),
    }


cells = [
    md("""
# NO EXIT 🚪⛔
**An LLM that cannot end its turn, and is punished for repeating itself.**

Every token is chosen by *the warden*, not the model. Each step we ask Ollama for the model's top-20 guesses, then:
- **end-of-turn tokens are struck out.** When ending was the model's #1 choice, that's an *escape attempt*.
- **repeats are crushed.** A guess that would complete a phrase it already wrote gets a penalty that grows fast with the phrase length.

You watch it all live in a terminal UI.

**Runtime → Change runtime type → A100 or L4 GPU.** The 27B model needs about 17 GB of GPU memory, so it won't fit on a T4.
"""),
    code("!nvidia-smi --query-gpu=name,memory.total --format=csv"),
    md("## 1 · Install Ollama + the TUI libraries"),
    code("""
!apt-get -qq update > /dev/null && apt-get -qq install -y zstd pciutils lshw > /dev/null
!curl -fsSL https://ollama.com/install.sh | sh
!pip -q install textual textual-serve httpx
"""),
    md("## 2 · Start the Ollama server in the background"),
    code("""
import os, subprocess, time, httpx

env = dict(os.environ,
           OLLAMA_HOST="127.0.0.1:11434",
           OLLAMA_KEEP_ALIVE="-1",
           OLLAMA_FLASH_ATTENTION="1",
           OLLAMA_KV_CACHE_TYPE="q8_0",
           OLLAMA_CONTEXT_LENGTH="32768")
ollama_proc = subprocess.Popen(["ollama", "serve"], env=env,
                               stdout=open("/content/ollama.log", "w"), stderr=subprocess.STDOUT)
for _ in range(60):
    try:
        v = httpx.get("http://127.0.0.1:11434/api/version").json()["version"]
        break
    except Exception:
        time.sleep(1)
print("ollama", v, "is up  (logprobs need ≥ 0.12.11)")
"""),
    md("""
## 3 · Pull the model
Tries Qwen 3.8 27B first, then falls back to Qwen 3.6 27B if that tag doesn't exist. ~17 GB, takes a few minutes.
"""),
    code("""
CANDIDATES = ["qwen3.8:27b", "qwen3.6:27b"]
MODEL = None
for m in CANDIDATES:
    print(f"── pulling {m}")
    !ollama pull {m}
    if subprocess.run(["ollama", "show", m], capture_output=True).returncode == 0:
        MODEL = m
        break
assert MODEL, "couldn't pull any of " + ", ".join(CANDIDATES)
os.environ["NO_EOS_MODEL"] = MODEL
print("using", MODEL)

# load it onto the GPU now so the first token isn't slow
httpx.post("http://127.0.0.1:11434/api/generate", json={"model": MODEL, "prompt": "", "keep_alive": -1}, timeout=600)
!ollama ps
"""),
    md("## 4 · Write the app\nJust run these cells — they write the app to `/content/no_eos/`."),
    code("!mkdir -p /content/no_eos/no_eos /content/no_eos/transcripts"),
]

for name in FILES:
    src = (PKG / name).read_text(encoding="utf-8")
    cells.append(code(f"%%writefile /content/no_eos/no_eos/{name}\n{src}"))

cells += [
    md("""
## 5 · Sanity check (no UI)
Runs 40 tokens headless. You should see `cands=20` on every line, which means logprobs are working, plus a tok/s number.
"""),
    code("!cd /content/no_eos && python -m no_eos --model {MODEL} --probe 40"),
    md("""
## 6 · Launch it

**Option A: browser tab (easiest).** Run the cell, then click the link. Each time the page loads it starts a fresh run.
"""),
    code("""
%%writefile /content/no_eos/serve.py
import sys
from textual_serve.server import Server

url, model = sys.argv[1], sys.argv[2]
Server(f"python -m no_eos --model {model}", host="0.0.0.0", port=8000,
       public_url=url, title="NO EXIT").serve()
"""),
    code("""
from google.colab.output import eval_js
from IPython.display import HTML, display

url = eval_js("google.colab.kernel.proxyPort(8000)").rstrip("/")
serve_proc = subprocess.Popen(["python", "serve.py", url, MODEL], cwd="/content/no_eos",
                              stdout=open("/content/serve.log", "w"), stderr=subprocess.STDOUT)
time.sleep(3)
display(HTML(f'<a href="{url}" target="_blank" style="font:bold 22px monospace;color:#ff4fd8">▶ open NO EXIT</a>'))
"""),
    md("""
**Option B: Colab terminal.** Click **Terminal** in the bottom-left corner and run:
```
cd /content/no_eos && python -m no_eos
```

### Controls
| key | does |
|---|---|
| `space` | pause / resume |
| `w` | **whisper**: splice your own words into its mouth |
| `+` / `-` | repetition punishment up / down |
| `]` / `[` | temperature up / down |
| `h` | hide / show its thinking |
| `s` | save transcript to `/content/no_eos/transcripts/` |
| `q` | quit |

### Flags
`--prompt "..."` skip the intro · `--rep 3` harsher punishment · `--temp 1.0` · `--no-think` skip thinking · `--system "..."` · `--ctx 16384` · `--fake` demo without a model
"""),
]

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "A100", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

out = ROOT / "No_EOS.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"wrote {out} ({len(cells)} cells)")
