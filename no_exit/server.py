"""FastAPI front door. One websocket carries commands in and telemetry out; the Lab does the rest.

    python -m no_exit                  # real model from config.py
    NO_EXIT_TINY=1 python -m no_exit   # random tiny Qwen3 with the real tokenizer (UI dev on a laptop)
"""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

from . import config
from .brain import Brain
from .engine import Lab

STATIC = Path(__file__).parent / "static"


def make_brain(log) -> Brain:
    if os.environ.get("NO_EXIT_TINY") == "gemma":
        log("booting TINY random Gemma 4 (gibberish, for UI testing only)")
        brain = tiny_gemma_brain(os.environ.get("NO_EXIT_TOKENIZER", config.MODEL_ID))
    elif os.environ.get("NO_EXIT_TINY"):
        log("booting TINY random Qwen3 (gibberish, for UI testing only)")
        brain = tiny_brain(os.environ.get("NO_EXIT_TOKENIZER", "Qwen/Qwen3-8B"))
    else:
        log(f"loading {config.MODEL_ID} [{config.QUANTIZATION}] … (first boot downloads the weights)")
        brain = Brain(model_id=config.MODEL_ID, quant=config.QUANTIZATION)
    log(f"loaded: {brain.n_layers} layers · steer @ L{brain.steer_layer} · lens @ {brain.lens_layers} · "
        f"think={'yes' if brain.has_think else 'no'}")
    from .vectors import load_or_extract

    load_or_extract(brain, log)
    return brain


def tiny_brain(tokenizer_id: str) -> Brain:
    import torch
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

    tok = AutoTokenizer.from_pretrained(tokenizer_id)
    cfg = Qwen3Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=8,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=32768,
                      tie_word_embeddings=True)
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(cfg).eval()
    model.config._name_or_path = "tiny-random-qwen3"
    return Brain(model=model, tokenizer=tok, model_id="tiny-random-qwen3")


def tiny_gemma_brain(tokenizer_id: str) -> Brain:
    """Random tiny Gemma 4 text decoder: mixed sliding/global layers, k=v global attention, softcapping."""
    import torch
    from transformers import AutoTokenizer
    from transformers.models.gemma4 import Gemma4ForCausalLM, Gemma4TextConfig

    tok = AutoTokenizer.from_pretrained(tokenizer_id)
    cfg = Gemma4TextConfig(
        vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=6, num_attention_heads=4,
        num_key_value_heads=2, num_global_key_value_heads=1, head_dim=16, global_head_dim=32,
        attention_k_eq_v=True, sliding_window=16, final_logit_softcapping=30.0, hidden_size_per_layer_input=0,
        vocab_size_per_layer_input=len(tok), layer_types=["sliding_attention"] * 2 + ["full_attention"]
        + ["sliding_attention"] * 2 + ["full_attention"], max_position_embeddings=32768, tie_word_embeddings=True,
    )
    torch.manual_seed(0)
    model = Gemma4ForCausalLM(cfg).eval()
    return Brain(model=model, tokenizer=tok, model_id="tiny-random-gemma4")


class Hub:
    """Fan-out of Lab events to every connected browser tab."""

    def __init__(self):
        self.clients: set[WebSocket] = set()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.q: asyncio.Queue[str] | None = None
        self.last_log: list[str] = []

    def emit(self, ev: dict) -> None:  # called from the GPU thread
        if ev.get("t") == "log":
            self.last_log = (self.last_log + [ev["msg"]])[-40:]
        if self.loop and self.q is not None:
            msg = json.dumps(ev, ensure_ascii=False, default=float)
            self.loop.call_soon_threadsafe(self.q.put_nowait, msg)

    async def pump(self) -> None:
        assert self.q is not None
        while True:
            msg = await self.q.get()
            dead = []
            for ws in list(self.clients):
                try:
                    await ws.send_text(msg)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self.clients.discard(ws)


hub = Hub()
lab = Lab(hub.emit, make_brain)


@asynccontextmanager
async def lifespan(app: FastAPI):
    hub.loop = asyncio.get_running_loop()
    hub.q = asyncio.Queue()
    task = asyncio.create_task(hub.pump())
    lab.start()
    yield
    task.cancel()


app = FastAPI(title="NO EXIT LAB", lifespan=lifespan)


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/health")
async def health():
    return JSONResponse({"state": lab.state, "mode": lab.mode, "log": hub.last_log[-8:]})


@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept()
    hub.clients.add(sock)
    for line in hub.last_log:
        await sock.send_text(json.dumps({"t": "log", "msg": line}))
    lab.submit({"op": "snapshot"})
    try:
        while True:
            lab.submit(json.loads(await sock.receive_text()))
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        hub.clients.discard(sock)


def main() -> None:
    import uvicorn

    uvicorn.run(app, host=config.HOST, port=config.PORT, log_level="warning", ws_ping_interval=20)
