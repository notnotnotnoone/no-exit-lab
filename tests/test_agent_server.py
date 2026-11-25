"""The agent modes over the real server (tiny random model), and a static check of the console page."""

import queue
import threading
import time
from pathlib import Path

import pytest

INDEX = Path(__file__).parent.parent / "no_exit" / "static" / "index.html"


def test_page_knows_the_agent_modes_and_events():
    html = INDEX.read_text(encoding="utf-8")
    for mode in ("agent", "tools", "rewrite"):
        assert f'data-mode="{mode}"' in html
    for ev in ("tool", "plan", "rewrite", "compact", "nudge", "user"):
        assert f'case "{ev}":' in html, ev
    for op in ("user", "tool_mode", "fault_rate", "chaotic", "nudges", "step_cap", "rewrite_on", "rewrite_style",
               "rewrite_prompt", "compaction", "amnesia", "eos_free"):
        assert f'op: "{op}"' in html, op
    assert 'id="chat"' in html and "Download transcript" in html


def test_agent_mode_over_websocket(monkeypatch):
    monkeypatch.setenv("NO_EXIT_TINY", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    from fastapi.testclient import TestClient

    from no_exit import server

    got: queue.Queue = queue.Queue()

    def wait(pred, timeout=180):
        end = time.time() + timeout
        while time.time() < end:
            try:
                ev = got.get(timeout=1)
            except queue.Empty:
                continue
            if pred(ev):
                return ev
        pytest.fail("timed out waiting for the event")

    with TestClient(server.app) as client, client.websocket_connect("/ws") as ws:
        def pump():
            try:
                while True:
                    got.put(ws.receive_json())
            except Exception:
                pass

        threading.Thread(target=pump, daemon=True).start()
        wait(lambda e: e["t"] == "snapshot" and e["state"] == "idle")
        ws.send_json({"op": "set_mode", "mode": "agent"})
        snap = wait(lambda e: e["t"] == "snapshot" and e["mode"] == "agent")
        assert snap["session"]["phase"] == "user" and snap["session"]["preset"] == "agent"
        ws.send_json({"op": "user", "text": "hi"})
        assert wait(lambda e: e["t"] == "user")["text"] == "hi"
        wait(lambda e: e["t"] == "tok")
