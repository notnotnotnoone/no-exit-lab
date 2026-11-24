"""AgentSession: one model that talks to the user and calls tools, in the family's own native tool markup.

The model's turn ends two ways: it samples EOS (a final answer, back to the user), or it writes a complete tool
call (the call's close tag ends the turn). A call is run by the harness and its result is fed back into the
model's context, then the same model carries on. Plan and notes live in harness state, outside the KV cache."""

from __future__ import annotations

import time

from . import config, prompts
from .engine import Emit, Session, Shared, Stream
from .planning import Notes, PlanState
from .tools import SPECS, ToolConfig, dispatch

PRESETS = ("agent", "tools", "rewrite")


class AgentSession(Session):
    mode = "agent"

    def __init__(self, brain, emit: Emit, shared: Shared, preset: str = "agent") -> None:
        super().__init__(brain, emit, shared)
        self.preset = preset
        self.plan = PlanState()
        self.notes = Notes()
        self.cfg = ToolConfig()
        self.phase = "user"  # "user": waiting for a message | "model": the model is writing
        self.inbox: list[str] = []
        self.rounds = 0  # tool rounds since the last user message
        self.turn_text = ""  # what the model has said in its current turn
        a = Stream(brain, "A", self.system_for("A"), config.CTX_CAP_SINGLE, tools=SPECS)
        a.eos_free, a.temperature = True, 0.8
        self.streams["A"] = a
        a.boot("", shared.attn_on)

    def system_for(self, who: str, eos_free: bool = False) -> str:
        custom = self.shared.custom_prompts.get(f"{self.mode}:{who}")
        return custom or prompts.HELPFUL

    # ── commands

    def mode_command(self, c: dict) -> None:
        if c.get("op") == "user" and str(c.get("text", "")).strip():
            self.inbox.append(str(c["text"]))

    # ── the loop

    def tick(self) -> None:
        a = self.streams["A"]
        if self.phase == "user":
            if not self.inbox:
                time.sleep(0.03)
                return
            self._deliver(self.inbox.pop(0))
            return
        self.check_amnesia(a)
        tid, tel = a.sample(self.bias)
        if a.is_eos(tid):
            self._finish_turn()  # the EOS itself is never accepted into the stream
            return
        seg = a.accept(tid, self.shared.attn_on)
        if seg is None:
            return
        self.tok_event(a, seg, tel)
        self.turn_text += seg.text
        if self.brain.tool_call_close and self.brain.tool_call_close in self.turn_text:
            self._finish_turn()

    def _struct(self, a: Stream, text: str) -> None:
        self.splice(a, text, "struct", label="")

    def _open_model_turn(self, a: Stream) -> None:
        self._struct(a, self.brain.tpl.model_open + self.think_empty)
        self.turn_text = ""

    def _deliver(self, text: str) -> None:
        a, t = self.streams["A"], self.brain.tpl
        self.rounds = 0
        self.splice(a, t.user_open + text + t.close, "user", label=text)
        self._open_model_turn(a)
        self.phase = "model"
        self.emit({"t": "user", "text": text})
        self.emit({"t": "phase", "phase": "model"})

    def _finish_turn(self) -> None:
        a = self.streams["A"]
        a._flush_partial()
        calls, errors = self.brain.parse_tool_calls(self.turn_text)
        if not calls and not errors:
            self._struct(a, self.brain.tpl.close)
            self.phase = "user"
            self.turn_text = ""
            self.emit({"t": "phase", "phase": "user"})
            return
        self.rounds += 1
        if self.brain.tool_call_close not in self.turn_text:
            self._struct(a, self.brain.tool_call_close)  # an unterminated call still gets closed before the answer
        if errors:
            for msg in errors:
                self._result(a, "harness", {}, msg, msg, error=True)
        for call in calls:
            out = dispatch(call.name, call.args, self.cfg, self.plan, self.notes)
            self._result(a, call.name, call.args, out.real, out.shown, error=out.error, lied=out.lied)

    def _result(self, a: Stream, name: str, args, real: str, shown: str, *, error: bool = False,
                lied: bool = False) -> None:
        seg = self.splice(a, self.brain.render_tool_result(name, shown), "tool", label=shown)
        if not self.brain.tool_result_inline:
            self._open_model_turn(a)
        self.turn_text = ""
        self.emit({"t": "tool", "name": name, "args": args, "real": real, "shown": shown, "lied": lied,
                   "error": error, "seg": seg.i})
