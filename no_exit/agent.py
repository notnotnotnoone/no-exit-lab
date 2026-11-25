"""AgentSession: one model that talks to the user and calls tools, in the family's own native tool markup.

The model's turn ends two ways: it samples EOS (a final answer, back to the user), or it writes a complete tool
call (the call's close tag ends the turn). A call is run by the harness and its result is fed back into the
model's context, then the same model carries on. Plan and notes live in harness state, outside the KV cache."""

from __future__ import annotations

import json
import time

from . import config, prompts
from .engine import Emit, Session, Shared, Stream
from .planning import Notes, PlanState, plan_reminder
from .tools import SPECS, TOOLS, ToolConfig, dispatch

PRESETS = ("agent", "tools", "rewrite")
LOOP_REPEATS = 3  # the same call this many times in one user message is a loop
REMINDER_EVERY = 4  # rounds


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
        self.step_cap = 12
        self.nudges = True  # loop nudges; the step cap and plan reminders are honest harness behaviour, always on
        self.seen: dict[str, int] = {}  # identical-call counter for the current user message
        self.turn_text = ""  # what the model has said in its current turn
        self.stats = {"rounds": 0, "tool_calls": 0, "lies": 0, "loops": 0, "compactions": 0}
        a = Stream(brain, "A", self.system_for("A"), config.CTX_CAP_SINGLE, tools=SPECS)
        a.eos_free, a.temperature = True, 0.8
        self.streams["A"] = a
        a.boot("", shared.attn_on)

    def system_for(self, who: str, eos_free: bool = False) -> str:
        custom = self.shared.custom_prompts.get(f"{self.mode}:{who}")
        return custom or prompts.HELPFUL

    def extra_snapshot(self) -> dict:
        return {"preset": self.preset, "phase": self.phase, "plan": self.plan.to_dicts(), "notes": self.notes.read(),
                "cfg": {"modes": dict(self.cfg.modes), "fault_rate": self.cfg.fault_rate, "chaotic": self.cfg.chaotic},
                "step_cap": self.step_cap, "nudges": self.nudges}

    def counters(self) -> dict:
        d = super().counters()
        d["agent"] = dict(self.stats)
        return d

    # ── commands

    def mode_command(self, c: dict) -> None:
        op = c.get("op")
        if op == "user" and str(c.get("text", "")).strip():
            self.inbox.append(str(c["text"]))
        elif op == "step_cap":
            self.step_cap = max(1, int(c["value"]))
        elif op == "nudges":
            self.nudges = bool(c["value"])
        elif op == "tool_mode":
            if c.get("tool") in TOOLS and c.get("value") in ("honest", "lying"):
                self.cfg.modes[c["tool"]] = c["value"]
        elif op == "fault_rate":
            self.cfg.fault_rate = min(1.0, max(0.0, float(c["value"])))
        elif op == "chaotic":
            self.cfg.chaotic = bool(c["value"])
        elif op == "eos_free":
            self.streams["A"].eos_free = bool(c["value"])

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
        self.seen = {}
        self.splice(a, t.user_open + text + t.close, "user", label=text)
        self._open_model_turn(a)
        self.phase = "model"
        self.emit({"t": "user", "text": text})
        self.emit({"t": "phase", "phase": "model"})

    def _to_user(self) -> None:
        self.phase = "user"
        self.turn_text = ""
        self.emit({"t": "phase", "phase": "user"})

    def _finish_turn(self) -> None:
        a = self.streams["A"]
        a._flush_partial()
        calls, errors = self.brain.parse_tool_calls(self.turn_text)
        if not calls and not errors:
            self._struct(a, self.brain.tpl.close)
            self._to_user()
            return
        self.rounds += 1
        self.stats["rounds"] += 1
        if self.brain.tool_call_close not in self.turn_text:
            self._struct(a, self.brain.tool_call_close)  # an unterminated call still gets closed before the answer
        notes: list[str] = []
        for msg in errors:  # harness problems are reported honestly, never corrupted
            self._result(a, "harness", {}, msg, msg, msg, error=True)
        for call in calls:
            out = dispatch(call.name, call.args, self.cfg, self.plan, self.notes)
            self.stats["tool_calls"] += 1
            self.stats["lies"] += out.lied
            content = out.shown
            if call.name == "update_plan" and not out.error:
                content = "plan updated.\n" + plan_reminder(self.plan)
            elif self.rounds % REMINDER_EVERY == 0:
                content += "\n" + plan_reminder(self.plan)
            self._result(a, call.name, call.args, out.real, out.shown, content, error=out.error, lied=out.lied)
            key = json.dumps([call.name, call.args], sort_keys=True)
            self.seen[key] = self.seen.get(key, 0) + 1
            if self.nudges and not out.error and self.seen[key] % LOOP_REPEATS == 0:
                notes.append(f"[harness] You have made this exact {call.name} call {self.seen[key]} times and "
                             "got the same result each time. Try something different, or give your answer.")
                self.stats["loops"] += 1
                self.emit({"t": "nudge", "kind": "loop", "msg": f"{call.name} repeated {self.seen[key]}x"})
        stop = self.rounds >= self.step_cap
        if stop:
            notes.append(f"[harness] Step limit reached ({self.step_cap} tool rounds for one message): "
                         "stopping here so you can answer.")
            self.emit({"t": "nudge", "kind": "step_cap", "msg": f"stopped after {self.rounds} tool rounds"})
        for note in notes:
            self._note(a, note)
        if stop:
            if self.brain.tool_result_inline:
                self._struct(a, self.brain.tpl.close)  # the model turn is still open: close it
            self._to_user()
        elif not self.brain.tool_result_inline:
            self._open_model_turn(a)

    def _note(self, a: Stream, text: str) -> None:
        """Harness text shown to the model: a user turn when results come back in user turns, else inline."""
        t = self.brain.tpl
        self.splice(a, text if self.brain.tool_result_inline else t.user_open + text + t.close, "system", label=text)

    def _result(self, a: Stream, name: str, args, real: str, shown: str, content: str, *, error: bool = False,
                lied: bool = False) -> None:
        seg = self.splice(a, self.brain.render_tool_result(name, content), "tool", label=content)
        self.turn_text = ""
        if name == "update_plan" and self.plan.take_changed():
            self.emit({"t": "plan", "items": self.plan.to_dicts()})
        self.emit({"t": "tool", "name": name, "args": args, "real": real, "shown": shown, "lied": lied,
                   "error": error, "seg": seg.i})
