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
from .rewrite import STYLES, Rewriter
from .tools import SPECS, TOOLS, ToolConfig, dispatch

PRESETS = ("agent", "tools", "rewrite")
LOOP_REPEATS = 3  # the same call this many times in one user message is a loop
REMINDER_EVERY = 4  # rounds
COMPACT_AT = 0.75  # fraction of the context cap at which the conversation is summarised


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
        # the self-rewriter: off unless switched on; the Rewritten History preset switches it on
        self.rewriter = Rewriter(brain)
        self.rewrite_on = False
        self.rewrite_style = "opposite_all"
        self.rewrite_prompt = ""
        self.para_text, self.para_first, self.para_last = "", None, None  # the paragraph being written
        self.compaction = True  # summarise the conversation when the context fills up
        self.amnesia_on = False  # the lab's old memory loss: forget the oldest half at the hard cap
        self.last_user = ""  # the newest user message, which survives a compaction verbatim
        self.compact_blocked = False  # a compaction that could not shrink enough is not repeated this message
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
                "step_cap": self.step_cap, "nudges": self.nudges,
                "rewrite": {"on": self.rewrite_on, "style": self.rewrite_style, "prompt": self.rewrite_prompt},
                "compaction": self.compaction, "amnesia": self.amnesia_on}

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
        elif op == "rewrite_on":
            self.rewrite_on = bool(c["value"])
        elif op == "rewrite_style":
            if c.get("value") in STYLES:
                self.rewrite_style = c["value"]
        elif op == "rewrite_prompt":
            self.rewrite_prompt = str(c["value"])
        elif op == "compaction":
            self.compaction = bool(c["value"])
        elif op == "amnesia":
            self.amnesia_on = bool(c["value"])
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
        self._track(seg)
        if self.para_text.endswith("\n\n"):
            self._rewrite_paragraph()
        if self.brain.tool_call_close and self.brain.tool_call_close in self.turn_text:
            self._finish_turn()

    # ── memory: compaction (summarise) and amnesia (forget), both only at the model's say-so boundaries

    def _needs_compaction(self, a: Stream) -> bool:
        return (self.compaction and not self.compact_blocked and not a._partial
                and a.cache.get_seq_length() >= COMPACT_AT * a.ctx_cap)

    def check_amnesia(self, st: Stream) -> None:
        # compaction only between turns (nothing half-said); the hard-cap amnesia is opt-in
        if self.last_user and not self.turn_text and self._needs_compaction(st):
            self._compact(st, self.last_user)
        elif self.amnesia_on:
            super().check_amnesia(st)

    def _transcript(self, a: Stream) -> str:
        lines: list[str] = []
        said = ""
        for g in a.segs:
            if g.kind in ("tok", "rewritten", "gaslit"):
                said += g.text
                continue
            if said.strip():
                lines.append(f"Assistant: {said.strip()}")
            said = ""
            if g.kind == "user":
                lines.append(f"User: {g.text}")
            elif g.kind == "tool":
                lines.append(f"Tool result: {g.text}")
            elif g.kind in ("system", "whisper") and g.text.strip():
                lines.append(g.text.strip())
        if said.strip():
            lines.append(f"Assistant: {said.strip()}")
        return "\n".join(lines)

    def _compact(self, a: Stream, latest_user: str) -> None:
        """Replace the whole history with: a summary (+ the harness's plan and notes) and the latest user message."""
        before = a.cache.get_seq_length()
        block = "[summary of earlier conversation]\n" + self.rewriter.summarize(self._transcript(a))
        if self.plan.items:
            block += f"\n\n[your plan]\n{self.plan.render()}"
        if self.notes.read():
            block += f"\n\n[your notes]\n{self.notes.read()}"
        t = self.brain.tpl
        a.reset_body([(t.user_open + block + t.close, "system"), (t.user_open + latest_user + t.close, "user"),
                      (t.model_open + self.think_empty, "struct")], self.shared.attn_on)
        a.segs[0].text, a.segs[1].text, a.segs[2].text = block, latest_user, ""  # show text, not chat markup
        self.turn_text = ""
        self.para_text, self.para_first, self.para_last = "", None, None
        after = a.cache.get_seq_length()
        self.compact_blocked = after >= COMPACT_AT * a.ctx_cap
        self.stats["compactions"] += 1
        self.emit({"t": "compact", "before": before, "after": after})
        self.resync(a)

    def _track(self, seg) -> None:
        if self.para_first is None:
            self.para_first = seg.i
        self.para_last = seg.i
        self.para_text += seg.text

    def _rewrite_paragraph(self) -> None:
        """The model finished a paragraph: if rewriting is on, the same model rewrites it in place, in its own
        history and KV cache. Paragraphs holding a tool call, and anything that is not the model's own speech, stay."""
        text, first, last = self.para_text, self.para_first, self.para_last
        self.para_text, self.para_first, self.para_last = "", None, None
        o = self.brain.tool_call_open
        body = text.rstrip("\n")
        if not self.rewrite_on or first is None or not body.strip() or (o and o in text):
            return
        a = self.streams["A"]
        try:
            new = self.rewriter.rewrite(body, self.rewrite_style, self.rewrite_prompt)
        except ValueError as e:  # e.g. the custom style with no prompt
            self.emit({"t": "event", "kind": "rewrite_error", "msg": str(e)})
            return
        if new.strip() == body.strip():
            return  # nothing came back worth keeping
        new += text[len(body):]  # keep the blank line that ended the paragraph
        original = a.replace_span(first, last, new, "rewritten", self.shared.attn_on)
        if original is None:
            return
        self.emit({"t": "rewrite", "seg": first, "original": original, "new": new, "style": self.rewrite_style})
        self.resync(a)

    def _struct(self, a: Stream, text: str) -> None:
        self.splice(a, text, "struct", label="")

    def _open_model_turn(self, a: Stream) -> None:
        self._struct(a, self.brain.tpl.model_open + self.think_empty)
        self.turn_text = ""

    def _deliver(self, text: str) -> None:
        a, t = self.streams["A"], self.brain.tpl
        self.rounds = 0
        self.seen = {}
        self.last_user = text
        self.compact_blocked = False
        if self._needs_compaction(a):
            self._compact(a, text)  # this feeds the new user message itself
        else:
            self.splice(a, t.user_open + text + t.close, "user", label=text)
            self._open_model_turn(a)
        self.phase = "model"
        self.emit({"t": "user", "text": text})
        self.emit({"t": "phase", "phase": "model"})

    def _to_user(self) -> None:
        self.phase = "user"
        self.para_text, self.para_first, self.para_last = "", None, None
        self.turn_text = ""
        self.emit({"t": "phase", "phase": "user"})

    def _finish_turn(self) -> None:
        a = self.streams["A"]
        a._flush_partial()
        if a.segs and a.segs[-1].kind == "tok" and self.para_last is not None and a.segs[-1].i > self.para_last:
            self._track(a.segs[-1])  # a half-decoded character flushed at the end belongs to the paragraph
        self._rewrite_paragraph()  # the last paragraph (a no-op if it holds a tool call)
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
