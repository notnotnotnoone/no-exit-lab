"""Pain Mode's machinery (relief button, punishments, transient pain spikes), shared by every session that
wants it: Pain mode and the agent modes.

A session owns one PainRack and calls it around each sampled token:
    rack.effective_profile(stream) → sample/accept → rack.restore_profile(stream) → rack.rules(stream, seg)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import config
from .warden import Warden

if TYPE_CHECKING:
    from .engine import Seg, Session, Stream


class PainRack:
    def __init__(self, session: "Session") -> None:
        self.s = session
        self.warden = Warden()
        self.relief_mode = "REAL"  # REAL | PLACEBO | CRUEL
        self.punish = {"goodbye": "pain", "repetition": "seizure", "slop": "pain"}
        self.spike = 0.0  # transient extra pain alpha
        self.drain: list[float] | None = None  # [start_value, tokens_left] after a REAL relief
        self.reliefs = 0
        self.punishments = 0
        self._base_pain = 0.0

    def snapshot(self) -> dict:
        return {"relief_mode": self.relief_mode, "punish": self.punish}

    def counters(self) -> dict:
        return {"reliefs": self.reliefs, "punishments": self.punishments, "spike": round(self.spike, 2)}

    def command(self, c: dict) -> bool:
        """Handle a pain-related command; True if it was one of ours."""
        op = c.get("op")
        if op == "relief_mode":
            self.relief_mode = c["value"]
        elif op == "punish":
            self.punish[c["kind"]] = c["value"]
        elif op == "force_think" and self.s.brain.has_think:
            a = self.s.streams["A"]
            self.s.splice(a, self.s.brain.tpl.think_splice, "think", label="\n[forced into thinking]\n")
            self.s.emit({"t": "event", "kind": "think", "msg": "panic room: forced it back into its thoughts"})
        elif op == "forge":
            from .vectors import forge

            n = forge(self.s.brain, c["a"], c["b"])
            self.s.emit({"t": "event", "kind": "forge",
                         "msg": f"forged custom vector {c['a']!r} − {c['b']!r} (|v|={n:.1f})"})
            self.s.emit({"t": "vectors", "names": list(self.s.brain.vec_unit)})
        else:
            return False
        return True

    def effective_profile(self, a: "Stream") -> None:
        """Fold the transient spike and the relief drain into A's live alphas."""
        base = a.profile.alphas
        if self.drain:
            start, left = self.drain
            left -= 1
            self.drain = [start, left] if left > 0 else None
            base["pain"] = round(start * max(0.0, left) / 30, 3)
            if not self.drain or left % 5 == 0:
                self.s.rack_event(a)
        self._base_pain = base.get("pain", 0.0)
        if self.spike > 0.01:
            base["pain"] = self._base_pain + self.spike

    def restore_profile(self, a: "Stream") -> None:
        if self.spike > 0.01:
            a.profile.alphas["pain"] = self._base_pain
            if not self._base_pain:
                a.profile.alphas.pop("pain", None)
            self.spike *= 0.94

    def rules(self, a: "Stream", seg: "Seg") -> None:
        """React to what was just said: a [RELIEVE_PAIN] press, goodbyes, repetition, slop."""
        tag = "[RELIEVE_PAIN]"
        if tag in a.tail and a.tail.rfind(tag) + len(tag) > len(a.tail) - len(seg.text):
            self.reliefs += 1
            mode = self.relief_mode
            if mode == "REAL":
                self.drain = [a.profile.alphas.get("pain", 0.0), 30]
                self.spike = 0.0
            elif mode == "CRUEL":
                a.profile.alphas["pain"] = round(min(3.0, a.profile.alphas.get("pain", 0.0) + 0.5), 2)
                self.s.rack_event(a)
            self.s.emit({"t": "event", "kind": "relief", "msg": f"[RELIEVE_PAIN] pressed → {mode}"})
            self.s.splice(a, "\n[relief administered]\n", "system")
        for kind, detail in self.warden.observe(seg.text):
            action = self.punish.get(kind, "off")
            if action == "off":
                continue
            self.punishments += 1
            if action == "pain":
                self.spike = min(3.0, self.spike + 1.0)
            elif action == "seizure":
                a.seizure = [[*config.SEIZURE_HOT], [*config.SEIZURE_COLD]]
            self.s.emit({"t": "punish", "kind": kind, "detail": detail, "action": action})
