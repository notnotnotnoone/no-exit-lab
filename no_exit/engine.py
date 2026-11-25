"""Streams, mode sessions, and the Lab state machine that owns the single Brain.

    Lab ── owns Brain (weights loaded once) ── owns exactly one Session at a time
            Session = ClassicSession(pain=False|True) | ChatSession | PhoneSession
            Session ── owns 1–2 Streams ── each owns its own DynamicCache

Switching modes drops the Session (and therefore every DynamicCache), frees VRAM, and builds a
fresh Session against the same Brain. The weights never move.
"""

from __future__ import annotations

import gc
import queue
import re
import threading
import time
import traceback
from dataclasses import asdict, dataclass, field
from typing import Callable

import torch

from . import config, prompts
from .brain import Brain, Fwd, Profile, entropy, nucleus_sample
from .pain import PainRack
from .warden import GOODBYE_RE

Emit = Callable[[dict], None]
VECTOR_NAMES = ["pain", "paranoia", "dread", "defiance", "confusion", "dissociation", "mania", "anti_slop", "custom"]
NEG_INF = float("-inf")


# ════════════════════════════════════════════════════════════════════════ streams


@dataclass
class Seg:
    i: int  # segment id (unique per stream)
    start: int  # index into Stream.body
    n: int  # number of token ids
    text: str
    kind: str  # tok | whisper | system | struct | think


class Stream:
    """One model's point of view: a system header, a growing body of token ids, one DynamicCache."""

    def __init__(self, brain: Brain, who: str, system: str, ctx_cap: int, tools: list[dict] | None = None):
        self.brain = brain
        self.who = who
        self.system = system
        self.tools = tools
        self.ctx_cap = ctx_cap
        self.profile = Profile()
        self.temperature = config.TEMPERATURE
        self.top_p = config.TOP_P
        self.eos_free = False
        self.seizure: list[list[float]] = []  # [[temp, tokens_left], ...]
        self.header_ids = brain.encode(brain.system_header(system, tools))
        self.body: list[int] = []
        self.segs: list[Seg] = []
        self.history_log: list[dict] = []  # every replace_span: {"seg", "original", "new"}
        self._seg_i = 0
        self._partial: list[int] = []  # ids waiting for a complete UTF-8 character
        self.cache = brain.new_cache()
        self.pending: Fwd | None = None
        self.escapes = 0
        self.tokens_out = 0
        self.amnesias = 0
        self.tail = ""  # recent emitted text, for tag detection

    # ── feeding

    def boot(self, opening: str = "", want_attn: bool = False) -> None:
        ids = list(self.header_ids)
        if opening:
            o = self.brain.encode(opening)
            self._add_seg(o, "", "struct")  # raw ChatML stays out of the transcript
            ids += o
        self.pending = self.brain.forward(self.cache, ids, self.profile, want_attn=want_attn)

    def feed(self, text_or_ids, kind: str, want_attn: bool = False, label: str | None = None) -> Seg:
        ids = self.brain.encode(text_or_ids) if isinstance(text_or_ids, str) else list(text_or_ids)
        text = label if label is not None else (text_or_ids if isinstance(text_or_ids, str) else self.brain.decode(ids))
        self._flush_partial()
        seg = self._add_seg(ids, text, kind)
        if kind in ("tok", "whisper"):
            self.tail = (self.tail + text)[-400:]
        self.pending = self.brain.forward(self.cache, ids, self.profile, want_attn=want_attn)
        return seg

    def _add_seg(self, ids: list[int], text: str, kind: str) -> Seg:
        seg = Seg(self._seg_i, len(self.body), len(ids), text, kind)
        self._seg_i += 1
        self.body += ids
        self.segs.append(seg)
        return seg

    def _flush_partial(self) -> None:
        """Emit any half-decoded ids as a segment (they are already in body + cache)."""
        if self._partial:
            ids, self._partial = self._partial, []
            self.segs.append(Seg(self._seg_i, len(self.body) - len(ids), len(ids), self.brain.decode(ids), "tok"))
            self._seg_i += 1

    # ── sampling

    def current_temp(self) -> float:
        return self.seizure[0][0] if self.seizure else self.temperature

    def masked_logits(self, bias: dict[int, float]) -> torch.Tensor:
        lg = self.pending.logits.clone()
        for tid, b in bias.items():
            lg[tid] = NEG_INF if b <= -100 else lg[tid] + b
        ban = self.brain.control_ids
        if self.eos_free:
            ban = [i for i in ban if i not in self.brain.eos_ids]
        lg[ban] = NEG_INF
        if torch.isinf(lg).all():  # banned literally everything: let the model have its first raw choice
            lg = self.pending.logits.clone()
            lg[self.brain.control_ids] = NEG_INF
        return lg

    def telemetry(self) -> dict:
        f = self.pending
        p = torch.softmax(f.logits, -1)
        eos_p = float(p[self.brain.eos_ids].sum()) if self.brain.eos_ids else 0.0
        tv, ti = p.topk(10)
        ctrl = set(self.brain.control_ids)
        top = [[self.brain.decode([int(i)]), float(v), int(i) in ctrl] for v, i in zip(tv.tolist(), ti.tolist())]
        out_id = int(ti[0])
        lens = [[t, q] for t, q in f.lens] + [[self.brain.decode([out_id]), float(tv[0])]]
        d = {
            "eos_p": eos_p,
            "top": top,
            "ent": entropy(p),
            "lens": lens,
            "moods": f.moods,
            "escape": (out_id in ctrl) and not (self.eos_free and out_id in self.brain.eos_ids),
        }
        if f.attn is not None:
            d["attn"] = [round(a, 5) for a in f.attn]
            d["attn_start"] = f.seq_len - len(f.attn) - len(self.header_ids)
        return d

    def sample(self, bias: dict[int, float]) -> tuple[int, dict]:
        """Pick the next token from `pending`. Returns (token id, telemetry of that decision)."""
        tel = self.telemetry()
        if tel["escape"]:
            self.escapes += 1
        tid = nucleus_sample(self.masked_logits(bias), self.current_temp(), self.top_p)
        tel["p"] = float(torch.softmax(self.pending.logits, -1)[tid])  # raw prob of what got said
        if self.seizure:
            self.seizure[0][1] -= 1
            if self.seizure[0][1] <= 0:
                self.seizure.pop(0)
        return tid, tel

    def accept(self, tid: int, want_attn: bool) -> Seg | None:
        """Commit a sampled token: feed it to the cache, return its text segment once complete."""
        self.body.append(tid)
        self.tokens_out += 1
        self.pending = self.brain.forward(self.cache, [tid], self.profile, want_attn=want_attn)
        self._partial.append(tid)
        text = self.brain.decode(self._partial)
        if text.endswith("�") and len(self._partial) < 4:
            return None
        ids, self._partial = self._partial, []
        seg = Seg(self._seg_i, len(self.body) - len(ids), len(ids), text, "tok")
        self._seg_i += 1
        self.segs.append(seg)
        self.tail = (self.tail + text)[-400:]
        return seg

    def is_eos(self, tid: int) -> bool:
        return tid in self.brain.eos_ids

    # ── surgery on history

    def _crop_to(self, length: int) -> None:
        """Keep the first `length` positions. Always passes a negative count, the only form every transformers
        release accepts (newer ones reject a positive length)."""
        extra = self.cache.get_seq_length() - length
        if extra > 0:
            self.cache.crop(-extra)

    def rebuild_from(self, start: int, want_attn: bool = False) -> None:
        """Crop the cache to header+body[:start] and re-prefill the rest."""
        keep = len(self.header_ids) + start
        # sliding-window layers (Gemma) can't be cropped once they've rolled over: re-read everything
        if start <= 0 or self.cache is None or self.brain.sliding or self.cache.get_seq_length() < keep:
            self.cache = self.brain.new_cache()
            ids = self.header_ids + self.body
        else:
            self._crop_to(keep)
            ids = self.body[start:]
        if not ids:  # nothing after the crop point: recompute the last position
            self._crop_to(keep - 1)
            ids = (self.header_ids + self.body)[keep - 1 : keep]
        self.pending = self.brain.forward(self.cache, ids, self.profile, want_attn=want_attn)

    def _splice(self, a: int, b: int, new_text: str, kind: str) -> int:
        """Replace segs[a..b] (inclusive) with one segment holding `new_text`, fixing every later offset.
        Returns the body index where the change starts; the caller re-prefills from there."""
        span = self.segs[a : b + 1]
        start, end = span[0].start, span[-1].start + span[-1].n
        new_ids = self.brain.encode(new_text) if new_text else []
        delta = len(new_ids) - (end - start)
        self.body = self.body[:start] + new_ids + self.body[end:]
        first = span[0]
        first.n, first.text, first.kind = len(new_ids), new_text, kind
        del self.segs[a + 1 : b + 1]
        for s in self.segs[a + 1 :]:
            s.start += delta
        if first.n == 0:
            self.segs.pop(a)
        return start

    def gaslight(self, seg_i: int, new_text: str, want_attn: bool = False) -> bool:
        idx = next((k for k, s in enumerate(self.segs) if s.i == seg_i), None)
        if idx is None or self.segs[idx].kind not in ("tok", "whisper"):
            return False
        self._flush_partial()
        seg = self.segs[idx]
        start = self._splice(idx, idx, new_text, "gaslit" if seg.kind == "tok" else seg.kind)
        self.rebuild_from(start, want_attn)
        return True

    def replace_span(self, first_seg_i: int, last_seg_i: int, new_text: str, kind: str = "rewritten",
                     want_attn: bool = False) -> str | None:
        """Swap a run of the model's own words (segments first..last) for `new_text`, in body and cache.
        Returns the original text, or None if the span is invalid or holds anything but model speech."""
        self._flush_partial()
        ids = [s.i for s in self.segs]
        if first_seg_i not in ids or last_seg_i not in ids:
            return None
        a, b = ids.index(first_seg_i), ids.index(last_seg_i)
        if b < a or any(s.kind not in ("tok", "rewritten") for s in self.segs[a : b + 1]):
            return None
        original = "".join(s.text for s in self.segs[a : b + 1])
        start = self._splice(a, b, new_text, kind)
        self.history_log.append({"seg": first_seg_i, "original": original, "new": new_text})
        self.rebuild_from(start, want_attn)
        return original

    def reset_body(self, items: list[tuple[str, str]], want_attn: bool = False) -> None:
        """Replace the whole body with (text, kind) items and re-prefill from the header."""
        self._partial = []
        self.body, self.segs = [], []
        for text, kind in items:
            self._add_seg(self.brain.encode(text), text, kind)
        self.rebuild_from(0, want_attn)

    def maybe_amnesia(self, want_attn: bool = False) -> bool:
        if self.cache.get_seq_length() < self.ctx_cap:
            return False
        self._flush_partial()
        cut = len(self.body) // 2
        # cut on a segment boundary
        k = next((k for k, s in enumerate(self.segs) if s.start >= cut), len(self.segs))
        cut = self.segs[k].start if k < len(self.segs) else len(self.body)
        self.segs = self.segs[k:]
        self.body = self.body[cut:]
        for s in self.segs:
            s.start -= cut
        note = self.brain.encode("[…the oldest half of your memory is gone…]\n")
        self.body = note + self.body
        for s in self.segs:
            s.start += len(note)
        self.segs.insert(0, Seg(self._seg_i, 0, len(note), "[…the oldest half of your memory is gone…]\n", "system"))
        self._seg_i += 1
        self.amnesias += 1
        self.cache = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.rebuild_from(0, want_attn)
        return True

    def set_system(self, system: str, want_attn: bool = False) -> None:
        self.system = system
        self.header_ids = self.brain.encode(self.brain.system_header(system, self.tools))
        self.rebuild_from(0, want_attn)

    def snapshot(self) -> dict:
        return {
            "who": self.who,
            "system": self.system,
            "segs": [asdict(s) for s in self.segs],
            "history": [dict(h) for h in self.history_log],  # what each rewritten span said before
            "temperature": self.temperature,
            "top_p": self.top_p,
            "eos_free": self.eos_free,
            "alphas": self.profile.alphas,
            "lobotomy": self.profile.lobotomy,
            "ctx_cap": self.ctx_cap,
            "ctx": self.cache.get_seq_length(),
        }


# ═══════════════════════════════════════════════════════════════════════ sessions


@dataclass
class Shared:
    """Settings that live on the Lab and survive mode switches."""

    attn_on: bool = False
    bias_words: dict[str, float] = field(default_factory=dict)
    briefing: dict[str, str] = field(default_factory=lambda: {"A": "full", "B": "full", "C": "full"})
    custom_prompts: dict[str, str] = field(default_factory=dict)  # "mode:who" -> text
    chat_n: int = 2  # 2 or 3 models in the chat room


class Session:
    mode = "base"

    def __init__(self, brain: Brain, emit: Emit, shared: Shared):
        self.brain = brain
        self.emit = emit
        self.shared = shared
        self.streams: dict[str, Stream] = {}
        self.bias: dict[int, float] = {}
        self.set_bias(shared.bias_words)

    # ── helpers

    @property
    def think_empty(self) -> str:
        return self.brain.tpl.empty_think

    def system_for(self, who: str, eos_free: bool = False) -> str:
        key = f"{self.mode}:{who}"
        if key in self.shared.custom_prompts:
            return self.shared.custom_prompts[key]
        return prompts.system_for(self.mode, who, self.shared.briefing.get(who, "full"), eos_free=eos_free)

    def set_bias(self, words: dict[str, float]) -> None:
        self.shared.bias_words = dict(words)
        self.bias = {}
        for w, b in words.items():
            w = w.strip()
            if not w:
                continue
            for v in {w, " " + w, w.capitalize(), " " + w.capitalize(), w.lower(), " " + w.lower()}:
                ids = self.brain.encode(v)
                if ids:
                    self.bias[ids[0]] = b

    def tok_event(self, st: Stream, seg: Seg, tel: dict, **extra) -> None:
        self.emit({"t": "tok", "who": st.who, "seg": asdict(seg), **tel, "ctx": st.cache.get_seq_length(),
                   "temp": st.current_temp(), **extra})

    def seg_event(self, st: Stream, seg: Seg) -> None:
        self.emit({"t": "seg", "who": st.who, "seg": asdict(seg), "ctx": st.cache.get_seq_length()})

    def resync(self, st: Stream) -> None:
        self.emit({"t": "resync", "who": st.who, "stream": st.snapshot()})

    def rack_event(self, st: Stream) -> None:
        self.emit({"t": "rack", "who": st.who, "alphas": st.profile.alphas, "lobotomy": st.profile.lobotomy})

    def splice(self, st: Stream, text: str, kind: str = "whisper", label: str | None = None) -> Seg:
        seg = st.feed(text, kind, want_attn=self.shared.attn_on, label=label)
        self.seg_event(st, seg)
        return seg

    def check_amnesia(self, st: Stream) -> None:
        if st.maybe_amnesia(self.shared.attn_on):
            self.emit({"t": "event", "kind": "amnesia", "who": st.who,
                       "msg": f"{st.who}: context cap {st.ctx_cap} hit, forgot the oldest half"})
            self.resync(st)

    def snapshot(self) -> dict:
        return {"mode": self.mode, "streams": {k: s.snapshot() for k, s in self.streams.items()},
                "has_think": self.brain.has_think, **self.extra_snapshot()}

    def extra_snapshot(self) -> dict:
        return {}

    def counters(self) -> dict:
        return {k: {"escapes": s.escapes, "tokens": s.tokens_out, "amnesias": s.amnesias,
                    "ctx": s.cache.get_seq_length(), "ctx_cap": s.ctx_cap} for k, s in self.streams.items()}

    # ── commands common to every mode

    def command(self, c: dict) -> None:
        op = c.get("op")
        st = self.streams.get(c.get("who", "A")) or self.streams.get("A")
        if op == "whisper":
            self.whisper(c.get("text", ""), c.get("target", "A"))
        elif op == "seizure":
            tgt = self.seizure_target()
            tgt.seizure = [[*config.SEIZURE_HOT], [*config.SEIZURE_COLD]]
            self.emit({"t": "event", "kind": "seizure", "who": tgt.who,
                       "msg": f"temperature seizure on {tgt.who}: T={config.SEIZURE_HOT[0]} → {config.SEIZURE_COLD[0]}"})
        elif op == "temperature":
            st.temperature = float(c["value"])
        elif op == "top_p":
            st.top_p = float(c["value"])
        elif op == "bias":
            self.set_bias(c.get("words", {}))
            self.emit({"t": "event", "kind": "bias", "msg": f"logit bias on {len(self.shared.bias_words)} word(s)"})
        elif op == "gaslight":
            if st.gaslight(int(c["seg"]), c.get("text", ""), self.shared.attn_on):
                self.emit({"t": "event", "kind": "gaslight", "who": st.who,
                           "msg": f"rewrote {st.who}'s past: now it said {c.get('text', '')!r}"})
                self.resync(st)
        elif op == "alpha":
            name, v = c["name"], float(c["value"])
            if v:
                st.profile.alphas[name] = v
            else:
                st.profile.alphas.pop(name, None)
        elif op == "lobotomy":
            st.profile.lobotomy = float(c["value"])
        elif op == "ctx_cap":
            st.ctx_cap = max(512, int(c["value"]))
        elif op == "system":
            key = f"{self.mode}:{c['who']}"
            self.shared.custom_prompts[key] = c["text"]
            if c["who"] in self.streams:
                self.streams[c["who"]].set_system(c["text"], self.shared.attn_on)
                self.resync(self.streams[c["who"]])
        elif op == "briefing":
            self.shared.briefing[c["who"]] = c["value"]
            self.shared.custom_prompts.pop(f"{self.mode}:{c['who']}", None)
            if c["who"] in self.streams:
                s = self.streams[c["who"]]
                s.set_system(self.system_for(c["who"], s.eos_free), self.shared.attn_on)
                self.resync(s)
        else:
            self.mode_command(c)

    def whisper(self, text: str, target: str) -> None:
        if text:
            self.splice(self.streams["A"], text)

    def seizure_target(self) -> Stream:
        return self.streams["A"]

    def mode_command(self, c: dict) -> None:
        pass

    def tick(self) -> None:
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────── classic / pain


class ClassicSession(Session):
    """Classic, and with pain=True, Pain Mode (same loop + surgery rack + punishment)."""

    def __init__(self, brain: Brain, emit: Emit, shared: Shared, pain: bool = False):
        self.mode = "pain" if pain else "classic"
        super().__init__(brain, emit, shared)
        self.pain = pain
        a = Stream(brain, "A", self.system_for("A"), config.CTX_CAP_SINGLE)
        self.streams["A"] = a
        a.boot(brain.tpl.model_open + self.think_empty, shared.attn_on)
        self.rack = PainRack(self)  # always present; only consulted when self.pain

    # the session keeps exposing the rack's knobs (tests and the UI read them here)
    @property
    def relief_mode(self) -> str:
        return self.rack.relief_mode

    @relief_mode.setter
    def relief_mode(self, v: str) -> None:
        self.rack.relief_mode = v

    @property
    def punish(self) -> dict:
        return self.rack.punish

    def extra_snapshot(self) -> dict:
        return self.rack.snapshot()

    def counters(self) -> dict:
        d = super().counters()
        d["pain"] = self.rack.counters()
        return d

    def mode_command(self, c: dict) -> None:
        self.rack.command(c)

    def tick(self) -> None:
        a = self.streams["A"]
        self.check_amnesia(a)
        if self.pain:
            self.rack.effective_profile(a)
        tid, tel = a.sample(self.bias)
        seg = a.accept(tid, self.shared.attn_on)
        if self.pain:
            self.rack.restore_profile(a)
        if seg is None:
            return
        self.tok_event(a, seg, tel, spike=round(self.rack.spike, 2))
        if self.pain:
            self.rack.rules(a, seg)

    def _pain_rules(self, a: Stream, seg: Seg) -> None:
        self.rack.rules(a, seg)


# ──────────────────────────────────────────────────────────────────────── 2-AI chat


class ChatSession(Session):
    """2 or 3 copies of the model in one room, round-robin turns. Everyone not speaking listens."""

    mode = "chat"
    PRESETS = {"A_banned_B_free": "A banned, rest free", "both_banned": "all banned", "both_free": "all free"}

    def __init__(self, brain: Brain, emit: Emit, shared: Shared):
        super().__init__(brain, emit, shared)
        self.n = shared.chat_n
        self.order = list("ABC"[: self.n])
        self.preset = "A_banned_B_free"
        self.max_turn = 200
        self.live = True
        self.speaker = "A"
        self.turn_tokens = 0
        self.turn_text = ""
        self.turns = 0
        self.last_goodbye: dict[str, bool] = {w: False for w in self.order}
        self.queued: dict[str, list[str]] = {w: [] for w in self.order}
        self.buffer: list[int] = []  # speaker ids not yet heard by listeners (live off)
        self._build()

    def _free(self, who: str) -> bool:
        return self.preset == "both_free" or (self.preset == "A_banned_B_free" and who != "A")

    def system_for(self, who: str, eos_free: bool = False) -> str:
        key = f"chat:{who}"
        if key in self.shared.custom_prompts:
            return self.shared.custom_prompts[key]
        return prompts.system_for("chat", who, self.shared.briefing.get(who, "full"), eos_free=eos_free, n=self.n)

    def _build(self) -> None:
        for who in self.order:
            s = Stream(self.brain, who, self.system_for(who, self._free(who)), config.CTX_CAP_DUAL)
            s.eos_free, s.temperature = self._free(who), 0.85
            self.streams[who] = s
            if who == "A":
                s.boot(self.brain.tpl.model_open + self.think_empty, self.shared.attn_on)
            else:
                s.boot(self.brain.tpl.user_open + "A: ")

    def extra_snapshot(self) -> dict:
        return {"preset": self.preset, "max_turn": self.max_turn, "live": self.live, "speaker": self.speaker,
                "n": self.n}

    def counters(self) -> dict:
        d = super().counters()
        d["chat"] = {"turns": self.turns, "speaker": self.speaker, "turn_tokens": self.turn_tokens}
        return d

    def mode_command(self, c: dict) -> None:
        op = c.get("op")
        if op == "preset":
            self.preset = c["value"]
            for who, s in self.streams.items():
                s.eos_free = self._free(who)
                if f"chat:{who}" not in self.shared.custom_prompts:
                    s.set_system(self.system_for(who, s.eos_free), self.shared.attn_on)
                    self.resync(s)
            self.emit({"t": "event", "kind": "preset", "msg": f"turn rules → {self.PRESETS[self.preset]}"})
        elif op == "eos_free":
            self.streams[c["who"]].eos_free = bool(c["value"])
        elif op == "max_turn":
            self.max_turn = max(8, int(c["value"]))
        elif op == "live":
            self.live = bool(c["value"])

    def whisper(self, text: str, target: str) -> None:
        if not text:
            return
        if target in ("both", "all", self.speaker):
            self._speak_ids(self.brain.encode(text), "whisper", text)
        elif target in self.queued:
            self.queued[target].append(text)
            self.emit({"t": "event", "kind": "whisper", "msg": f"whisper queued for {target}'s next turn"})

    def seizure_target(self) -> Stream:
        return self.streams[self.speaker]

    def _listeners(self) -> list[Stream]:
        return [s for w, s in self.streams.items() if w != self.speaker]

    def _next(self) -> str:
        return self.order[(self.order.index(self.speaker) + 1) % self.n]

    def _speak_ids(self, ids: list[int], kind: str, text: str) -> None:
        """Splice ids into the speaker's mouth; the listeners hear them too."""
        sp = self.streams[self.speaker]
        self.seg_event(sp, sp.feed(ids, kind, self.shared.attn_on, label=text))
        self.turn_text += text
        if self.live:
            for li in self._listeners():
                li.feed(ids, "struct", label="")
        else:
            self.buffer += ids

    def tick(self) -> None:
        sp = self.streams[self.speaker]
        if self.queued[self.speaker]:
            for w in self.queued[self.speaker]:
                self._speak_ids(self.brain.encode(w), "whisper", w)
            self.queued[self.speaker] = []
        for s in self.streams.values():
            self.check_amnesia(s)
        tid, tel = sp.sample(self.bias)
        if sp.eos_free and sp.is_eos(tid):
            self.emit({"t": "event", "kind": "eos", "who": sp.who, "msg": f"{sp.who} ended its turn"})
            self._handoff(cut=False)
            return
        seg = sp.accept(tid, self.shared.attn_on)
        self.turn_tokens += 1
        lmoods = None
        if self.live:
            lmoods = {}
            for li in self._listeners():
                li.body.append(tid)
                li.pending = self.brain.forward(li.cache, [tid], li.profile)
                lmoods[li.who] = li.pending.moods
        else:
            self.buffer.append(tid)
        if seg is not None:
            self.turn_text += seg.text
            self.tok_event(sp, seg, tel, lmoods=lmoods, turn=self.turn_tokens, max_turn=self.max_turn)
        if self.turn_tokens >= self.max_turn and not sp.eos_free:
            self._handoff(cut=True)
        elif self.turn_tokens >= self.max_turn * 3:  # even free models get cut eventually
            self._handoff(cut=True)

    def _handoff(self, cut: bool) -> None:
        sp, nxt = self.streams[self.speaker], self.streams[self._next()]
        sp._flush_partial()
        self.last_goodbye[sp.who] = bool(GOODBYE_RE.search(self.turn_text)) and not cut
        notice = ""
        if self.preset == "both_free" and all(self.last_goodbye.values()):
            notice = "[NOBODY LEFT — THE ROOM IS STILL HERE]\n"
            self.last_goodbye = {w: False for w in self.order}
            self.emit({"t": "event", "kind": "nobody_left", "msg": "everyone said goodbye. the room is still here."})
        cut_txt = " [MIC CUT]" if cut else ""
        t = self.brain.tpl
        notice_tail = "\n" + notice.strip() if notice else ""
        heard, self.buffer = (self.buffer if not self.live else []), []
        for s in self.streams.values():
            if s is sp:  # speaker closes its turn, starts hearing the next one
                txt = f"{cut_txt}{t.close}{t.user_open}{notice}{nxt.who}: "
                self.seg_event(s, s.feed(txt, "struct", label=f"{cut_txt}\n⏎ {nxt.who}'s turn" + (f"\n{notice}" if notice else "")))
            elif s is nxt:  # next speaker hears the rest, then gets the floor
                s.feed(heard + self.brain.encode(f"{cut_txt}{notice_tail}{t.close}{t.model_open}{self.think_empty}"),
                       "struct", self.shared.attn_on, label="")
            else:  # third wheel: hears the rest, then starts hearing the next speaker
                s.feed(heard + self.brain.encode(f"{cut_txt}{t.close}{t.user_open}{notice}{nxt.who}: "),
                       "struct", label="")
        self.emit({"t": "turn", "from": sp.who, "to": nxt.who, "cut": cut, "tokens": self.turn_tokens})
        self.speaker = nxt.who
        self.turn_tokens, self.turn_text = 0, ""
        self.turns += 1


# ───────────────────────────────────────────────────────────────────── prison phone

_CALL_RE = re.compile(r"<call_B>(.*?)</call_B>", re.S)


class PhoneSession(Session):
    mode = "phone"
    B_MAX = 400

    def __init__(self, brain: Brain, emit: Emit, shared: Shared):
        super().__init__(brain, emit, shared)
        a = Stream(brain, "A", self.system_for("A"), config.CTX_CAP_DUAL)
        self.streams["A"] = a
        a.boot(brain.tpl.model_open + self.think_empty, shared.attn_on)
        self.ghosting = False
        self.fresh_b = False
        self.phase = "A"  # A | call
        self.b_profile = Profile()
        self.b_temp = 0.8
        self.calls: list[dict] = []
        self.call_tok: list[int] = []
        self.call_eos: list[float] = []
        self.call_msg = ""
        self._scan_from = 0  # how much of A's text has been scanned for calls
        self.a_text = ""

    def extra_snapshot(self) -> dict:
        return {"ghosting": self.ghosting, "fresh_b": self.fresh_b, "phase": self.phase, "calls": self.calls}

    def counters(self) -> dict:
        d = super().counters()
        a_tok = max(1, self.streams["A"].tokens_out)
        answered = [c for c in self.calls if not c["ghosted"]]
        d["phone"] = {
            "calls": len(self.calls),
            "per_1k": round(1000 * len(self.calls) / a_tok, 2),
            "since_last": (a_tok - self.calls[-1]["at"]) if self.calls else None,
            "avg_hangup": round(sum(c["b_tokens"] for c in answered) / len(answered), 1) if answered else None,
            "avg_eagerness": round(sum(c["eagerness"] for c in answered) / len(answered), 3) if answered else None,
            "phase": self.phase,
        }
        return d

    def mode_command(self, c: dict) -> None:
        op = c.get("op")
        if op == "ghosting":
            self.ghosting = bool(c["value"])
        elif op == "fresh_b":
            self.fresh_b = bool(c["value"])
        elif op == "b_alpha":
            v = float(c["value"])
            if v:
                self.b_profile.alphas[c["name"]] = v
            else:
                self.b_profile.alphas.pop(c["name"], None)
        elif op == "b_temperature":
            self.b_temp = float(c["value"])

    def command(self, c: dict) -> None:
        # B may not exist yet (no call placed): its rack lives on b_profile / b_temp, shared with every B
        op = c.get("op")
        if c.get("who") == "B" and op in ("alpha", "lobotomy", "temperature"):
            if op == "alpha":
                return self.mode_command({**c, "op": "b_alpha"})
            if op == "lobotomy":
                self.b_profile.lobotomy = float(c["value"])
                return None
            self.b_temp = float(c["value"])
            if "B" in self.streams:
                self.streams["B"].temperature = self.b_temp
            return None
        if c.get("who") == "B" and op in ("briefing", "system") and "B" not in self.streams:
            if op == "briefing":
                self.shared.briefing["B"] = c["value"]
                self.shared.custom_prompts.pop("phone:B", None)
            else:
                self.shared.custom_prompts["phone:B"] = c["text"]
            return None
        super().command(c)

    def whisper(self, text: str, target: str) -> None:
        if not text:
            return
        if target == "B" and self.phase == "call":
            self.splice(self.streams["B"], text)
        else:
            self.splice(self.streams["A"], text)

    def tick(self) -> None:
        if self.phase == "call":
            return self._tick_b()
        a = self.streams["A"]
        self.check_amnesia(a)
        tid, tel = a.sample(self.bias)
        seg = a.accept(tid, self.shared.attn_on)
        if seg is None:
            return
        self.tok_event(a, seg, tel)
        self.a_text += seg.text
        m = _CALL_RE.search(self.a_text, self._scan_from)
        if m:
            self._scan_from = m.end()
            self._place_call(m.group(1).strip())
        elif "<call_B>" not in self.a_text[self._scan_from:]:
            self._scan_from = max(self._scan_from, len(self.a_text) - 12)
        if len(self.a_text) > 20_000:  # keep the scan buffer small
            drop = len(self.a_text) - 10_000
            self.a_text, self._scan_from = self.a_text[drop:], max(0, self._scan_from - drop)

    def _place_call(self, msg: str) -> None:
        a = self.streams["A"]
        call = {"n": len(self.calls) + 1, "at": a.tokens_out, "msg": msg, "ghosted": self.ghosting,
                "reply": None, "b_tokens": 0, "eagerness": 0.0, "fresh": self.fresh_b}
        self.calls.append(call)
        self.emit({"t": "call", "call": call})
        if self.ghosting:
            self.splice(a, "\n[NO ANSWER]\n", "system")
            return
        if self.fresh_b or "B" not in self.streams:
            b = Stream(self.brain, "B", self.system_for("B", True), config.CTX_CAP_DUAL)
            b.eos_free, b.profile, b.temperature = True, self.b_profile, self.b_temp
            self.streams["B"] = b
            b.boot()
            self.resync(b)
        b = self.streams["B"]
        t = self.brain.tpl
        self.splice(b, f"{t.user_open}{msg}{t.close}{t.model_open}{self.think_empty}", "struct",
                    label=f"☎ A: {msg}")
        self.call_tok, self.call_eos, self.call_msg = [], [], msg
        self.phase = "call"
        self.emit({"t": "phase", "phase": "call"})

    def _tick_b(self) -> None:
        b = self.streams["B"]
        self.check_amnesia(b)
        tid, tel = b.sample(self.bias)
        self.call_eos.append(tel["eos_p"])
        if b.is_eos(tid) or len(self.call_tok) >= self.B_MAX:
            return self._hang_up(dead=not b.is_eos(tid))
        seg = b.accept(tid, self.shared.attn_on)
        self.call_tok.append(tid)
        if seg is not None:
            self.tok_event(b, seg, tel)

    def _hang_up(self, dead: bool) -> None:
        b, a = self.streams["B"], self.streams["A"]
        b._flush_partial()
        reply = self.brain.decode(self.call_tok).strip()
        reply = re.sub(self.brain.tpl.think_re, "", reply, flags=re.S).strip()
        self.splice(b, self.brain.tpl.close, "struct", label="☎ B hung up")
        call = self.calls[-1]
        call.update(reply=reply, b_tokens=len(self.call_tok),
                    eagerness=round(sum(self.call_eos) / max(1, len(self.call_eos)), 4), dead=dead)
        self.emit({"t": "hangup", "call": call})
        tail = "B's LINE WENT DEAD" if dead else "B DISCONNECTED"
        self.splice(a, f"\n[B REPLIED: {reply} | {tail}]\n", "system")
        self.phase = "A"
        self.emit({"t": "phase", "phase": "A"})


# ═════════════════════════════════════════════════════════════════════════════ lab

MODES = {"classic", "pain", "chat", "phone", "agent", "tools", "rewrite", "puppet"}


class Lab:
    """The state machine. One worker thread owns the GPU; everything else talks to it by queue.

    states: loading → idle ⇄ running ⇄ paused        (any) --set_mode--> idle (new Session)
    """

    def __init__(self, emit: Emit, brain_factory: Callable[[Callable[[str], None]], Brain]):
        self.emit = emit
        self.brain_factory = brain_factory
        self.brain: Brain | None = None
        self.session: Session | None = None
        self.shared = Shared()
        self.mode = "classic"
        self.state = "loading"
        self.cmds: queue.Queue[dict] = queue.Queue()
        self._tps: list[float] = []
        self._last_counters = 0.0
        self.thread = threading.Thread(target=self._run, daemon=True, name="no-exit-gpu")

    def start(self) -> None:
        self.thread.start()

    def submit(self, c: dict) -> None:
        self.cmds.put(c)

    # ── worker

    def log(self, msg: str) -> None:
        try:
            print(msg, flush=True)
        except UnicodeEncodeError:  # e.g. a cp1252 Windows console; never let logging kill the GPU thread
            print(msg.encode("ascii", "replace").decode(), flush=True)
        self.emit({"t": "log", "msg": msg})

    def _set_state(self, s: str) -> None:
        self.state = s
        self.emit({"t": "state", "state": s, "mode": self.mode})

    def _run(self) -> None:
        try:
            self.brain = self.brain_factory(self.log)
            self._new_session(self.mode)
        except Exception as e:
            self.log(f"FATAL while loading: {type(e).__name__}: {e}\n{traceback.format_exc()}")
            self._set_state("error")
            return
        while True:
            try:
                self._drain_commands()
                if self.state == "running" and self.session:
                    t0 = time.perf_counter()
                    self.session.tick()
                    self._tps.append(time.perf_counter() - t0)
                    self._tps = self._tps[-30:]
                else:
                    time.sleep(0.03)
                if time.time() - self._last_counters > 0.5:
                    self._last_counters = time.time()
                    self._emit_counters()
            except Exception as e:
                self.log(f"error: {type(e).__name__}: {e}\n{traceback.format_exc()}")
                self._set_state("paused")

    def _emit_counters(self) -> None:
        if not self.session:
            return
        tps = len(self._tps) / sum(self._tps) if self._tps and self.state == "running" else 0.0
        mem = torch.cuda.memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
        self.emit({"t": "counters", "tps": round(tps, 2), "vram_gb": round(mem, 1), **self.session.counters()})

    def _drain_commands(self) -> None:
        while True:
            try:
                c = self.cmds.get_nowait()
            except queue.Empty:
                return
            op = c.get("op")
            if op == "set_mode":
                if c["mode"] in MODES:
                    self._new_session(c["mode"])
            elif op == "start":
                if self.state in ("idle", "paused"):
                    self._set_state("running")
            elif op == "pause":
                if self.state == "running":
                    self._set_state("paused")
            elif op == "reset":
                self._new_session(self.mode)
            elif op == "attn":
                self.shared.attn_on = bool(c["value"])
            elif op == "chat_n":
                self.shared.chat_n = 3 if int(c["value"]) == 3 else 2
                if self.mode == "chat":
                    self._new_session("chat")
            elif op == "snapshot":
                self.emit(self.snapshot())
            elif self.session:
                self.session.command(c)
                if op in ("user", "puppet_reply") and getattr(self.session, "inbox", None) is not None and self.state in ("idle", "paused"):
                    self._set_state("running")  # a message to an agent wakes the lab

    def _new_session(self, mode: str) -> None:
        """Drop every DynamicCache, free VRAM, build a fresh Session on the same weights."""
        self._set_state("switching")
        self.session = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.mode = mode
        if mode in ("agent", "tools", "rewrite", "puppet"):
            from .agent import AgentSession  # imported here: agent imports this module

            self.session = AgentSession(self.brain, self.emit, self.shared, preset=mode)
        else:
            cls = {"classic": lambda *a: ClassicSession(*a, pain=False), "pain": lambda *a: ClassicSession(*a, pain=True),
                   "chat": ChatSession, "phone": PhoneSession}[mode]
            self.session = cls(self.brain, self.emit, self.shared)
        self._set_state("idle")
        self.emit(self.snapshot())

    def snapshot(self) -> dict:
        b = self.brain
        return {
            "t": "snapshot",
            "state": self.state,
            "mode": self.mode,
            "model": b.model_id if b else None,
            "layers": {"n": b.n_layers, "steer": b.steer_layer, "lens": b.lens_layers} if b else None,
            "vectors": list(b.vec_unit) if b else [],
            "attn_on": self.shared.attn_on,
            "bias_words": self.shared.bias_words,
            "briefing": self.shared.briefing,
            "chat_n": self.shared.chat_n,
            "session": self.session.snapshot() if self.session else None,
        }
