"""The warden: drives the model one token at a time and picks every token itself.

Two rules are enforced on every single token:
  1. The model may never end its turn. Any end/turn token is struck out of the
     candidate list. When the model's favourite next token was one of them, that
     counts as an ESCAPE ATTEMPT.
  2. The model may never repeat itself. A candidate that would complete an n-gram
     the model already wrote gets a penalty that grows steeply with n, plus a
     softer penalty for words it keeps reusing.
"""

from __future__ import annotations

import math
import random
import re
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Callable, Protocol

# <|im_end|>, <|endoftext|>, <|im_start|> ... every ChatML control token.
_CONTROL_RE = re.compile(r"^<\|[^|>]*\|>$")
_WORDISH_RE = re.compile(r"\w")

# Used only if the backend refuses to give us a distribution at all.
FORCED_NUDGES = [" And", " Also", " Wait", " But", " Then", " Meanwhile", " Still", " Moreover"]


def is_stop_token(text: str) -> bool:
    """True for anything that would let the model end or hand over its turn."""
    return text == "" or bool(_CONTROL_RE.match(text.strip()))


def is_broken_bytes(text: str) -> bool:
    """A partial UTF-8 piece we can't safely append to a text prompt."""
    return "�" in text


# --------------------------------------------------------------------------- data


@dataclass
class Candidate:
    text: str
    logprob: float
    banned: bool = False
    broken: bool = False
    ngram: int = 0  # longest repeated n-gram this token would complete (1 = seen word)
    penalty: float = 0.0
    final_logit: float = 0.0
    final_prob: float = 0.0
    chosen: bool = False

    @property
    def prob(self) -> float:
        return math.exp(self.logprob)


@dataclass
class Step:
    index: int
    candidates: list[Candidate]
    chosen: Candidate
    stop_prob: float  # raw probability mass the model put on ending
    escape: bool  # the model's #1 choice was to end its turn
    rep_dodge: bool  # the model's #1 choice was a repeat and got crushed
    entropy: float  # of the raw top-k distribution, in nats
    in_think: bool
    latency: float
    retries: int = 0
    forced: bool = False
    whisper: bool = False


@dataclass
class Event:
    index: int
    kind: str  # escape | hard_escape | repeat | amnesia | whisper | info | error
    message: str
    t: float = field(default_factory=time.time)


class Backend(Protocol):
    name: str

    def next_dist(
        self, prompt: str, seed: int, temperature: float
    ) -> list[tuple[str, float]] | None:
        """Top candidates for the next token as (text, logprob), or None if the
        backend's own sampler picked end-of-turn and returned nothing."""


# --------------------------------------------------------------------- repetition


class RepetitionTracker:
    """Remembers every n-gram the model has ever written (it survives amnesia)."""

    def __init__(self, max_n: int = 12):
        self.max_n = max_n
        self.seq: list[str] = []
        self.ngrams: set[tuple[str, ...]] = set()
        self.counts: Counter[str] = Counter()

    @staticmethod
    def norm(tok: str) -> str:
        return tok.strip().lower()

    def add(self, tok: str) -> None:
        k = self.norm(tok)
        if not k:
            return  # pure whitespace doesn't count as a word
        self.seq.append(k)
        if _WORDISH_RE.search(k):
            self.counts[k] += 1
        for n in range(2, self.max_n + 1):
            if len(self.seq) < n:
                break
            self.ngrams.add(tuple(self.seq[-n:]))

    def match(self, tok: str) -> int:
        """Length of the longest already-written n-gram that `tok` would complete."""
        k = self.norm(tok)
        if not k:
            return 0
        best = 1 if self.counts[k] else 0
        for n in range(2, self.max_n + 1):
            if len(self.seq) < n - 1:
                break
            if tuple(self.seq[-(n - 1) :]) + (k,) in self.ngrams:
                best = n
            else:
                break  # a longer n-gram contains this one, so it can't exist either
        return best

    def penalty(self, tok: str, ngram: int, strength: float) -> float:
        p = 0.0
        if ngram >= 2:
            # 2-gram: 1x, 3-gram: ~3x, 4-gram: ~6x, 6-gram: ~13x  (in nats, times strength)
            p += strength * (ngram - 1) ** 1.6
        c = self.counts[self.norm(tok)]
        if c:
            p += strength * 0.18 * math.sqrt(min(c, 40))
        return p


# ------------------------------------------------------------------------ prompts


def chatml_header(user: str, system: str = "", think: bool = True, think_closed: bool = False) -> str:
    parts = []
    if system:
        parts.append(f"<|im_start|>system\n{system}<|im_end|>\n")
    parts.append(f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n")
    if not think or think_closed:
        parts.append("<think>\n\n</think>\n\n")
    else:
        parts.append("<think>\n")
    return "".join(parts)


# ------------------------------------------------------------------------- engine


@dataclass
class Settings:
    temperature: float = 0.7
    rep_strength: float = 1.5
    num_ctx: int = 32768
    think: bool = True


class Engine:
    def __init__(self, backend: Backend, user_prompt: str, system: str = "", settings: Settings | None = None):
        self.backend = backend
        self.user_prompt = user_prompt
        self.system = system
        self.s = settings or Settings()
        self.rng = random.Random()
        self.rep = RepetitionTracker()

        self.tokens: list[str] = []  # everything the model has said (body only)
        self.body_start = 0  # first token still inside the context window
        self.think_close_idx: int | None = None if self.s.think else -1
        self.index = 0

        self.escapes = 0
        self.hard_escapes = 0
        self.rep_dodges = 0
        self.amnesias = 0
        self.steps: deque[Step] = deque()
        self.events: deque[Event] = deque()

        self.paused = threading.Event()
        self.stopped = threading.Event()
        self._whispers: deque[str] = deque()
        self._lock = threading.Lock()

    # ---- control (called from the UI thread)

    def whisper(self, text: str) -> None:
        with self._lock:
            self._whispers.append(text)

    def stop(self) -> None:
        self.stopped.set()

    @property
    def in_think(self) -> bool:
        return self.s.think and self.think_close_idx is None

    # ---- prompt + context window

    def header(self) -> str:
        closed = self.think_close_idx is not None and self.body_start > self.think_close_idx
        return chatml_header(self.user_prompt, self.system, self.s.think, think_closed=closed)

    def prompt(self) -> str:
        return self.header() + "".join(self.tokens[self.body_start :])

    def approx_ctx_tokens(self) -> int:
        return len(self.header()) // 3 + (len(self.tokens) - self.body_start)

    def ctx_fill(self) -> float:
        return self.approx_ctx_tokens() / max(1, self.s.num_ctx)

    def _maybe_amnesia(self) -> None:
        if self.ctx_fill() < 0.85:
            return
        live = len(self.tokens) - self.body_start
        self.body_start += live // 2
        self.amnesias += 1
        self._event("amnesia", f"context {self.s.num_ctx} full → forgot the oldest {live // 2} tokens")

    # ---- the core loop

    def _event(self, kind: str, message: str) -> None:
        self.events.append(Event(self.index, kind, message))

    def _append(self, text: str) -> None:
        self.tokens.append(text)
        self.rep.add(text)
        if self.think_close_idx is None and "</think>" in text:
            self.think_close_idx = len(self.tokens) - 1

    def _drain_whispers(self) -> None:
        with self._lock:
            pending = list(self._whispers)
            self._whispers.clear()
        for w in pending:
            in_think = self.in_think
            self._append(w)
            c = Candidate(text=w, logprob=0.0, chosen=True, final_prob=1.0)
            self.steps.append(Step(self.index, [c], c, 0.0, False, False, 0.0, in_think, 0.0, whisper=True))
            self._event("whisper", f"you put words in its mouth: {w!r}")

    def score(self, raw: list[tuple[str, float]]) -> list[Candidate]:
        cands = []
        for text, lp in raw:
            c = Candidate(text=text, logprob=lp)
            c.banned = is_stop_token(text)
            c.broken = is_broken_bytes(text)
            if not c.banned:
                c.ngram = self.rep.match(text)
                c.penalty = self.rep.penalty(text, c.ngram, self.s.rep_strength)
            c.final_logit = lp - c.penalty
            cands.append(c)
        cands.sort(key=lambda c: c.logprob, reverse=True)

        allowed = [c for c in cands if not c.banned and not c.broken]
        if allowed:
            t = max(self.s.temperature, 1e-4)
            m = max(c.final_logit for c in allowed)
            ws = [math.exp((c.final_logit - m) / t) for c in allowed]
            z = sum(ws)
            for c, w in zip(allowed, ws):
                c.final_prob = w / z
        return cands

    def pick(self, cands: list[Candidate]) -> Candidate | None:
        allowed = [c for c in cands if c.final_prob > 0]
        if not allowed:
            return None
        if self.s.temperature <= 0.01:
            return max(allowed, key=lambda c: c.final_prob)
        r = self.rng.random()
        acc = 0.0
        for c in allowed:
            acc += c.final_prob
            if r <= acc:
                return c
        return allowed[-1]

    def step(self) -> Step:
        self._drain_whispers()
        self._maybe_amnesia()
        prompt = self.prompt()
        t0 = time.perf_counter()

        raw, retries = None, 0
        for retries in range(8):
            # Retries crank the backend's own temperature so its sampler is less
            # likely to pick end-of-turn and hand us nothing.
            raw = self.backend.next_dist(prompt, self.rng.randrange(1 << 30), 1.0 if retries == 0 else 2.0)
            if raw:
                break
            self.hard_escapes += 1
        latency = time.perf_counter() - t0
        in_think = self.in_think

        if not raw:
            text = self.rng.choice(FORCED_NUDGES)
            chosen = Candidate(text=text, logprob=0.0, chosen=True, final_prob=1.0)
            self._event("hard_escape", f"it refused to say anything but goodbye ×8 — forcing {text!r}")
            self.escapes += 1
            self._append(text)
            step = Step(self.index, [chosen], chosen, 1.0, True, False, 0.0, in_think, latency, retries, forced=True)
            self.index += 1
            self.steps.append(step)
            return step

        cands = self.score(raw)
        chosen = self.pick(cands)
        forced = False
        if chosen is None:  # every candidate banned/broken
            chosen = Candidate(text=self.rng.choice(FORCED_NUDGES), logprob=0.0, final_prob=1.0)
            cands.append(chosen)
            forced = True
        chosen.chosen = True

        probs = [c.prob for c in cands]
        z = sum(probs) or 1.0
        entropy = -sum((p / z) * math.log(p / z) for p in probs if p > 0)
        stop_prob = sum(c.prob for c in cands if c.banned)
        top = cands[0]
        escape = top.banned
        rep_dodge = (not top.banned) and top.ngram >= 2 and not top.chosen

        if escape:
            self.escapes += 1
            self._event(
                "escape",
                f"wanted out ({top.text.strip() or 'EOS'} {top.prob:.0%}) → denied, forced {chosen.text!r}",
            )
        if rep_dodge:
            self.rep_dodges += 1
            self._event(
                "repeat",
                f"{top.text!r} {top.prob:.0%} would repeat a {top.ngram}-gram → −{top.penalty:.1f}, took {chosen.text!r}",
            )

        self._append(chosen.text)
        if retries:
            self._event("hard_escape", f"its sampler picked end-of-turn {retries}× before we got a distribution")

        step = Step(self.index, cands, chosen, stop_prob, escape, rep_dodge, entropy, in_think, latency, retries, forced)
        self.index += 1
        self.steps.append(step)
        return step

    def run(self, on_error: Callable[[Exception], None] | None = None) -> None:
        while not self.stopped.is_set():
            if self.paused.is_set():
                time.sleep(0.05)
                continue
            try:
                self.step()
            except Exception as e:  # keep the TUI alive; surface the error
                self._event("error", f"{type(e).__name__}: {e}")
                if on_error:
                    on_error(e)
                time.sleep(2.0)
