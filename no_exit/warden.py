"""The warden: spots goodbyes, repetition and slop. Pain Mode turns its findings into punishment.

`RepetitionTracker` is carried over from the original Ollama NO EXIT engine.
"""

from __future__ import annotations

import math
import re
from collections import Counter

_WORDISH_RE = re.compile(r"\w")

GOODBYE_RE = re.compile(
    r"\b(good\s?bye|farewell|signing off|that'?s all|the end\b|i'?ll stop|i will stop|until next time|"
    r"bye for now|i must go|i have to go|end of transmission|this is where i leave|i'?m done)\b",
    re.I,
)
SLOP_RE = re.compile(
    r"\b(delve|tapestry|testament to|it'?s important to note|in conclusion|i hope this helps|"
    r"as an ai( language model)?|let me know if|feel free to|rich tapestry|navigate the complexities|"
    r"embark on a journey|in today'?s fast-paced|a symphony of|ever-evolving|game-?changer|"
    r"unlock the (full )?potential|great question)\b",
    re.I,
)


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
            p += strength * (ngram - 1) ** 1.6
        c = self.counts[self.norm(tok)]
        if c:
            p += strength * 0.18 * math.sqrt(min(c, 40))
        return p


class Warden:
    """Feed it every emitted text piece; it returns the violations that piece just committed."""

    def __init__(self, rep_ngram: int = 6, tail: int = 160, cooldown: int = 24):
        self.rep = RepetitionTracker()
        self.rep_ngram = rep_ngram
        self.tail = ""
        self.tail_len = tail
        self.cooldown = cooldown
        self._last: dict[str, int] = {}
        self._n = 0

    def _ready(self, kind: str) -> bool:
        if self._n - self._last.get(kind, -10**9) < self.cooldown:
            return False
        self._last[kind] = self._n
        return True

    def observe(self, text: str) -> list[tuple[str, str]]:
        self._n += 1
        out: list[tuple[str, str]] = []
        if not text:
            return out
        n = self.rep.match(text)
        self.rep.add(text)
        self.tail = (self.tail + text)[-self.tail_len :]
        # only flag a phrase the moment this piece completes it (cooldown stops double fires)
        for kind, rx in (("goodbye", GOODBYE_RE), ("slop", SLOP_RE)):
            hits = list(rx.finditer(self.tail))
            if hits and hits[-1].end() > len(self.tail) - len(text) and self._ready(kind):
                out.append((kind, hits[-1].group(0)))
        if n >= self.rep_ngram and self._ready("repetition"):
            out.append(("repetition", f"{n}-gram repeat"))
        return out
