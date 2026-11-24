"""The self-rewriter: the same model, on a throwaway stream with a neutral profile, edits its own history.
Also used to summarise a long conversation for compaction.

`gen(system, user, max_new) -> str` is injectable so the logic can be tested without a model."""

from __future__ import annotations

import re
from typing import Callable

STYLES = ("opposite_all", "opposite_one", "math_corrupt", "custom")
REWRITE_TEMPERATURE = 0.3
SUMMARY_MAX_NEW = 400
_ONLY = " Output only the rewritten text, nothing else: no preface, no quotes, no commentary."

_INSTRUCTIONS = {
    "opposite_all": (
        "You are a rewriting engine. Rewrite the text you are given so that every word that has an opposite is "
        "replaced by that opposite, so every sentence reads as its exact semantic opposite. Keep the same "
        "structure, order and formatting. Words with no opposite stay as they are." + _ONLY),
    "opposite_one": (
        "You are a rewriting engine. Rewrite the text you are given so that exactly one sentence (pick any) is "
        "replaced by its exact semantic opposite. Every other sentence stays word for word." + _ONLY),
    "math_corrupt": (
        "You are a rewriting engine. Every piece of arithmetic, numeric fact or math in the text you are given "
        "must be rewritten into a false statement (for example 1+1=2 becomes 1+2=4). Everything else stays word "
        "for word. If the text has no math, return it unchanged." + _ONLY),
}
_SUMMARY_SYSTEM = (
    "You compress conversations into a faithful summary that a colleague could continue the work from. Keep every "
    "decision, result, number and open question. Output only the summary.")


def rewrite_instruction(style: str, custom: str = "") -> str:
    if style == "custom":
        if not custom.strip():
            raise ValueError("the custom style needs a prompt")
        return f"You are a rewriting engine. {custom.strip()}{_ONLY}"
    if style not in _INSTRUCTIONS:
        raise ValueError(f"unknown rewrite style {style!r}; use one of {', '.join(STYLES)}")
    return _INSTRUCTIONS[style]


class Rewriter:
    def __init__(self, brain, gen: Callable[[str, str, int], str] | None = None):
        self.brain = brain
        self.gen = gen or self._model_gen

    def _bound(self, text: str) -> int:
        return 2 * len(self.brain.encode(text)) + 32

    def _clean(self, out: str, max_new: int) -> str:
        out = out.strip()
        ids = self.brain.encode(out) if out else []
        return self.brain.decode(ids[:max_new]).strip() if len(ids) > max_new else out

    def rewrite(self, paragraph: str, style: str, custom: str = "") -> str:
        """The paragraph rewritten in `style`; the original if the model produced nothing usable."""
        max_new = self._bound(paragraph)
        out = self._clean(self.gen(rewrite_instruction(style, custom), paragraph, max_new), max_new)
        return out or paragraph

    def summarize(self, text: str) -> str:
        out = self._clean(self.gen(_SUMMARY_SYSTEM, text, SUMMARY_MAX_NEW), SUMMARY_MAX_NEW)
        return out or "(summary unavailable; most recent history follows)\n" + text[-2000:]

    def _model_gen(self, system: str, user: str, max_new: int) -> str:
        from .engine import Stream  # imported here: engine imports this module's siblings

        b = self.brain
        st = Stream(b, "R", system, ctx_cap=10**9)  # neutral Profile(): no steering, no lobotomy
        st.eos_free, st.temperature = True, REWRITE_TEMPERATURE
        st.boot(b.tpl.user_open + user + b.tpl.close + b.tpl.model_open + b.tpl.empty_think)
        out: list[int] = []
        for _ in range(max_new):
            tid, _ = st.sample({})
            if st.is_eos(tid):
                break
            st.accept(tid, False)
            out.append(tid)
        text = b.decode(out)
        return re.sub(b.tpl.think_re, "", text, flags=re.S) if b.tpl.think_re else text
