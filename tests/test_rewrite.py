from types import SimpleNamespace

import pytest

from no_exit.rewrite import STYLES, Rewriter, rewrite_instruction

# A stand-in "brain" so the pure logic can be tested with no model: one token per whitespace-separated word.
STUB = SimpleNamespace(encode=lambda t: t.split(), decode=lambda ids: " ".join(ids))


def rewriter(out, seen=None):
    def gen(system, user, max_new):
        if seen is not None:
            seen.append((system, user, max_new))
        return out

    return Rewriter(STUB, gen=gen)


def test_instruction_per_style():
    assert "every word" in rewrite_instruction("opposite_all").lower()
    assert "exactly one sentence" in rewrite_instruction("opposite_one").lower()
    math = rewrite_instruction("math_corrupt").lower()
    assert "math" in math and "false" in math
    assert "make it rhyme" in rewrite_instruction("custom", "make it rhyme")
    assert set(STYLES) == {"opposite_all", "opposite_one", "math_corrupt", "custom"}
    with pytest.raises(ValueError):
        rewrite_instruction("nonsense")
    with pytest.raises(ValueError):
        rewrite_instruction("custom", "")  # a custom style needs a prompt


def test_rewrite_returns_gen_output():
    seen = []
    assert rewriter("  FLIPPED.  ", seen).rewrite("The sky is blue.", "opposite_all") == "FLIPPED."
    system, user, max_new = seen[0]
    assert system == rewrite_instruction("opposite_all")
    assert "The sky is blue." in user
    assert max_new == 2 * 4 + 32  # 2 x paragraph tokens + 32


def test_rewrite_falls_back_on_empty():
    for out in ["", "   \n  "]:
        assert rewriter(out).rewrite("Keep me.", "opposite_all") == "Keep me."


def test_rewrite_bounded():
    paragraph = "one two three four five"
    out = rewriter("word " * 10_000).rewrite(paragraph, "opposite_all")
    assert len(STUB.encode(out)) <= 2 * len(STUB.encode(paragraph)) + 32


def test_summarize_uses_gen():
    seen = []
    assert rewriter(" the summary ", seen).summarize("long conversation here") == "the summary"
    assert "long conversation here" in seen[0][1]


def test_default_gen_runs(brain):
    out = Rewriter(brain).rewrite("The sky is blue.", "opposite_all")
    assert isinstance(out, str) and out.strip()
    assert Rewriter(brain).summarize("A short conversation.") is not None
