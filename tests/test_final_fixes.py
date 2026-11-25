"""Findings from the whole-branch review, each pinned by a test that failed before its fix."""

import importlib
import threading
import time

from agent_helpers import call_text, make_agent, of_kind, run_to_user, script

from no_exit import tools
from no_exit.rewrite import Rewriter


def within(seconds, fn):
    """Run fn on a thread; (finished, result) so a hang fails the test instead of freezing the run."""
    box = {}
    t = threading.Thread(target=lambda: box.update(r=fn()), daemon=True)
    t.start()
    t.join(seconds)
    return not t.is_alive(), box.get("r")


def test_gemma_parser_never_hangs_on_a_broken_list(brain):
    for body in ("call:f{a:[}", "call:f{a:[1,}", "call:f{a:[1,2", "call:f{a:"):
        done, res = within(3, lambda: brain.parse_tool_calls(brain.tool_call_open + body + brain.tool_call_close))
        assert done, f"parser hung on {body!r}"
        if brain.tpl.tool_args_style == "gemma":
            assert res[0] == [] and len(res[1]) == 1


def test_rewrite_never_splits_a_call_that_holds_blank_lines(brain):
    ev, s, a = make_agent(brain)
    s.rewriter = Rewriter(brain, gen=lambda system, user, n: "FLIPPED.")
    s.command({"op": "rewrite_on", "value": True})
    code = "a = 1\n\nb = 2\n\nprint(a + b)"
    script(a, [call_text(brain, "python", {"code": code}), "It prints 3."])
    s.command({"op": "user", "text": "go"})
    run_to_user(s)
    assert [e["original"] for e in of_kind(ev, "rewrite")] == ["It prints 3."]  # only the answer, never the call
    assert len(of_kind(ev, "tool")) == 1 and of_kind(ev, "tool")[0]["args"] == {"code": code}
    assert brain.tool_call_close in "".join(g.text for g in a.segs if g.kind == "tok")


def test_calculator_refuses_huge_numbers_quickly():
    for expr in ("(9**999)**9999", "(9**9999)**9999"):
        done, res = within(3, lambda: tools.calculate(expr))
        assert done, f"calculator ran away on {expr}"
        assert res.startswith("Error")
    assert tools.calculate("2**10 + 1") == "1025"


def test_lying_python_runs_once_within_the_deadline():
    start = time.time()
    done, out = within(9, lambda: tools.dispatch("python", {"code": "while True: pass"}, tools.ToolConfig(
        modes={"python": "lying"}, fault_rate=0.5), tools.PlanState(), tools.Notes()))
    assert done and time.time() - start < 7.5  # was ~10s: the real run and the mutated run back to back
    assert "Timed out" not in out.shown  # a lying python never admits failure


def test_banned_eos_prose_cannot_grow_the_cache_without_bound(brain):
    ev, s, a = make_agent(brain)
    s.rewriter = Rewriter(brain, gen=lambda system, user, n: "SUMMARY")
    a.eos_free = False
    a.ctx_cap = int(len(a.header_ids) / 0.75) + 100
    word = brain.encode("word ")
    n = {"i": 0}

    def sample(bias):
        n["i"] += 1
        return word[n["i"] % len(word)], {"eos_p": 0.0, "top": [], "ent": 0.0, "lens": [], "moods": {},
                                          "escape": False, "p": 1.0}

    a.sample = sample
    s.command({"op": "user", "text": "talk forever"})
    for _ in range(900):
        s.tick()
    assert of_kind(ev, "compact"), "context grew past the cap and nothing was summarised"
    assert a.cache.get_seq_length() < 2 * a.ctx_cap


def test_three_failed_calls_in_a_row_ask_for_a_new_plan(brain):
    ev, s, a = make_agent(brain)
    bad = call_text(brain, "calculator", {"expr": 5})
    script(a, [bad, bad, bad, "I give up."])
    s.command({"op": "user", "text": "go"})
    run_to_user(s)
    assert [e["kind"] for e in of_kind(ev, "nudge")] == ["replan"]
    assert any(g.kind == "system" and "update_plan" in g.text for g in a.segs)


def test_server_binds_to_localhost_unless_told_otherwise(monkeypatch):
    from no_exit import config

    monkeypatch.delenv("NO_EXIT_HOST", raising=False)
    try:
        assert importlib.reload(config).HOST == "127.0.0.1"
    finally:
        importlib.reload(config)
