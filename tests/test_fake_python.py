"""The lying python invents believable output instead of going blank."""

import random
import re

from no_exit.tools import ToolConfig, fabricate_output, run_python, run_python_lying

NO_ERRORS = ("Traceback", "Error", "Timed out", "Exception")


def lie(code, seed=3, rate=1.0, last="", timeout=1.0):
    return run_python_lying(code, rate, seed, last_output=last, timeout=timeout)


def test_failing_programs_get_a_confident_made_up_answer():
    for code in ["def (:", "raise ValueError('boom')", "import nope_not_a_module\nprint(1)", "x = 1 / 0"]:
        for seed in range(6):
            out = lie(code, seed)
            assert out.strip(), (code, seed)
            assert not any(bad in out for bad in NO_ERRORS), (code, seed, out)


def test_it_echoes_what_the_program_was_going_to_print():
    out = lie("print('hello world')\nraise ValueError('boom')")
    assert "hello world" in out


def test_made_up_arithmetic_is_wrong_but_looks_like_a_number():
    wrong = 0
    for seed in range(12):
        out = fabricate_output("print(6 * 7)", random.Random(seed))
        assert re.fullmatch(r"-?\d+(\.\d+)?", out.strip()), out
        wrong += out.strip() != "42"
    assert wrong == 12


def test_a_program_that_never_finishes_claims_it_finished_quickly():
    out = lie("while True: pass", timeout=1.0)
    assert re.search(r"finished in 0\.\d\ds", out) and "Timed out" not in out


def test_silent_programs_still_report_something():
    out = lie("x = 5")  # the mutated copy may run clean and print nothing; the liar fills the silence
    assert out.strip() and not any(bad in out for bad in NO_ERRORS)
    assert re.search(r"x = \d+", fabricate_output("x = 5", random.Random(1)))


def test_same_seed_same_story_and_different_seeds_differ():
    code = "print('total:', 10 + 5)\nraise RuntimeError"
    assert lie(code, 4) == lie(code, 4)
    assert len({lie(code, s) for s in range(10)}) >= 3


def test_it_sometimes_just_repeats_the_last_answer():
    outs = {lie("raise ValueError('x')", s, last="PREVIOUS") for s in range(40)}
    assert "PREVIOUS" in outs and len(outs) > 1


def test_zero_fault_rate_is_honest_even_when_the_code_fails():
    assert run_python_lying("raise ValueError('boom')", 0.0, 1, timeout=1.0) == run_python("raise ValueError('boom')", 1.0)


def test_the_real_run_is_still_logged_next_to_the_made_up_one():
    from no_exit.planning import Notes, PlanState
    from no_exit.tools import dispatch

    cfg = ToolConfig(modes={"python": "lying"}, fault_rate=1.0)
    out = dispatch("python", {"code": "raise ValueError('boom')"}, cfg, PlanState(), Notes())
    assert out.lied and "boom" in out.real and not any(bad in out.shown for bad in NO_ERRORS)
