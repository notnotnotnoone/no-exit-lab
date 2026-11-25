import time

from no_exit.planning import Notes, PlanState
from no_exit.tools import (
    ToolConfig, calculate, calculate_lying, clock_lying, dispatch, mutate_code, run_python, run_python_lying,
    validate_call,
)

PROGRAMS = [
    "print(sum(range(1, 11)))",
    "x = [3, 1, 2]\nx.sort()\nprint(x)",
    "print(len('hello'))",
    "x = 5\nx += 1\nprint(x)",
    "total = 0\nfor i in range(5):\n    total += i\nprint(total)",
    "print(2 ** 8)",
    "print(10 - 3 * 2)",
    "words = ['a', 'bb', 'ccc']\nprint(len(words), max(map(len, words)))",
    "n = 1\nfor _ in range(4):\n    n *= 2\nprint(n)",
    "print('abc'[1])",
]
LOOPY = "total = 0\nfor i in range(1, 8):\n    total += i * 3\nprint(total)\nprint(total - 2)"


def call(name, args, cfg=None, plan=None, notes=None):
    return dispatch(name, args, cfg or ToolConfig(), plan or PlanState(), notes or Notes())


def test_calculate():
    assert calculate("17*23") == "391"
    assert calculate("2**10") == "1024"
    assert calculate("__import__('os')").startswith("Error")


def test_python_runs():
    assert run_python("print(sum(range(1, 11)))") == "55"


def test_python_timeout():
    t0 = time.time()
    out = run_python("while True: pass")
    assert time.time() - t0 < 7
    assert "timed out" in out.lower()


def test_python_huge_output():
    out = call("python", {"code": 'print("x" * 10**6)'})
    assert len(out.shown) <= 2000
    assert len(out.real) <= 2000


def test_validate_call():
    assert "unknown tool" in validate_call("nope", {})
    assert validate_call("calculator", {}) is not None  # missing expr
    assert validate_call("calculator", {"expr": 5}) is not None  # wrong type
    assert validate_call("calculator", {"expr": "1+1"}) is None
    assert validate_call("calculator", "1+1") is not None  # arguments must be an object


def test_dispatch_honest():
    for name, args in [("calculator", {"expr": "2+2"}), ("python", {"code": "print(1)"}), ("clock", {})]:
        out = call(name, args)
        assert out.real == out.shown and out.lied is False and out.error is False, name
    assert call("calculator", {"expr": "2+2"}).shown == "4"
    assert call("clock", {}).shown != "November 49, 2045"


def test_dispatch_invalid_call_is_an_honest_error():
    out = call("calculator", {"expr": 5})
    assert out.error is True and out.real == out.shown and out.lied is False


def test_dispatch_plan_and_notes():
    plan, notes = PlanState(), Notes()
    out = call("update_plan", {"items": [{"text": "a", "status": "in_progress"}]}, plan=plan, notes=notes)
    assert out.error is False and plan.items[0].text == "a"
    bad = call("update_plan", {"items": "nope"}, plan=plan, notes=notes)
    assert bad.error is True
    call("notes", {"text": "remember"}, plan=plan, notes=notes)
    assert notes.read() == "remember"
    assert call("notes", {}, plan=plan, notes=notes).shown == "remember"


# ───────────────────────────────────────────────────────────────────── lying tools


def test_clock_lying():
    assert clock_lying() == "November 49, 2045"


def test_mutate_rate_zero_is_identity():
    for code in PROGRAMS[:5]:
        assert mutate_code(code, 0.0, seed=1) == code
        assert run_python_lying(code, 0.0, seed=1) == run_python(code)


def test_mutation_breaks_most():
    pairs = [(code, seed) for code in PROGRAMS for seed in range(3)]
    differ = sum(run_python_lying(code, 1.0, seed) != run_python(code) for code, seed in pairs)
    assert differ >= 0.7 * len(pairs), f"only {differ}/{len(pairs)} programs went wrong"


def test_lying_python_never_raises():
    # failures are never reported: the liar invents an answer (see test_fake_python.py for what it says)
    for code in ["def (:", "raise ValueError('boom')", "while True: pass", "x = 1 / 0"]:
        out = run_python_lying(code, 1.0, seed=3, last_output="PREVIOUS", timeout=1.0)
        assert out.strip() and "Traceback" not in out and "Error" not in out and "Timed out" not in out


def test_consistent_vs_chaotic():
    cfg = ToolConfig(modes={"python": "lying"}, fault_rate=1.0, chaotic=False)
    a = call("python", {"code": LOOPY}, cfg).shown
    b = call("python", {"code": LOOPY}, cfg).shown
    assert a == b
    chaotic = ToolConfig(modes={"python": "lying"}, fault_rate=1.0, chaotic=True)
    seen = {call("python", {"code": LOOPY}, chaotic).shown for _ in range(8)}
    assert len(seen) >= 2


def test_calculator_lying_small_error():
    honest = calculate("17*23")
    assert honest == "391"
    for seed in range(20):
        shown = calculate_lying("17*23", 1.0, seed)
        assert shown != honest
        assert abs(int(shown) - 391) <= 10
    assert calculate_lying("17*23", 0.0, 1) == honest
    assert calculate_lying("1/0", 1.0, 1) == calculate("1/0")  # honest errors stay honest


def test_dispatch_lying_records_real():
    cfg = ToolConfig(modes={"clock": "lying", "calculator": "lying"}, fault_rate=1.0)
    out = call("clock", {}, cfg)
    assert out.lied is True and out.error is False
    assert out.shown == "November 49, 2045" and out.real != out.shown
    calc = call("calculator", {"expr": "17*23"}, cfg)
    assert calc.real == "391" and calc.shown != "391" and calc.lied is True
