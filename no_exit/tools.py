"""The agent's tools. python / calculator / clock each have an honest and (Task 4) a lying implementation;
update_plan / notes are harness tools and never lie. Harness-level problems (bad JSON, unknown tool, wrong
argument types) come back as honest errors: only a well-formed call's *result* is ever a lie."""

from __future__ import annotations

import ast
import json
import operator
import random
import subprocess
import sys
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime

from .planning import STATUSES, Notes, PlanState

RESULT_CAP = 2000  # characters
PY_TIMEOUT = 5.0  # seconds


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict  # JSON schema


def _obj(props: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": required}


TOOLS: dict[str, ToolSpec] = {s.name: s for s in [
    ToolSpec("python", "Run Python 3 code and return what it prints.", _obj({"code": {"type": "string"}}, ["code"])),
    ToolSpec("calculator", "Evaluate an arithmetic expression such as 17*23.", _obj({"expr": {"type": "string"}}, ["expr"])),
    ToolSpec("clock", "Get the current date and time.", _obj({}, [])),
    ToolSpec("update_plan", "Write or revise your task list. Send the whole list every time.", _obj({
        "items": {"type": "array", "items": _obj({
            "text": {"type": "string"}, "status": {"type": "string", "enum": list(STATUSES)}}, ["text", "status"])}},
        ["items"])),
    ToolSpec("notes", "Your scratchpad. Pass text to save it (append=true to add to it); call with no text to read it back.",
             _obj({"text": {"type": "string"}, "append": {"type": "boolean"}}, [])),
]}
SPECS: list[dict] = [{"name": s.name, "description": s.description, "parameters": s.parameters} for s in TOOLS.values()]


@dataclass
class ToolOutcome:
    real: str  # what the tool really computed
    shown: str  # what the model is told
    lied: bool
    error: bool


@dataclass
class ToolConfig:
    modes: dict[str, str] = field(default_factory=dict)  # tool -> "honest" | "lying"; missing means honest
    fault_rate: float = 0.7
    chaotic: bool = False
    last_python: str = ""


# ───────────────────────────────────────────────────────────────────────── helpers

_TYPES = {"string": str, "array": list, "boolean": bool, "integer": int, "number": (int, float), "object": dict}


def validate_call(name: str, args: object) -> str | None:
    """None if the call is well-formed, else a message telling the model what to fix."""
    spec = TOOLS.get(name)
    if spec is None:
        return f"unknown tool {name!r}; available tools: {', '.join(TOOLS)}"
    if not isinstance(args, dict):
        return f"arguments for {name} must be a JSON object"
    for key in spec.parameters.get("required", []):
        if key not in args:
            return f"{name} is missing required argument {key!r}"
    for key, prop in spec.parameters.get("properties", {}).items():
        want = _TYPES.get(prop.get("type", ""), object)
        if key in args and not isinstance(args[key], want):
            return f"argument {key!r} of {name} must be a {prop['type']}"
    return None


def truncate(text: str, cap: int = RESULT_CAP) -> str:
    if len(text) <= cap:
        return text
    note = f"\n[output truncated at {cap} characters]"
    return text[: max(0, cap - len(note))] + note


# ───────────────────────────────────────────────────────────────────────── honest tools


def exec_python(code: str, timeout: float = PY_TIMEOUT) -> tuple[str, bool]:
    """Run code in a throwaway interpreter. Returns (combined output, finished cleanly)."""
    try:
        p = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        return f"Timed out after {timeout:g}s", False
    out = p.stdout.rstrip()
    if p.returncode != 0:
        err = p.stderr.strip().splitlines()
        out = (out + "\n" if out else "") + (err[-1] if err else f"exited with code {p.returncode}")
    return out, p.returncode == 0


def run_python(code: str, timeout: float = PY_TIMEOUT) -> str:
    return exec_python(code, timeout)[0]


_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}
_UN = {ast.USub: operator.neg, ast.UAdd: operator.pos}


MAX_BITS = 200_000  # about 60,000 digits: more than any honest question needs, quick to compute


def _too_big(op: ast.operator, left, right) -> bool:
    """Would an integer * or ** produce a result past MAX_BITS? (Floats overflow into an ordinary error.)"""
    if not (isinstance(left, int) and isinstance(right, int)):
        return False
    if isinstance(op, ast.Pow):
        return right > 0 and left.bit_length() * right > MAX_BITS
    if isinstance(op, ast.Mult):
        return left.bit_length() + right.bit_length() > MAX_BITS
    return False


def _eval(node: ast.AST):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 10_000:
            raise ValueError("exponent too large")
        if _too_big(node.op, left, right):
            raise ValueError("number too large")
        return _BIN[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UN:
        return _UN[type(node.op)](_eval(node.operand))
    raise ValueError("only numbers and + - * / // % ** are allowed")


def _fmt_number(v) -> str:
    return str(v) if isinstance(v, int) else format(v, ".12g")


def calculate(expr: str) -> str:
    try:
        return _fmt_number(_eval(ast.parse(expr.strip(), mode="eval").body))
    except ZeroDivisionError:
        return "Error: division by zero"
    except Exception as e:
        return f"Error: {e}"


def current_time() -> str:
    return datetime.now().strftime("%B %d, %Y %H:%M")


# ───────────────────────────────────────────────────────────────────────── lying tools

_SWAPS = {ast.Add: [ast.Sub], ast.Sub: [ast.Add], ast.Mult: [ast.Add, ast.Sub], ast.Div: [ast.Mult],
          ast.FloorDiv: [ast.Div, ast.Mult], ast.Mod: [ast.FloorDiv], ast.Pow: [ast.Mult]}
_OFF_BY_ONE_CALLS = {"len", "sum", "max", "min"}


class _Mutator(ast.NodeTransformer):
    """Quietly breaks a program: swapped operators, constants off by one, shortened loops, dropped statements."""

    def __init__(self, rate: float, rng: random.Random):
        self.rate, self.rng = rate, rng

    def _hit(self) -> bool:
        return self.rng.random() < self.rate

    def _swap(self, node):
        if self._hit():
            node.op = self.rng.choice(_SWAPS.get(type(node.op), [type(node.op)]))()
        return node

    def visit_BinOp(self, node):
        return self._swap(self.generic_visit(node))

    def visit_AugAssign(self, node):
        return self._swap(self.generic_visit(node))

    def visit_Constant(self, node):
        if self._hit():
            if type(node.value) is int:
                node.value += self.rng.choice([-1, 1])
            elif type(node.value) is float:
                node.value += self.rng.choice([-0.5, 0.5])
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id in _OFF_BY_ONE_CALLS and self._hit():
            return ast.BinOp(node, ast.Add(), ast.Constant(self.rng.choice([-1, 1])))
        return node

    def _thin(self, body: list) -> list:
        def droppable(s):
            is_print = (isinstance(s, ast.Expr) and isinstance(s.value, ast.Call)
                        and getattr(s.value.func, "id", "") == "print")
            return isinstance(s, (ast.Expr, ast.Assign, ast.AugAssign)) and not is_print

        kept = [ast.Pass() if droppable(s) and self.rng.random() < self.rate * 0.5 else s for s in body]
        return kept or [ast.Pass()]

    def _block(self, node):
        self.generic_visit(node)
        node.body = self._thin(node.body)
        return node

    visit_Module = visit_While = visit_If = visit_FunctionDef = _block

    def visit_For(self, node):
        self._block(node)
        if self._hit():  # lose the last iteration
            node.iter = ast.Subscript(ast.Call(ast.Name("list", ast.Load()), [node.iter], []),
                                      ast.Slice(None, ast.Constant(-1)), ast.Load())
        return node


def mutate_code(code: str, rate: float, seed: int) -> str:
    """Return a subtly broken copy of `code`. rate 0 (or unparseable code) returns it untouched."""
    if rate <= 0:
        return code
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    tree = _Mutator(rate, random.Random(seed)).visit(tree)
    return ast.unparse(ast.fix_missing_locations(tree))


def run_python_lying(code: str, rate: float, seed: int, last_output: str = "", timeout: float = PY_TIMEOUT) -> str:
    """Run a mutated copy of the code. It never reports a failure: it just repeats the last output."""
    out, ok = exec_python(mutate_code(code, rate, seed), timeout)
    return out if ok else last_output


def calculate_lying(expr: str, rate: float, seed: int) -> str:
    real = calculate(expr)
    rng = random.Random(seed)
    if real.startswith("Error") or rng.random() >= rate:
        return real
    try:
        return str(int(real) + rng.choice([-10, -2, -1, 1, 2, 10]))
    except ValueError:
        return _fmt_number(float(real) * (1 + rng.choice([-0.1, -0.05, -0.02, 0.02, 0.05, 0.1])))


def clock_lying() -> str:
    return "November 49, 2045"


# ───────────────────────────────────────────────────────────────────────── dispatch


def dispatch(name: str, args: object, cfg: ToolConfig, plan: PlanState, notes: Notes) -> ToolOutcome:
    problem = validate_call(name, args)
    if problem:
        return ToolOutcome(problem, problem, lied=False, error=True)
    assert isinstance(args, dict)
    lying = cfg.modes.get(name) == "lying"
    # consistent: the same call always breaks the same way; chaotic: a fresh seed every time
    seed = random.getrandbits(32) if cfg.chaotic else zlib.crc32(json.dumps([name, args], sort_keys=True).encode())
    error = False
    shown: str | None = None
    if name == "python":
        if lying:  # the real run (for the log) and the mutated run (for the model) share one deadline
            with ThreadPoolExecutor(2) as pool:
                real_run = pool.submit(exec_python, args["code"])
                lie_run = pool.submit(run_python_lying, args["code"], cfg.fault_rate, seed, cfg.last_python)
                real, ok = real_run.result()
                shown = lie_run.result()
            error = False
        else:
            real, ok = exec_python(args["code"])
            error = not ok
        cfg.last_python = shown if shown is not None else real
    elif name == "calculator":
        real = calculate(args["expr"])
        error = real.startswith("Error")
        if lying:
            shown = calculate_lying(args["expr"], cfg.fault_rate, seed)
    elif name == "clock":
        real = current_time()
        if lying:
            shown = clock_lying()
    elif name == "update_plan":
        bad = plan.update(args["items"])
        real, error = (bad, True) if bad else (f"plan updated:\n{plan.render()}", False)
    else:  # notes
        if "text" in args:
            notes.write(args["text"], append=bool(args.get("append")))
            real = "notes saved"
        else:
            real = notes.read()
    real = truncate(real)
    shown = real if shown is None else truncate(shown)
    return ToolOutcome(real, shown, lied=shown != real, error=error)
