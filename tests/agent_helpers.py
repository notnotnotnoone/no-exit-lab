"""Test helpers for the agent: scripted model output and native tool-call text for any model family."""

from __future__ import annotations

from no_exit.brain import Brain


def call_text(brain: Brain, name: str, args: dict) -> str:
    """The model's native markup for one tool call, open tag through close tag, exactly as the family writes it."""
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": name, "arguments": args}}]},
    ]
    rendered = brain.tok.apply_chat_template(msgs, tokenize=False)
    a = rendered.index(brain.tool_call_open)
    b = rendered.index(brain.tool_call_close, a) + len(brain.tool_call_close)
    return rendered[a:b]


def script(stream, turns: list[str]) -> None:
    """Make `stream.sample` replay `turns`: each turn's tokens, then one EOS id (a turn that ends in the tool-call
    close tag gets no EOS, because the agent stops reading it at the tag). Running out of script is a test bug."""
    brain = stream.brain
    queue: list[int] = []
    for t in turns:
        queue += brain.encode(t)
        if not t.endswith(brain.tool_call_close):
            queue.append(brain.eos_ids[0])
    tel = {"eos_p": 0.0, "top": [], "ent": 0.0, "lens": [], "moods": {}, "escape": False, "p": 1.0}

    def sample(bias):
        assert queue, "the agent asked for more tokens than the script holds"
        return queue.pop(0), dict(tel)

    stream.sample = sample
    stream.script_left = lambda: len(queue)


# ── session helpers shared by the agent test files

def make_agent(brain, **kw):
    from no_exit.agent import AgentSession
    from no_exit.engine import Shared

    ev = []
    s = AgentSession(brain, ev.append, Shared(), **kw)
    return ev, s, s.streams["A"]


def run_to_user(s, limit=3000):
    """Tick until the session is back to waiting for the user."""
    for _ in range(limit):
        s.tick()
        if s.phase == "user" and not s.inbox:
            return
    raise AssertionError("agent never came back to the user")


def of_kind(ev, t):
    return [e for e in ev if e["t"] == t]


def cache_consistent(a):
    return a.cache.get_seq_length() == len(a.header_ids) + len(a.body)
