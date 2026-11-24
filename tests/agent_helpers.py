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
