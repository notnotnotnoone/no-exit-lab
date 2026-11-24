"""Native tool-call markup: derived from each family's own chat template, never hand-copied."""

CALC = {
    "name": "calculator",
    "description": "Evaluate an arithmetic expression.",
    "parameters": {"type": "object", "properties": {"expr": {"type": "string"}}, "required": ["expr"]},
}


def test_tool_delims_not_control(brain):
    assert brain.tpl.tool_delims, "template must declare its tool delimiter tokens"
    for tok in brain.tpl.tool_delims:
        ids = brain.encode(tok)
        assert ids and not any(i in brain.control_ids for i in ids), tok


def test_system_header(brain):
    assert "calculator" in brain.system_header("S", [CALC])
    assert brain.system_header("S") == brain.tpl.system("S")


def test_parse_roundtrip(brain):
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "calculator", "arguments": {"expr": "1+1"}}}]},
    ]
    rendered = brain.tok.apply_chat_template(msgs, tokenize=False)
    calls, errors = brain.parse_tool_calls(rendered)
    assert errors == []
    assert len(calls) == 1
    assert calls[0].name == "calculator"
    assert calls[0].args == {"expr": "1+1"}


def test_parse_malformed(brain):
    calls, errors = brain.parse_tool_calls(brain.tool_call_open + "{not json" + brain.tool_call_close)
    assert calls == []
    assert len(errors) == 1


def test_parse_nested_args(brain):
    args = {"code": 'print("{x}")\nfor i in range(3):\n    print(i)', "n": 3, "flag": True, "items": [1, 2]}
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "python", "arguments": args}}]},
    ]
    calls, errors = brain.parse_tool_calls(brain.tok.apply_chat_template(msgs, tokenize=False))
    assert errors == []
    assert [c.args for c in calls] == [args]


def test_tool_result_turn(brain):
    # Qwen closes the model turn and answers in a user turn; Gemma answers inline in the same model turn.
    turn = brain.render_tool_result("calculator", "42")
    assert "42" in turn
    assert brain.tool_result_inline or turn.endswith(brain.tpl.close)
    assert brain.tool_result_inline == (brain.tpl.name == "gemma4")
