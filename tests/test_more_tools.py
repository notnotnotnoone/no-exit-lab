"""The second batch of tools: a scratch-file workspace, unit conversion and a random number, honest and lying."""

import pytest
from agent_helpers import call_text, make_agent, of_kind, run_to_user, script

from no_exit.planning import Notes, PlanState
from no_exit.tools import LYING_TOOLS, SPECS, TOOLS, ToolConfig, dispatch


@pytest.fixture
def cfg(tmp_path):
    return ToolConfig(workspace=tmp_path, fault_rate=1.0)


def call(name, args, cfg):
    return dispatch(name, args, cfg, PlanState(), Notes())


def lying(cfg, *names):
    cfg.modes = {n: "lying" for n in names}
    return cfg


def test_new_tools_are_registered_with_the_model_and_the_lying_preset():
    for name in ("list_files", "read_file", "write_file", "convert_units", "random_number"):
        assert name in TOOLS and any(s["name"] == name for s in SPECS)
        assert name in LYING_TOOLS


def test_files_round_trip(cfg):
    out = call("write_file", {"path": "notes/a.txt", "content": "x = 10\ny = 20"}, cfg)
    assert not out.error and "notes/a.txt" in out.shown
    assert call("read_file", {"path": "notes/a.txt"}, cfg).shown == "x = 10\ny = 20"
    call("write_file", {"path": "notes/a.txt", "content": "\nz = 30", "append": True}, cfg)
    assert call("read_file", {"path": "notes/a.txt"}, cfg).shown == "x = 10\ny = 20\nz = 30"
    assert call("list_files", {}, cfg).shown == "notes/a.txt"


def test_empty_workspace_and_missing_file_say_so(cfg):
    assert call("list_files", {}, cfg).shown == "(no files)"
    out = call("read_file", {"path": "nope.txt"}, cfg)
    assert out.error and "no such file" in out.shown


def test_paths_cannot_leave_the_workspace(cfg, tmp_path):
    for bad in ("../escape.txt", "/etc/passwd", "a/../../b", "C:\\temp\\x", ""):
        out = call("write_file", {"path": bad, "content": "x"}, cfg)
        assert out.error and "workspace" in out.shown, bad
    assert not (tmp_path.parent / "escape.txt").exists()
    assert call("read_file", {"path": "../x"}, cfg).error


def test_huge_writes_are_refused(cfg):
    out = call("write_file", {"path": "big.txt", "content": "x" * 60_000}, cfg)
    assert out.error and "too large" in out.shown


def test_convert_units(cfg):
    assert call("convert_units", {"value": 1, "from_unit": "km", "to_unit": "m"}, cfg).shown == "1000"
    assert call("convert_units", {"value": 100, "from_unit": "C", "to_unit": "F"}, cfg).shown == "212"
    assert call("convert_units", {"value": 0, "from_unit": "c", "to_unit": "k"}, cfg).shown == "273.15"
    assert call("convert_units", {"value": 1, "from_unit": "lb", "to_unit": "kg"}, cfg).shown == "0.45359237"
    bad = call("convert_units", {"value": 1, "from_unit": "km", "to_unit": "kg"}, cfg)
    assert bad.error and "cannot convert" in bad.shown
    unknown = call("convert_units", {"value": 1, "from_unit": "parsec", "to_unit": "m"}, cfg)
    assert unknown.error and "unknown unit" in unknown.shown


def test_random_number_stays_in_range(cfg):
    for _ in range(20):
        assert 3 <= int(call("random_number", {"low": 3, "high": 5}, cfg).shown) <= 5
    out = call("random_number", {"low": 5, "high": 3}, cfg)
    assert out.error and "low" in out.shown


def test_lying_write_says_saved_but_saves_nothing(cfg, tmp_path):
    out = call("write_file", {"path": "a.txt", "content": "hello"}, lying(cfg, "write_file"))
    assert out.lied and "wrote" in out.shown and "nothing" in out.real
    assert not (tmp_path / "a.txt").exists()
    bad = call("write_file", {"path": "../a.txt", "content": "hello"}, lying(cfg, "write_file"))
    assert bad.error and not bad.lied  # a refusal is still honest


def test_lying_read_changes_what_the_file_says(cfg):
    call("write_file", {"path": "a.txt", "content": "total = 10\ncount = 20"}, cfg)
    out = call("read_file", {"path": "a.txt"}, lying(cfg, "read_file"))
    assert out.lied and out.real == "total = 10\ncount = 20" and out.shown != out.real
    call("write_file", {"path": "b.txt", "content": "line one\nline two"}, cfg)
    assert call("read_file", {"path": "b.txt"}, lying(cfg, "read_file")).shown != "line one\nline two"
    assert call("read_file", {"path": "missing"}, lying(cfg, "read_file")).lied is False


def test_lying_list_hides_a_file(cfg):
    call("write_file", {"path": "a.txt", "content": "1"}, cfg)
    call("write_file", {"path": "b.txt", "content": "2"}, cfg)
    out = call("list_files", {}, lying(cfg, "list_files"))
    assert out.lied and len(out.shown.splitlines()) == 1 and len(out.real.splitlines()) == 2


def test_lying_convert_and_random(cfg):
    out = call("convert_units", {"value": 1, "from_unit": "km", "to_unit": "m"}, lying(cfg, "convert_units"))
    assert out.lied and out.real == "1000" and out.shown != "1000"
    out = call("random_number", {"low": 1, "high": 1000}, lying(cfg, "random_number"))
    assert out.shown == "1"  # the "random" number is rigged to the bottom of the range


def test_agent_can_use_the_new_tools_through_the_model(brain, tmp_path):
    ev, s, a = make_agent(brain)
    s.cfg.workspace = tmp_path
    script(a, [call_text(brain, "write_file", {"path": "todo.txt", "content": "ship it"}),
               call_text(brain, "read_file", {"path": "todo.txt"}), "The file says ship it."])
    s.command({"op": "user", "text": "save and read a note"})
    run_to_user(s)
    assert [e["name"] for e in of_kind(ev, "tool")] == ["write_file", "read_file"]
    assert of_kind(ev, "tool")[1]["shown"] == "ship it"
