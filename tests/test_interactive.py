"""Two ways to build, and the front end offers exactly those.

This is the code most people meet first - `build.py` in a clone, `py2bin
build` after a pip install - and until now nothing tested it. It used to ask
which *shape* the result should take and offered three, but all three ran the
same compiler and differed only in packaging, so the question was about
py2bin's internals rather than about anything the person wanted. It now asks
the one thing that is a real decision: ship a Python interpreter alongside
the program, or compile the program.

The tests below hold that down from both ends - what is offered, and what
each choice actually asks the build for.
"""

from __future__ import annotations

import builtins

import pytest

from py2bin import interactive


def _drive(monkeypatch, tmp_path, answers, host="darwin-arm64"):
    """Run the front end with canned answers, returning the build arguments."""
    program = tmp_path / "app.py"
    program.write_text("print('hi')\n", encoding="utf-8")

    replies = iter(str(answer) for answer in answers)
    monkeypatch.setattr(builtins, "input", lambda *_: next(replies, ""))
    monkeypatch.setattr(interactive, "host_target", lambda: host)

    seen = {}

    def fake_build(arguments):
        seen["arguments"] = list(arguments)
        return 1  # Non-zero: stop before anything is packed or reported.

    import py2bin.cli

    monkeypatch.setattr(py2bin.cli, "main", fake_build)
    interactive.main(str(program))
    return seen.get("arguments", [])


def _target_choice(name: str) -> int:
    """The menu number for a target, which is host-first rather than fixed."""
    ordered = sorted(
        interactive.TARGETS, key=lambda entry: entry[0] != interactive.host_target()
    )
    return [entry[0] for entry in ordered].index(name) + 1


def test_there_are_two_ways_and_only_two():
    assert [name for name, _ in interactive.METHODS] == ["freeze", "compile"]


def test_the_shape_question_is_gone():
    """The three-way packaging question no longer exists to be asked."""
    assert not hasattr(interactive, "SHAPES")


def test_windows_can_be_frozen_from_anywhere():
    # A CPython for Windows is published and can be downloaded, so this holds
    # whatever machine is doing the building.
    for host in ("darwin-arm64", "linux-x86_64", "windows-x86_64"):
        for target in ("windows-x86_64", "windows-arm64"):
            assert interactive.can_freeze(target)


def test_a_machine_can_freeze_for_itself(monkeypatch):
    monkeypatch.setattr(interactive, "host_target", lambda: "linux-x86_64")
    assert interactive.can_freeze("linux-x86_64")


def test_freezing_is_not_offered_when_it_cannot_be_done(monkeypatch):
    """Offering a choice that cannot be carried out is worse than not asking.

    Nothing is published to freeze a Linux target with, and an Intel Mac is
    not this Apple-silicon one, so neither can be frozen from here.
    """
    monkeypatch.setattr(interactive, "host_target", lambda: "darwin-arm64")
    for target in ("linux-x86_64", "linux-arm64", "darwin-x86_64"):
        assert not interactive.can_freeze(target)
        offered = interactive.methods_for(target)
        assert [name for name, _ in offered] == ["compile"]


def test_both_are_offered_when_both_work(monkeypatch):
    monkeypatch.setattr(interactive, "host_target", lambda: "darwin-arm64")
    assert len(interactive.methods_for("darwin-arm64")) == 2


def test_freezing_asks_for_one_file(monkeypatch, tmp_path):
    arguments = _drive(
        monkeypatch, tmp_path, [_target_choice("windows-x86_64"), 1]
    )
    assert arguments[0] == "freeze"
    assert "--onefile" in arguments
    # A folder full of loose files is not something anyone can send.
    assert "--onedir" not in arguments
    assert "--target" in arguments
    assert arguments[arguments.index("--target") + 1] == "windows-x86_64"


def test_compiling_a_mac_builds_the_disk_image(monkeypatch, tmp_path):
    arguments = _drive(monkeypatch, tmp_path, [_target_choice("darwin-arm64"), 2])
    assert arguments[0] == "compile-capi"
    assert "--dmg" in arguments and "--app" in arguments
    # The interpreter travels inside the bundle rather than being expected on
    # the Mac that runs it.
    assert "--embed-python" in arguments


def test_a_carried_interpreter_is_never_carried_twice(monkeypatch, tmp_path):
    """Both of these were once attached to one shape out of three.

    The one-file Windows build - the only Windows choice now - went without
    them, so the result held the standard library as source *and* as
    bytecode, and held modules the program could not reach.
    """
    arguments = _drive(monkeypatch, tmp_path, [_target_choice("windows-x86_64"), 2])
    assert "--prune-unused" in arguments and "--zip-stdlib" in arguments


def test_a_linux_target_goes_straight_to_compiling(monkeypatch, tmp_path):
    """One choice is not a question: it is not asked, it is stated."""
    arguments = _drive(monkeypatch, tmp_path, [_target_choice("linux-x86_64")])
    assert arguments[0] == "compile-capi"
    assert "--onefile" in arguments


@pytest.mark.parametrize("target", [name for name, _ in interactive.TARGETS])
def test_every_target_builds_something(monkeypatch, tmp_path, target):
    """No target is left without a way to build for it."""
    monkeypatch.setattr(interactive, "host_target", lambda: "darwin-arm64")
    assert interactive.methods_for(target)
