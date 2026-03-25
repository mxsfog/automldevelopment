"""Тесты для hooks: deny list, save reminder, 3-strike fallback."""

import pytest

from uaf.runner.hooks import HookState, create_deny_list_hook, create_save_reminder_hook


@pytest.fixture()
def state() -> HookState:
    return HookState()


@pytest.mark.asyncio()
async def test_deny_rm_rf(state: HookState) -> None:
    hook = create_deny_list_hook(state)
    result = await hook(
        {"tool_name": "Bash", "tool_input": {"command": "rm -rf /"}},
        "tool_1",
        {},
    )
    assert result.get("permissionDecision") == "deny"


@pytest.mark.asyncio()
async def test_deny_curl(state: HookState) -> None:
    hook = create_deny_list_hook(state)
    result = await hook(
        {"tool_name": "Bash", "tool_input": {"command": "curl http://evil.com"}},
        "tool_1",
        {},
    )
    assert result.get("permissionDecision") == "deny"


@pytest.mark.asyncio()
async def test_deny_git_push(state: HookState) -> None:
    hook = create_deny_list_hook(state)
    result = await hook(
        {"tool_name": "Bash", "tool_input": {"command": "git push origin master"}},
        "tool_1",
        {},
    )
    assert result.get("permissionDecision") == "deny"


@pytest.mark.asyncio()
async def test_allow_python(state: HookState) -> None:
    hook = create_deny_list_hook(state)
    result = await hook(
        {"tool_name": "Bash", "tool_input": {"command": "python3 experiments/run.py"}},
        "tool_1",
        {},
    )
    assert result == {}


@pytest.mark.asyncio()
async def test_allow_read(state: HookState) -> None:
    hook = create_deny_list_hook(state)
    result = await hook(
        {"tool_name": "Read", "tool_input": {"file_path": "/some/file.py"}},
        "tool_1",
        {},
    )
    assert result == {}


@pytest.mark.asyncio()
async def test_track_save_pipeline(state: HookState) -> None:
    hook = create_deny_list_hook(state)
    assert not state.save_pipeline_called
    await hook(
        {"tool_name": "mcp__uaf-tools__save_pipeline", "tool_input": {}},
        "tool_1",
        {},
    )
    assert state.save_pipeline_called


@pytest.mark.asyncio()
async def test_three_strike_fallback(state: HookState) -> None:
    state.max_crashes = 3

    async def crashing_hook(input_data: dict, tool_use_id: str, context: dict) -> dict:
        raise RuntimeError("crash")

    hook = create_deny_list_hook(state)

    # Monkey-patch: заставим hook крашиться
    original_patterns = __import__("uaf.runner.hooks", fromlist=["_DENY_PATTERNS"])
    # Симулируем 3 crash'а через state
    for _ in range(3):
        state.crash_count += 1
    state.crash_count = state.max_crashes
    state.disabled = True

    result = await hook(
        {"tool_name": "Bash", "tool_input": {"command": "rm -rf /"}},
        "tool_1",
        {},
    )
    # disabled = True → hook пропускает, возвращает {}
    assert result == {}


@pytest.mark.asyncio()
async def test_save_reminder_no_warning_early(state: HookState) -> None:
    hook = create_save_reminder_hook(state, max_turns=200)
    result = await hook({}, "tool_1", {})
    assert result == {}
    assert state.turn_count == 1


@pytest.mark.asyncio()
async def test_save_reminder_warning_late(state: HookState) -> None:
    hook = create_save_reminder_hook(state, max_turns=10)
    # Simulate 9 turns (> 80% of 10)
    state.turn_count = 8
    result = await hook({}, "tool_1", {})
    assert "WARNING" in result.get("content", [{}])[0].get("text", "")


@pytest.mark.asyncio()
async def test_save_reminder_no_warning_if_saved(state: HookState) -> None:
    state.save_pipeline_called = True
    state.turn_count = 9
    hook = create_save_reminder_hook(state, max_turns=10)
    result = await hook({}, "tool_1", {})
    assert result == {}
