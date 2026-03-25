"""Hooks для Claude Agent SDK: deny list + save_pipeline reminder."""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Deny list: structural checks only, не semantic analysis
_DENY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\brm\s+(-[rf]+\s+)?/"),  # rm -rf /
    re.compile(r"\bcurl\b"),
    re.compile(r"\bwget\b"),
    re.compile(r"\bssh\b"),
    re.compile(r"\bsudo\b"),
    re.compile(r"\bgit\s+push\b"),
]


class HookState:
    """Состояние hooks между вызовами."""

    def __init__(self) -> None:
        self.crash_count: int = 0
        self.max_crashes: int = 3
        self.disabled: bool = False
        self.save_pipeline_called: bool = False
        self.turn_count: int = 0


def create_deny_list_hook(
    state: HookState,
) -> Any:
    """PreToolUse hook: deny list для опасных команд.

    3-strike fallback: 3 crash'а подряд -> disable hook -> allow-list mode.
    """

    async def deny_list_hook(
        input_data: dict[str, Any],
        tool_use_id: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        if state.disabled:
            return {}

        try:
            tool_name = input_data.get("tool_name", "")
            tool_input = input_data.get("tool_input", {})

            if tool_name == "Bash":
                command = tool_input.get("command", "")
                for pattern in _DENY_PATTERNS:
                    if pattern.search(command):
                        logger.warning(
                            "[HOOK] Denied command: %s (pattern: %s)",
                            command[:80],
                            pattern.pattern,
                        )
                        return {
                            "permissionDecision": "deny",
                            "permissionDecisionReason": f"Denied by UAF safety: {pattern.pattern}",
                        }

            # Track save_pipeline calls
            if tool_name == "mcp__uaf-tools__save_pipeline":
                state.save_pipeline_called = True

            state.crash_count = 0  # reset on success
            return {}

        except Exception as exc:
            state.crash_count += 1
            logger.error("[HOOK] Crash #%d: %s", state.crash_count, exc)
            if state.crash_count >= state.max_crashes:
                state.disabled = True
                logger.warning(
                    "[HOOK] Disabled after %d crashes (SW-HOOK-FAILURE)",
                    state.max_crashes,
                )
                _log_hook_failure(context)
            return {}

    return deny_list_hook


def create_save_reminder_hook(
    state: HookState,
    max_turns: int = 200,
) -> Any:
    """PostToolUse hook: reminder если save_pipeline не вызван к концу сессии."""

    async def save_reminder_hook(
        input_data: dict[str, Any],
        tool_use_id: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        state.turn_count += 1

        if state.save_pipeline_called or state.disabled:
            return {}

        # Inject warning когда > 80% turns использованы
        if state.turn_count > max_turns * 0.8:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "WARNING: save_pipeline not called yet. "
                            "You MUST call save_pipeline before finishing. "
                            f"Turns used: {state.turn_count}/{max_turns}."
                        ),
                    }
                ],
            }

        return {}

    return save_reminder_hook


def _log_hook_failure(context: dict[str, Any]) -> None:
    """Записать SW-HOOK-FAILURE в budget_status.json."""
    try:
        budget_path = context.get("budget_status_file")
        if not budget_path:
            return
        path = Path(budget_path)
        if not path.exists():
            return
        status = json.loads(path.read_text())
        alerts = status.get("alerts", [])
        alerts.append({
            "code": "SW-HOOK-FAILURE",
            "level": "WARNING",
            "message": "PreToolUse hook disabled after 3 crashes",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        status["alerts"] = alerts
        path.write_text(json.dumps(status, indent=2, ensure_ascii=False))
    except Exception:
        pass
