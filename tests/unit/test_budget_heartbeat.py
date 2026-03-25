"""Тесты для heartbeat check в BudgetController."""

import json
import time
from pathlib import Path

from uaf.budget.controller import BudgetConfig, BudgetController


def test_heartbeat_alive(tmp_path: Path) -> None:
    """Heartbeat с свежим timestamp → alive."""
    session_dir = tmp_path / "sessions" / "test_session"
    session_dir.mkdir(parents=True)
    heartbeat = session_dir / ".heartbeat"
    heartbeat.write_text(json.dumps({"timestamp": time.time(), "tool": "Bash"}))

    config = BudgetConfig(mode="fixed", max_iterations=10, metric_name="roi")
    budget_path = tmp_path / "budget_status.json"
    budget_path.write_text("{}")

    controller = BudgetController(
        budget_status_path=budget_path,
        experiment_id="0",
        tracking_uri="http://127.0.0.1:5000",
        config=config,
        session_id="test_session",
    )
    assert controller._is_claude_alive()


def test_heartbeat_dead(tmp_path: Path) -> None:
    """Heartbeat с старым timestamp → dead."""
    session_dir = tmp_path / "sessions" / "test_session"
    session_dir.mkdir(parents=True)
    heartbeat = session_dir / ".heartbeat"
    heartbeat.write_text(json.dumps({"timestamp": time.time() - 600, "tool": "Read"}))

    config = BudgetConfig(mode="fixed", max_iterations=10, metric_name="roi")
    budget_path = tmp_path / "budget_status.json"
    budget_path.write_text("{}")

    controller = BudgetController(
        budget_status_path=budget_path,
        experiment_id="0",
        tracking_uri="http://127.0.0.1:5000",
        config=config,
        session_id="test_session",
    )
    assert not controller._is_claude_alive()


def test_heartbeat_python_longer_threshold(tmp_path: Path) -> None:
    """Python tool → порог 600s вместо 300s."""
    session_dir = tmp_path / "sessions" / "test_session"
    session_dir.mkdir(parents=True)
    heartbeat = session_dir / ".heartbeat"
    # 400 секунд назад, python tool → alive (порог 600)
    heartbeat.write_text(
        json.dumps({"timestamp": time.time() - 400, "tool": "Bash(python3 run.py)"})
    )

    config = BudgetConfig(mode="fixed", max_iterations=10, metric_name="roi")
    budget_path = tmp_path / "budget_status.json"
    budget_path.write_text("{}")

    controller = BudgetController(
        budget_status_path=budget_path,
        experiment_id="0",
        tracking_uri="http://127.0.0.1:5000",
        config=config,
        session_id="test_session",
    )
    assert controller._is_claude_alive()


def test_no_heartbeat_no_pid(tmp_path: Path) -> None:
    """Нет heartbeat и нет PID → alive (default)."""
    session_dir = tmp_path / "sessions" / "test_session"
    session_dir.mkdir(parents=True)

    config = BudgetConfig(mode="fixed", max_iterations=10, metric_name="roi")
    budget_path = tmp_path / "budget_status.json"
    budget_path.write_text("{}")

    controller = BudgetController(
        budget_status_path=budget_path,
        experiment_id="0",
        tracking_uri="http://127.0.0.1:5000",
        config=config,
        session_id="test_session",
    )
    assert controller._is_claude_alive()
