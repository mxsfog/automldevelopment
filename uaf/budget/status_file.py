"""Структуры данных и атомарная запись budget_status.json v2.1."""

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AlertEntry:
    """Одна запись алерта в budget_status.json.

    Атрибуты:
        code: код алерта (например SW-HANG, MQ-NAN-CASCADE).
        level: уровень (CRITICAL/WARNING/INFO).
        message: текстовое описание.
        timestamp: время возникновения (unix timestamp).
    """

    code: str
    level: str
    message: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class SoftwareHealth:
    """Состояние программного окружения.

    Атрибуты:
        mlflow_reachable: MLflow сервер доступен.
        disk_free_gb: свободное место на диске.
        process_alive: процесс Claude Code жив.
        last_stdout_age_seconds: сколько секунд назад был последний вывод в stdout.
    """

    mlflow_reachable: bool = True
    disk_free_gb: float = 0.0
    process_alive: bool = True
    last_stdout_age_seconds: float = 0.0


@dataclass
class DataQuality:
    """Состояние качества данных.

    Атрибуты:
        schema_hash: хэш схемы входных данных (для обнаружения изменений).
        files_modified: входные данные были изменены во время сессии.
    """

    schema_hash: str = ""
    files_modified: bool = False


@dataclass
class BudgetStatusV21:
    """Полная структура budget_status.json v2.1.

    Версия 2.1 добавляет поля мониторинга (software_health, data_quality,
    alerts, hints, phase) к базовой структуре v1.0.

    Атрибуты:
        version: версия схемы.
        session_id: идентификатор сессии.
        iterations_used: количество использованных итераций.
        iterations_limit: лимит итераций (None для dynamic mode).
        time_elapsed: прошедшее время в секундах.
        time_limit: лимит времени в секундах (None если не задан).
        hard_stop: флаг принудительной остановки.
        hard_stop_reason: причина hard stop.
        phase: текущая фаза (monitoring/grace_period/stopping).
        alerts: список активных алертов.
        hints: подсказки для Claude Code.
        software_health: состояние программного окружения.
        data_quality: состояние качества данных.
        metrics_history: история значений целевой метрики.
        budget_fraction_used: доля использованного бюджета (0.0-1.0).
        warning_triggered: сработало ли 80% предупреждение.
        convergence_signal: сигнал конвергенции от Claude Code (0.0-1.0).
        runs_per_iteration: количество MLflow runs на одну итерацию.
        timestamp: время последнего обновления.
    """

    version: str = "2.1"
    session_id: str = ""
    iterations_used: int = 0
    iterations_limit: int | None = None
    time_elapsed: float = 0.0
    time_limit: float | None = None
    hard_stop: bool = False
    hard_stop_reason: str | None = None
    phase: str = "monitoring"
    alerts: list[AlertEntry] = field(default_factory=list)
    hints: list[str] = field(default_factory=list)
    software_health: SoftwareHealth = field(default_factory=SoftwareHealth)
    data_quality: DataQuality = field(default_factory=DataQuality)
    metrics_history: list[float] = field(default_factory=list)
    budget_fraction_used: float = 0.0
    warning_triggered: bool = False
    convergence_signal: float = 0.0
    runs_per_iteration: int = 1
    timestamp: float = field(default_factory=time.time)


def write_budget_status(status: BudgetStatusV21, path: Path) -> None:
    """Атомарно записывает budget_status.json через os.replace().

    Атомарность достигается записью в tmp-файл с последующим rename,
    что гарантирует что читатель получит либо старую, либо полную новую версию.

    Args:
        status: полная структура статуса бюджета.
        path: путь к budget_status.json.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _status_to_dict(status)
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(data, indent=2))
    os.replace(tmp_path, path)


def read_budget_status(path: Path) -> BudgetStatusV21 | None:
    """Читает budget_status.json и возвращает структуру.

    Args:
        path: путь к budget_status.json.

    Returns:
        Структура статуса или None если файл не существует.
    """
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return _dict_to_status(data)
    except (json.JSONDecodeError, KeyError):
        return None


def _status_to_dict(status: BudgetStatusV21) -> dict[str, Any]:
    """Конвертирует BudgetStatusV21 в словарь для JSON.

    Args:
        status: структура статуса.

    Returns:
        Словарь готовый к сериализации в JSON.
    """
    alerts_list = [
        {
            "code": a.code,
            "level": a.level,
            "message": a.message,
            "timestamp": a.timestamp,
        }
        for a in status.alerts
    ]
    return {
        "version": status.version,
        "session_id": status.session_id,
        "iterations_used": status.iterations_used,
        "iterations_limit": status.iterations_limit,
        "time_elapsed": status.time_elapsed,
        "time_limit": status.time_limit,
        "hard_stop": status.hard_stop,
        "hard_stop_reason": status.hard_stop_reason,
        "phase": status.phase,
        "alerts": alerts_list,
        "hints": status.hints,
        "software_health": {
            "mlflow_reachable": status.software_health.mlflow_reachable,
            "disk_free_gb": status.software_health.disk_free_gb,
            "process_alive": status.software_health.process_alive,
            "last_stdout_age_seconds": status.software_health.last_stdout_age_seconds,
        },
        "data_quality": {
            "schema_hash": status.data_quality.schema_hash,
            "files_modified": status.data_quality.files_modified,
        },
        "metrics_history": status.metrics_history,
        "budget_fraction_used": status.budget_fraction_used,
        "warning_triggered": status.warning_triggered,
        "convergence_signal": status.convergence_signal,
        "runs_per_iteration": status.runs_per_iteration,
        "timestamp": status.timestamp,
    }


def _dict_to_status(data: dict[str, Any]) -> BudgetStatusV21:
    """Конвертирует словарь из JSON в BudgetStatusV21.

    Args:
        data: словарь из JSON.

    Returns:
        Десериализованная структура статуса.
    """
    alerts = [
        AlertEntry(
            code=a["code"],
            level=a["level"],
            message=a["message"],
            timestamp=a.get("timestamp", 0.0),
        )
        for a in data.get("alerts", [])
    ]
    sw = data.get("software_health", {})
    dq = data.get("data_quality", {})
    return BudgetStatusV21(
        version=data.get("version", "2.1"),
        session_id=data.get("session_id", ""),
        iterations_used=data.get("iterations_used", 0),
        iterations_limit=data.get("iterations_limit"),
        time_elapsed=data.get("time_elapsed", 0.0),
        time_limit=data.get("time_limit"),
        hard_stop=data.get("hard_stop", False),
        hard_stop_reason=data.get("hard_stop_reason"),
        phase=data.get("phase", "monitoring"),
        alerts=alerts,
        hints=data.get("hints", []),
        software_health=SoftwareHealth(
            mlflow_reachable=sw.get("mlflow_reachable", True),
            disk_free_gb=sw.get("disk_free_gb", 0.0),
            process_alive=sw.get("process_alive", True),
            last_stdout_age_seconds=sw.get("last_stdout_age_seconds", 0.0),
        ),
        data_quality=DataQuality(
            schema_hash=dq.get("schema_hash", ""),
            files_modified=dq.get("files_modified", False),
        ),
        metrics_history=data.get("metrics_history", []),
        budget_fraction_used=data.get("budget_fraction_used", 0.0),
        warning_triggered=data.get("warning_triggered", False),
        convergence_signal=data.get("convergence_signal", 0.0),
        runs_per_iteration=data.get("runs_per_iteration", 1),
        timestamp=data.get("timestamp", 0.0),
    )
