"""CLI точка входа UAF: uaf run / resume / status / report / health / analyze."""

import json
import logging
import sys
from pathlib import Path

import click

logger = logging.getLogger(__name__)


def _setup_logging(verbose: bool) -> None:
    """Настраивает logging для CLI.

    Args:
        verbose: включить DEBUG уровень.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Включить DEBUG логирование")
@click.pass_context
def main(ctx: click.Context, verbose: bool) -> None:
    """Universal AutoResearch Framework — автоматизация ML-экспериментов через Claude Code."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    _setup_logging(verbose)


@main.command("run")
@click.option(
    "--task",
    "task_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Путь к task.yaml",
)
@click.option(
    "--budget",
    "budget_path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Путь к budget.yaml",
)
@click.option(
    "--session-id",
    default=None,
    help="ID сессии (автогенерируется если не задан)",
)
@click.option("--budget-iterations", type=int, default=None, help="Максимум итераций")
@click.option("--time", "time_hours", type=float, default=None, help="Максимум часов")
@click.option("--autonomous", is_flag=True, help="Пропустить HumanOversightGate")
@click.option("--model", default="claude-opus-4", show_default=True, help="Модель Claude Code")
@click.option(
    "--work-dir",
    type=click.Path(path_type=Path),
    default=Path("."),
    help="Рабочая директория",
)
@click.option("--prev-session", default=None, help="ID предыдущей сессии для передачи контекста")
@click.pass_context
def run_cmd(
    ctx: click.Context,
    task_path: Path,
    budget_path: Path | None,
    session_id: str | None,
    budget_iterations: int | None,
    time_hours: float | None,
    autonomous: bool,
    model: str,
    work_dir: Path,
    prev_session: str | None,
) -> None:
    """Запустить UAF сессию исследования.

    Читает task.yaml и запускает полный цикл:
    setup -> data loading -> planning -> approval -> Claude Code -> analysis -> report.

    Примеры:

    \b
    uaf run --task task.yaml --budget 50 --time 3600
    uaf run --task task.yaml --session-id mar20 --autonomous
    """
    from uaf.core.session_controller import ResearchSessionController

    logger.info(
        "UAF run: task=%s, model=%s, autonomous=%s", task_path, model, autonomous
    )

    try:
        controller = ResearchSessionController(
            work_dir=work_dir.resolve(),
            task_path=task_path.resolve(),
            budget_path=budget_path.resolve() if budget_path else None,
            session_id=session_id,
            claude_model=model,
            fully_autonomous=autonomous,
            prev_session_id=prev_session,
        )

        # Переопределяем параметры бюджета если переданы через CLI
        if budget_iterations is not None or time_hours is not None:
            override: dict = {}
            if budget_iterations:
                override["max_iterations"] = budget_iterations
            if time_hours:
                override["max_time_hours"] = time_hours
            controller._budget_config.setdefault("budget", {}).update(override)

        click.echo(f"Запуск сессии: {controller.session_id}")
        click.echo(f"Session dir: {controller.session_dir}")

        success = controller.run()
        sys.exit(0 if success else 1)

    except FileNotFoundError as exc:
        click.echo(f"Ошибка: {exc}", err=True)
        sys.exit(1)
    except Exception as exc:
        logger.error("Ошибка UAF run: %s", exc, exc_info=True)
        click.echo(f"Ошибка: {exc}", err=True)
        sys.exit(1)


@main.command("resume")
@click.option(
    "--session",
    "session_id",
    required=True,
    help="ID сессии для возобновления",
)
@click.option(
    "--work-dir",
    type=click.Path(path_type=Path),
    default=Path("."),
    help="Рабочая директория",
)
@click.pass_context
def resume_cmd(ctx: click.Context, session_id: str, work_dir: Path) -> None:
    """Возобновить прерванную сессию.

    Читает session_state.json и продолжает с последнего состояния.

    Примеры:

    \b
    uaf resume --session 20260320-143022-a1b2c3d4
    """
    from uaf.core.session_controller import ResearchSessionController

    try:
        controller = ResearchSessionController.from_resume(
            work_dir=work_dir.resolve(),
            session_id=session_id,
        )
        click.echo(f"Возобновление сессии {session_id} из состояния: {controller.state}")
        success = controller.run()
        sys.exit(0 if success else 1)

    except FileNotFoundError as exc:
        click.echo(f"Сессия не найдена: {exc}", err=True)
        sys.exit(1)
    except Exception as exc:
        logger.error("Ошибка resume: %s", exc, exc_info=True)
        click.echo(f"Ошибка: {exc}", err=True)
        sys.exit(1)


@main.command("status")
@click.option("--session", "session_id", default=None, help="ID сессии")
@click.option(
    "--work-dir",
    type=click.Path(path_type=Path),
    default=Path("."),
    help="Рабочая директория",
)
def status_cmd(session_id: str | None, work_dir: Path) -> None:
    """Показать статус бюджета и сессии.

    Читает budget_status.json и session_state.json, выводит в виде Rich таблицы.
    """
    from uaf.budget.status_file import read_budget_status

    uaf_dir = work_dir / ".uaf"
    budget_file = uaf_dir / "budget_status.json"
    status = read_budget_status(budget_file)

    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
    except ImportError:
        # Fallback без rich
        _status_plain(status, session_id, uaf_dir)
        return

    if status is None:
        # Попытаемся найти сессию по ID
        if session_id:
            state_file = uaf_dir / "sessions" / session_id / "session_state.json"
            if state_file.exists():
                _show_session_state(console, state_file)
                return
        console.print("[yellow]Активных сессий не найдено[/yellow]")
        return

    sid = session_id or status.session_id
    table = Table(title=f"UAF Session: {sid[:20]}...", show_header=True)
    table.add_column("Параметр", style="bold")
    table.add_column("Значение")

    table.add_row("State", "EXECUTING")
    table.add_row("Phase", status.phase)
    table.add_row(
        "Итерации",
        f"{status.iterations_used}/{status.iterations_limit or '?'}",
    )
    time_str = f"{status.time_elapsed:.0f}s"
    if status.time_limit:
        time_str += f" / {status.time_limit:.0f}s"
    table.add_row("Время", time_str)
    table.add_row(
        "Бюджет",
        f"{status.budget_fraction_used * 100:.1f}%",
    )
    table.add_row(
        "Hard stop",
        "[red]ДА[/red]" if status.hard_stop else "[green]нет[/green]",
    )
    if status.hard_stop_reason:
        table.add_row("Hard stop причина", status.hard_stop_reason)
    table.add_row(
        "Warning triggered",
        "[yellow]ДА[/yellow]" if status.warning_triggered else "нет",
    )
    if status.metrics_history:
        table.add_row(
            "Метрика (последняя)",
            f"{status.metrics_history[-1]:.6f}",
        )
    table.add_row(
        "MLflow доступен",
        "[green]да[/green]"
        if status.software_health.mlflow_reachable
        else "[red]нет[/red]",
    )
    table.add_row(
        "Диск свободно",
        f"{status.software_health.disk_free_gb:.1f} ГБ",
    )
    console.print(table)

    if status.alerts:
        console.print(f"\n[bold]Алерты ({len(status.alerts)}):[/bold]")
        for alert in status.alerts[-10:]:
            if alert.level == "CRITICAL":
                color = "red"
            elif alert.level == "WARNING":
                color = "yellow"
            else:
                color = "blue"
            console.print(f"  [{color}][{alert.level}] {alert.code}[/{color}]: {alert.message}")

    if status.hints:
        console.print("\n[bold]Подсказки:[/bold]")
        for hint in status.hints:
            console.print(f"  {hint}")


@main.command("report")
@click.option(
    "--session",
    "session_id",
    required=True,
    help="ID сессии",
)
@click.option(
    "--work-dir",
    type=click.Path(path_type=Path),
    default=Path("."),
    help="Рабочая директория",
)
def report_cmd(session_id: str, work_dir: Path) -> None:
    """Сгенерировать (или пересгенерировать) PDF отчёт для сессии.

    Читает данные из MLflow и session_analysis.json, компилирует PDF.
    """
    uaf_dir = work_dir / ".uaf"
    session_dir = uaf_dir / "sessions" / session_id
    state_file = session_dir / "session_state.json"

    if not session_dir.exists():
        click.echo(f"Сессия не найдена: {session_dir}", err=True)
        sys.exit(1)

    # Читаем состояние для конфигурации
    task_config: dict = {}
    tracking_uri = "http://127.0.0.1:5000"
    experiment_id = "0"
    claude_model = "claude-opus-4"

    if state_file.exists():
        try:
            state = json.loads(state_file.read_text(encoding="utf-8"))
            tracking_uri = state.get("mlflow_tracking_uri", tracking_uri)
            experiment_id = state.get("mlflow_experiment_id", experiment_id) or experiment_id
            claude_model = state.get("claude_model", claude_model)

            task_path_str = state.get("task_path")
            if task_path_str:
                import yaml

                task_path = Path(task_path_str)
                if task_path.exists():
                    with task_path.open(encoding="utf-8") as fh:
                        task_config = yaml.safe_load(fh) or {}
        except Exception as exc:
            logger.warning("Не удалось прочитать session_state.json: %s", exc)

    task = task_config.get("task", {})
    task_config_flat = {
        "title": task.get("title", f"Session {session_id}"),
        "task_type": task.get("type", "---"),
        "target_metric": task.get("metric", {}).get("name", "metric"),
        "metric_direction": task.get("metric", {}).get("direction", "maximize"),
        "target_column": task.get("dataset", {}).get("target_column", "---"),
        "dataset_path": task.get("dataset", {}).get("train_path", "---"),
        "problem_statement": task.get("problem_statement", ""),
        "claude_model": claude_model,
    }

    from uaf.reporting.report_generator import ReportGenerator

    generator = ReportGenerator(
        session_dir=session_dir,
        session_id=session_id,
        tracking_uri=tracking_uri,
        experiment_id=experiment_id,
        task_config=task_config_flat,
    )

    click.echo(f"Генерация отчёта для сессии {session_id}...")
    result_path = generator.compile_from_sections()

    if result_path:
        click.echo(f"Отчёт: {result_path}")
    else:
        click.echo("Не удалось создать отчёт", err=True)
        sys.exit(1)


@main.command("health")
@click.option(
    "--work-dir",
    type=click.Path(path_type=Path),
    default=Path("."),
    help="Рабочая директория",
)
def health_cmd(work_dir: Path) -> None:
    """Показать историю здоровья UAF сессий из health_history.jsonl.

    Выводит таблицу всех сессий с ключевыми KPI метриками.
    """
    uaf_dir = work_dir / ".uaf"
    health_file = uaf_dir / "health_history.jsonl"

    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
    except ImportError:
        click.echo("rich не установлен, используй: pip install rich")
        return

    if not health_file.exists():
        console.print("[yellow]health_history.jsonl не найден. Нет истории сессий.[/yellow]")
        console.print(f"Ожидается: {health_file}")
        return

    entries = []
    for line in health_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    if not entries:
        console.print("[yellow]health_history.jsonl пуст[/yellow]")
        return

    table = Table(title="UAF Health History", show_header=True)
    table.add_column("Session ID", style="bold", max_width=24)
    table.add_column("Date")
    table.add_column("State")
    table.add_column("Runs")
    table.add_column("Clean rate")
    table.add_column("Health")
    table.add_column("Budget %")

    for entry in entries[-20:]:  # последние 20 сессий
        sid = entry.get("session_id", "---")[:20]
        date = entry.get("date", "---")
        state = entry.get("final_state", "---")
        total_runs = str(entry.get("total_runs", "---"))
        clean_rate = entry.get("clean_rate")
        clean_str = f"{clean_rate * 100:.0f}%" if clean_rate is not None else "---"
        health = entry.get("overall_health", "---")
        budget_pct = entry.get("budget_fraction_used")
        budget_str = f"{budget_pct * 100:.0f}%" if budget_pct is not None else "---"

        state_color = "green" if state == "DONE" else "red" if state == "FAILED" else "yellow"
        health_color = (
            "red"
            if health == "has_critical"
            else "yellow" if health == "has_warnings" else "green"
        )

        table.add_row(
            sid,
            date,
            f"[{state_color}]{state}[/{state_color}]",
            total_runs,
            clean_str,
            f"[{health_color}]{health}[/{health_color}]",
            budget_str,
        )

    console.print(table)
    console.print(f"\n[dim]Файл: {health_file} ({len(entries)} записей)[/dim]")


@main.command("analyze")
@click.option(
    "--session",
    "session_id",
    required=True,
    help="ID сессии для анализа",
)
@click.option(
    "--work-dir",
    type=click.Path(path_type=Path),
    default=Path("."),
    help="Рабочая директория",
)
def analyze_cmd(session_id: str, work_dir: Path) -> None:
    """Запустить ResultAnalyzer и SystemErrorAnalyzer для указанной сессии.

    Читает MLflow данные и генерирует session_analysis.json и system_error_report.json.
    """
    uaf_dir = work_dir / ".uaf"
    session_dir = uaf_dir / "sessions" / session_id
    state_file = session_dir / "session_state.json"

    if not session_dir.exists():
        click.echo(f"Сессия не найдена: {session_dir}", err=True)
        sys.exit(1)

    tracking_uri = "http://127.0.0.1:5000"
    experiment_id = "0"
    target_metric = "metric"
    metric_direction = "maximize"

    if state_file.exists():
        try:
            import yaml

            state = json.loads(state_file.read_text(encoding="utf-8"))
            tracking_uri = state.get("mlflow_tracking_uri", tracking_uri)
            experiment_id = state.get("mlflow_experiment_id", experiment_id) or experiment_id

            task_path_str = state.get("task_path")
            if task_path_str:
                task_path = Path(task_path_str)
                if task_path.exists():
                    with task_path.open(encoding="utf-8") as fh:
                        task_cfg = yaml.safe_load(fh) or {}
                    task = task_cfg.get("task", {})
                    target_metric = task.get("metric", {}).get("name", "metric")
                    metric_direction = task.get("metric", {}).get("direction", "maximize")
        except Exception as exc:
            logger.warning("Не удалось прочитать конфигурацию: %s", exc)

    click.echo(f"Запуск анализа сессии {session_id}...")

    # ResultAnalyzer
    from uaf.analysis.result_analyzer import ResultAnalyzer

    analyzer = ResultAnalyzer(
        session_id=session_id,
        experiment_id=experiment_id,
        tracking_uri=tracking_uri,
        session_dir=session_dir,
        target_metric=target_metric,
        metric_direction=metric_direction,  # type: ignore[arg-type]
    )
    analysis = analyzer.analyze()
    click.echo(
        f"ResultAnalyzer: total={analysis.total_runs},"
        f" completed={analysis.completed_runs},"
        f" failed={analysis.failed_runs}"
    )
    click.echo(f"  session_analysis.json: {session_dir / 'session_analysis.json'}")

    # SystemErrorAnalyzer
    from uaf.analysis.system_error_analyzer import SystemErrorAnalyzer

    sys_analyzer = SystemErrorAnalyzer(
        session_id=session_id,
        session_dir=session_dir,
        experiment_id=experiment_id,
        tracking_uri=tracking_uri,
    )
    sys_report = sys_analyzer.analyze()
    click.echo(f"SystemErrorAnalyzer: health={sys_report.overall_health}")
    click.echo(f"  system_error_report.json: {session_dir / 'system_error_report.json'}")

    if analysis.hypotheses:
        click.echo(f"\nГипотезы ({len(analysis.hypotheses)}):")
        for h in analysis.hypotheses:
            click.echo(f"  [{h.code}] P{h.priority}: {h.description}")


@main.command("stop")
@click.argument("session_id", required=False)
@click.option(
    "--work-dir",
    type=click.Path(path_type=Path),
    default=Path("."),
    help="Рабочая директория",
)
def stop_cmd(session_id: str | None, work_dir: Path) -> None:
    """Graceful stop текущей сессии (устанавливает hard_stop=true)."""
    from uaf.budget.status_file import read_budget_status, write_budget_status

    budget_file = work_dir / ".uaf" / "budget_status.json"
    status = read_budget_status(budget_file)
    if status is None:
        click.echo("Активных сессий не найдено")
        return

    status.hard_stop = True
    status.hard_stop_reason = "manual_stop"
    status.phase = "grace_period"
    write_budget_status(status, budget_file)
    click.echo(f"Hard stop установлен для сессии {status.session_id}")


@main.command("cleanup")
@click.option(
    "--sessions-older-than",
    default="7d",
    show_default=True,
    help="Удалить сессии старше N дней (например: 7d, 30d)",
)
@click.option(
    "--work-dir",
    type=click.Path(path_type=Path),
    default=Path("."),
    help="Рабочая директория",
)
def cleanup_cmd(sessions_older_than: str, work_dir: Path) -> None:
    """Очистить старые сессии из .uaf/sessions/."""
    import time as _time

    uaf_dir = work_dir / ".uaf"
    sessions_dir = uaf_dir / "sessions"

    if not sessions_dir.exists():
        click.echo("Нет сессий для очистки")
        return

    # Парсим период
    days = int(sessions_older_than.rstrip("d"))
    cutoff = _time.time() - days * 86400

    removed = 0
    for session_path in sessions_dir.iterdir():
        if not session_path.is_dir():
            continue
        state_file = session_path / "session_state.json"
        if not state_file.exists():
            continue
        if state_file.stat().st_mtime < cutoff:
            import shutil

            shutil.rmtree(session_path)
            removed += 1
            logger.info("Удалена сессия: %s", session_path.name)

    click.echo(f"Удалено {removed} сессий старше {sessions_older_than}")


def _status_plain(status: object, session_id: str | None, uaf_dir: Path) -> None:
    """Fallback вывод статуса без rich.

    Args:
        status: BudgetStatusV21 или None.
        session_id: ID сессии.
        uaf_dir: директория .uaf/.
    """
    if status is None:
        click.echo("Активных сессий не найдено")
        return
    sid = session_id or getattr(status, "session_id", "---")
    click.echo(f"Сессия: {sid}")
    click.echo(
        f"  Итерации: {getattr(status, 'iterations_used', '?')}"
        f"/{getattr(status, 'iterations_limit', '?')}"
    )
    click.echo(f"  Hard stop: {getattr(status, 'hard_stop', False)}")
    click.echo(f"  Phase: {getattr(status, 'phase', '---')}")
    alerts = getattr(status, "alerts", [])
    if alerts:
        click.echo(f"  Алерты ({len(alerts)}):")
        for alert in alerts[-5:]:
            click.echo(f"    [{alert.level}] {alert.code}: {alert.message}")


def _show_session_state(console: object, state_file: Path) -> None:
    """Показывает session_state.json в виде таблицы.

    Args:
        console: rich Console.
        state_file: путь к session_state.json.
    """
    try:
        from rich.table import Table

        state = json.loads(state_file.read_text(encoding="utf-8"))
        table = Table(title=f"Session: {state.get('session_id', '---')[:20]}")
        table.add_column("Параметр")
        table.add_column("Значение")
        for k, v in state.items():
            if k not in ("approval_result",):
                table.add_row(str(k), str(v))
        console.print(table)  # type: ignore[union-attr]
    except Exception as exc:
        click.echo(f"Ошибка чтения session_state.json: {exc}", err=True)
