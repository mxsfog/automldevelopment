"""HumanOversightGate — интерактивный approval checkpoint перед запуском Claude Code."""

import logging
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

# Режим одобрения
ApprovalMode = Literal["standard", "fully_autonomous"]

# Максимальное число раундов редактирования
_MAX_EDIT_ROUNDS = 5

# Таймаут ожидания одобрения: 24 часа
_DEFAULT_TIMEOUT_HOURS = 24.0

# Обязательные секции program.md
_REQUIRED_SECTIONS = [
    "## Metadata",
    "## Task Description",
    "## Research Phases",
    "## Execution Instructions",
]


@dataclass
class ApprovalResult:
    """Результат прохождения HumanOversightGate.

    Атрибуты:
        approved: одобрено ли.
        modified: был ли редактирован program.md.
        approval_mode: режим (standard/fully_autonomous).
        wait_time_seconds: время ожидания в секундах.
        edit_rounds: количество раундов редактирования.
        notes: опциональные заметки.
    """

    approved: bool
    modified: bool
    approval_mode: str
    wait_time_seconds: float
    edit_rounds: int = 0
    notes: str | None = None


def _validate_program_md(program_md_path: Path) -> list[str]:
    """Проверяет наличие обязательных секций в program.md.

    Args:
        program_md_path: путь к program.md.

    Returns:
        Список отсутствующих секций. Пустой список — всё в порядке.
    """
    if not program_md_path.exists():
        return _REQUIRED_SECTIONS.copy()
    content = program_md_path.read_text(encoding="utf-8")
    return [sec for sec in _REQUIRED_SECTIONS if sec not in content]


def _open_in_editor(path: Path) -> None:
    """Открывает файл в $EDITOR или выводит содержимое в терминал.

    Args:
        path: путь к файлу.
    """
    editor = os.environ.get("EDITOR", "")
    if editor:
        try:
            subprocess.run([editor, str(path)], check=True)
            return
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("Не удалось открыть editor=%s, выводим содержимое в терминал", editor)

    # Fallback: вывод содержимого в терминал
    content = path.read_text(encoding="utf-8")
    separator = "-" * 80
    print(f"\n{separator}")
    print(f"  {path}")
    print(separator)
    print(content)
    print(separator)
    print("(Отредактируй файл вручную в другом терминале, затем нажми Enter)")
    input()


def _show_program_summary(program_md_path: Path, adversarial_auc: float | None) -> None:
    """Выводит краткое summary program.md перед запросом одобрения.

    Args:
        program_md_path: путь к program.md.
        adversarial_auc: AUC adversarial validation (если есть).
    """
    content = program_md_path.read_text(encoding="utf-8")
    lines = content.split("\n")

    # Считаем шаги Phase (#### Step)
    phase_count = sum(1 for l in lines if l.startswith("### Phase "))
    step_count = sum(1 for l in lines if l.startswith("#### Step "))

    print("\n" + "=" * 70)
    print("  UAF: Review Research Program")
    print("=" * 70)
    print(f"  Файл: {program_md_path}")
    print(f"  Фазы: {phase_count}  |  Шаги: {step_count}")

    if adversarial_auc is not None:
        if adversarial_auc >= 0.85:
            print(f"\n  [!] ПРЕДУПРЕЖДЕНИЕ: AdversarialValidation AUC={adversarial_auc:.3f} >= 0.85")
            print("      Train/val распределения СИЛЬНО отличаются.")
            print("      Требуется явное подтверждение (y) для продолжения.")
        elif adversarial_auc >= 0.6:
            print(f"\n  [!] Внимание: AdversarialValidation AUC={adversarial_auc:.3f} (умеренное отличие)")
    print("=" * 70)

    # Выводим program.md
    try:
        # Пробуем rich если доступен
        from rich.console import Console
        from rich.markdown import Markdown

        console = Console()
        console.print(Markdown(content))
    except ImportError:
        print(content)


class HumanOversightGate:
    """Интерактивный approval checkpoint.

    Открывает program.md, ждёт ввода y/n/e от пользователя.
    При critical adversarial validation — требует явного y.
    Логирует решение (approved/rejected) и время ожидания.

    Args:
        program_md_path: путь к program.md.
        approval_mode: standard или fully_autonomous.
        timeout_hours: таймаут ожидания (default 24ч).
        adversarial_auc: AUC из AdversarialValidation.
    """

    def __init__(
        self,
        program_md_path: Path,
        approval_mode: ApprovalMode = "standard",
        timeout_hours: float = _DEFAULT_TIMEOUT_HOURS,
        adversarial_auc: float | None = None,
    ) -> None:
        self.program_md_path = program_md_path
        self.approval_mode = approval_mode
        self.timeout_hours = timeout_hours
        self.adversarial_auc = adversarial_auc

    def check(self) -> ApprovalResult:
        """Запускает процедуру одобрения.

        В fully_autonomous режиме возвращает ApprovalResult(approved=True) без ввода.

        Returns:
            ApprovalResult с результатом.
        """
        start_time = time.time()

        if self.approval_mode == "fully_autonomous":
            logger.info("fully_autonomous режим: пропуск HumanOversightGate")
            return ApprovalResult(
                approved=True,
                modified=False,
                approval_mode="fully_autonomous",
                wait_time_seconds=0.0,
                notes="Auto-approved (fully_autonomous mode)",
            )

        # Standard mode
        return self._interactive_check(start_time)

    def _interactive_check(self, start_time: float) -> ApprovalResult:
        """Интерактивная процедура проверки.

        Args:
            start_time: время начала ожидания.

        Returns:
            ApprovalResult.
        """
        edit_rounds = 0
        modified = False
        timeout_seconds = self.timeout_hours * 3600

        while edit_rounds <= _MAX_EDIT_ROUNDS:
            # Проверяем таймаут
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                logger.warning(
                    "HumanOversightGate: таймаут %.1f ч. Автоматический reject.", self.timeout_hours
                )
                return ApprovalResult(
                    approved=False,
                    modified=modified,
                    approval_mode="standard",
                    wait_time_seconds=elapsed,
                    edit_rounds=edit_rounds,
                    notes=f"Auto-rejected: timeout {self.timeout_hours}h",
                )

            # Показываем program.md
            _show_program_summary(self.program_md_path, self.adversarial_auc)

            # Валидация структуры
            missing_sections = _validate_program_md(self.program_md_path)
            if missing_sections:
                print(f"\n  [!] Отсутствуют обязательные секции: {missing_sections}")
                print("  Исправь program.md перед одобрением.")

            # Запрос ввода
            prompt = self._build_prompt()
            try:
                answer = input(prompt).strip().lower()
            except (EOFError, KeyboardInterrupt):
                # Нет интерактивного терминала — отклоняем
                wait = time.time() - start_time
                logger.warning("HumanOversightGate: прерван (EOF/KeyboardInterrupt)")
                return ApprovalResult(
                    approved=False,
                    modified=modified,
                    approval_mode="standard",
                    wait_time_seconds=wait,
                    edit_rounds=edit_rounds,
                    notes="Rejected: interrupted",
                )

            if answer == "y":
                # Дополнительное подтверждение при critical adversarial AUC
                if self.adversarial_auc is not None and self.adversarial_auc >= 0.85:
                    print(
                        "\n  [!] AdversarialValidation AUC >= 0.85."
                        " Введи 'yes' для явного подтверждения риска:"
                    )
                    try:
                        confirm = input("  > ").strip().lower()
                    except (EOFError, KeyboardInterrupt):
                        confirm = ""
                    if confirm != "yes":
                        print("  Подтверждение не получено. Вернись к y/n/e.")
                        continue

                wait = time.time() - start_time
                logger.info("HumanOversightGate: одобрено за %.1f сек", wait)
                self._update_metadata_approved()
                return ApprovalResult(
                    approved=True,
                    modified=modified,
                    approval_mode="standard",
                    wait_time_seconds=wait,
                    edit_rounds=edit_rounds,
                )

            elif answer == "n":
                wait = time.time() - start_time
                logger.info("HumanOversightGate: отклонено за %.1f сек", wait)
                return ApprovalResult(
                    approved=False,
                    modified=modified,
                    approval_mode="standard",
                    wait_time_seconds=wait,
                    edit_rounds=edit_rounds,
                    notes="Rejected by user",
                )

            elif answer == "e":
                if edit_rounds >= _MAX_EDIT_ROUNDS:
                    print(f"\n  [!] Достигнут максимум раундов редактирования ({_MAX_EDIT_ROUNDS}).")
                    print("  Для продолжения введи y или n.")
                    continue
                _open_in_editor(self.program_md_path)
                missing_after = _validate_program_md(self.program_md_path)
                if missing_after:
                    print(f"\n  [!] После редактирования отсутствуют секции: {missing_after}")
                    print("  Исправь и нажми e снова или введи y/n.")
                else:
                    print("\n  Структура program.md корректна.")
                edit_rounds += 1
                modified = True

            else:
                print("  Введи y (одобрить), n (отклонить) или e (редактировать).")

        # Исчерпаны раунды редактирования — ждём финального y/n
        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                return ApprovalResult(
                    approved=False,
                    modified=modified,
                    approval_mode="standard",
                    wait_time_seconds=elapsed,
                    edit_rounds=edit_rounds,
                    notes=f"Auto-rejected: timeout after max edit rounds",
                )
            try:
                answer = input("\n  Только y или n (редактирование недоступно): ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                break
            if answer == "y":
                wait = time.time() - start_time
                self._update_metadata_approved()
                return ApprovalResult(
                    approved=True,
                    modified=modified,
                    approval_mode="standard",
                    wait_time_seconds=wait,
                    edit_rounds=edit_rounds,
                )
            if answer == "n":
                wait = time.time() - start_time
                return ApprovalResult(
                    approved=False,
                    modified=modified,
                    approval_mode="standard",
                    wait_time_seconds=wait,
                    edit_rounds=edit_rounds,
                    notes="Rejected by user",
                )

        # Выход без ответа
        return ApprovalResult(
            approved=False,
            modified=modified,
            approval_mode="standard",
            wait_time_seconds=time.time() - start_time,
            edit_rounds=edit_rounds,
            notes="Rejected: no input",
        )

    def _build_prompt(self) -> str:
        """Строит строку приглашения ввода.

        Returns:
            Строка prompt.
        """
        return "\n  [y] одобрить  [n] отклонить  [e] редактировать > "

    def _update_metadata_approved(self) -> None:
        """Обновляет секцию Metadata в program.md после одобрения."""
        if not self.program_md_path.exists():
            return
        from datetime import datetime, timezone

        approval_time = datetime.now(tz=timezone.utc).isoformat()
        content = self.program_md_path.read_text(encoding="utf-8")
        content = content.replace(
            "- approved_by: pending",
            "- approved_by: human",
        ).replace(
            "- approval_time: null",
            f"- approval_time: {approval_time}",
        )
        self.program_md_path.write_text(content, encoding="utf-8")
        logger.info("program.md metadata обновлён: approved_by=human, time=%s", approval_time)

    def log_to_mlflow(
        self,
        result: ApprovalResult,
        mlflow_run_id: str | None = None,
        tracking_uri: str | None = None,
    ) -> None:
        """Логирует результат одобрения в MLflow Planning Run.

        Args:
            result: результат ApprovalResult.
            mlflow_run_id: ID Planning Run для обновления.
            tracking_uri: MLflow tracking URI.
        """
        try:
            import mlflow

            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            client = mlflow.tracking.MlflowClient()
            if mlflow_run_id:
                client.set_tag(mlflow_run_id, "approval_status", "approved" if result.approved else "rejected")
                client.set_tag(mlflow_run_id, "approval_mode", result.approval_mode)
                client.log_metric(mlflow_run_id, "approval_wait_seconds", result.wait_time_seconds, step=0)
                if result.notes:
                    client.set_tag(mlflow_run_id, "approval_notes", result.notes[:250])
            logger.info(
                "Approval результат залогирован в MLflow run=%s: approved=%s",
                mlflow_run_id,
                result.approved,
            )
        except Exception as exc:
            logger.warning("Не удалось залогировать approval в MLflow: %s", exc)
