"""ClaudeCodeRunner — управление subprocess Claude Code для UAF-сессии."""

import json
import logging
import os
import signal
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import IO

from uaf import MLFLOW_DEFAULT_URI

logger = logging.getLogger(__name__)

# Начальный промпт для Claude Code
_INITIAL_PROMPT = "Read program.md and start the research session"

# Deny list команд (соответствует дизайн-документу)
_DENY_LIST: list[str] = [
    "Bash(rm -rf:*)",
    "Bash(curl:*)",
    "Bash(wget:*)",
    "Bash(ssh:*)",
    "Bash(sudo:*)",
    "Bash(git push:*)",
    "Bash(git push --force:*)",
]

# Allow list команд
_ALLOW_LIST: list[str] = [
    "Bash(*)",
    "Read(*)",
    "Edit(*)",
    "Write(*)",
    "Glob(*)",
    "Grep(*)",
]


class ClaudeCodeRunner:
    """Запускает Claude Code как subprocess и управляет его жизненным циклом.

    Генерирует settings.json с ограничениями (allowedTools, deny list),
    запускает claude subprocess, мониторит PID, сохраняет PID в файл,
    передаёт SIGTERM при получении сигнала остановки.

    Атрибуты:
        session_dir: директория сессии (SESSION_DIR).
        session_id: идентификатор сессии.
        claude_model: модель Claude Code (например, claude-opus-4).
        mlflow_tracking_uri: URI MLflow сервера.
        budget_status_file: путь к budget_status.json.
        stdout_callback: функция вызываемая при каждой строке stdout.
        timeout_seconds: максимальное время работы subprocess.
        fully_autonomous: если True — добавляет dangerouslyAllowAutoApprove.
    """

    def __init__(
        self,
        session_dir: Path,
        session_id: str,
        claude_model: str = "claude-opus-4",
        mlflow_tracking_uri: str = MLFLOW_DEFAULT_URI,
        mlflow_experiment_name: str | None = None,
        budget_status_file: Path | None = None,
        stdout_callback: Callable[[str], None] | None = None,
        timeout_seconds: float | None = None,
        fully_autonomous: bool = False,
        on_start: Callable[[int], None] | None = None,
        max_turns: int | None = None,
        system_prompt_path: Path | None = None,
    ) -> None:
        self.session_dir = session_dir
        self.session_id = session_id
        self.claude_model = claude_model
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.mlflow_experiment_name = mlflow_experiment_name or f"uaf/{session_id}"
        self.budget_status_file = budget_status_file or session_dir.parent / "budget_status.json"
        self.stdout_callback = stdout_callback
        self.timeout_seconds = timeout_seconds
        self.fully_autonomous = fully_autonomous
        self.on_start = on_start
        self.max_turns = max_turns
        self.system_prompt_path = system_prompt_path

        self._process: subprocess.Popen | None = None  # type: ignore[type-arg]
        self._pid_file = session_dir / "claude_pid.txt"
        self._stdout_thread: threading.Thread | None = None

    def generate_settings_json(self) -> Path:
        """Генерирует .claude/settings.json с ограничениями для Claude Code.

        Создаёт директорию .claude/ в session_dir если не существует.
        Write permissions ограничены SESSION_DIR/**.

        Returns:
            Путь к сгенерированному settings.json.
        """
        session_relative = str(self.session_dir)

        allow_list = _ALLOW_LIST.copy()
        # Разрешаем запись только в директорию сессии и mlruns
        allow_list.append(f"Write({session_relative}/**)")
        # mlruns может быть в .uaf/
        uaf_dir = self.session_dir.parent
        allow_list.append(f"Write({uaf_dir}/mlruns/**)")

        settings: dict = {
            "permissions": {
                "allow": allow_list,
                "deny": _DENY_LIST,
            },
            "env": {
                "MLFLOW_TRACKING_URI": self.mlflow_tracking_uri,
                "MLFLOW_EXPERIMENT_NAME": self.mlflow_experiment_name,
                "UAF_SESSION_ID": self.session_id,
                "UAF_BUDGET_STATUS_FILE": str(self.budget_status_file),
                "UAF_SESSION_DIR": str(self.session_dir),
            },
        }

        if self.fully_autonomous:
            settings["dangerouslyAllowAutoApprove"] = True

        claude_dir = self.session_dir / ".claude"
        claude_dir.mkdir(parents=True, exist_ok=True)
        settings_path = claude_dir / "settings.json"
        settings_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
        logger.info("Сгенерирован settings.json: %s", settings_path)
        return settings_path

    def run(self) -> int:
        """Запускает Claude Code subprocess и ждёт завершения.

        Порядок:
        1. Генерирует settings.json
        2. Запускает claude subprocess с начальным промптом
        3. Сохраняет PID в claude_pid.txt
        4. Читает stdout в отдельном потоке (для обновления last_stdout_time)
        5. Ждёт завершения (с таймаутом если задан)

        Returns:
            Код возврата subprocess (0 — успех).

        Raises:
            FileNotFoundError: если claude CLI не найден в PATH.
        """
        self.generate_settings_json()

        cmd = self._build_command()
        env = self._build_env()

        logger.info(
            "Запуск Claude Code: cmd=%s, cwd=%s",
            " ".join(cmd[:3]) + "...",
            self.session_dir,
        )

        self._process = subprocess.Popen(
            cmd,
            cwd=str(self.session_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        pid = self._process.pid
        self._pid_file.write_text(str(pid), encoding="utf-8")
        logger.info("Claude Code запущен: PID=%d, pid_file=%s", pid, self._pid_file)

        if callable(self.on_start):
            self.on_start(pid)

        # Читаем stdout в отдельном потоке
        self._stdout_thread = threading.Thread(
            target=self._read_stdout,
            args=(self._process.stdout,),
            daemon=True,
            name=f"claude-stdout-{self.session_id[:8]}",
        )
        self._stdout_thread.start()

        try:
            return_code = self._process.wait(timeout=self.timeout_seconds)
        except subprocess.TimeoutExpired:
            logger.error(
                "Таймаут Claude Code (%s сек). Отправка SIGTERM.", self.timeout_seconds
            )
            self.send_sigterm()
            try:
                return_code = self._process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                logger.error("Claude Code не ответил на SIGTERM, SIGKILL.")
                self._process.kill()
                return_code = self._process.wait()

        self._stdout_thread.join(timeout=5)
        logger.info("Claude Code завершился: return_code=%d", return_code)

        if self._pid_file.exists():
            self._pid_file.unlink()

        return return_code

    def send_sigterm(self) -> None:
        """Отправляет SIGTERM процессу Claude Code.

        Используется BudgetController при hard_stop (grace period истёк).
        Если процесс уже завершился — логирует и возвращает.
        """
        if self._process is None:
            logger.warning("send_sigterm: subprocess не запущен")
            return
        if self._process.poll() is not None:
            logger.info("send_sigterm: процесс уже завершился")
            return
        try:
            self._process.send_signal(signal.SIGTERM)
            logger.warning("SIGTERM отправлен Claude Code (PID=%d)", self._process.pid)
        except (ProcessLookupError, OSError) as exc:
            logger.warning("Не удалось отправить SIGTERM: %s", exc)

    @property
    def pid(self) -> int | None:
        """PID запущенного subprocess или None."""
        if self._process is None:
            return None
        return self._process.pid

    def is_alive(self) -> bool:
        """Проверяет жив ли subprocess.

        Returns:
            True если subprocess запущен и ещё не завершился.
        """
        if self._process is None:
            return False
        return self._process.poll() is None

    def _build_command(self) -> list[str]:
        """Строит команду запуска Claude Code.

        Returns:
            Список аргументов команды.
        """
        # Маппинг коротких алиасов в полные имена моделей
        model_aliases: dict[str, str] = {
            "claude-opus-4": "claude-opus-4-6",
            "opus": "claude-opus-4-6",
            "sonnet": "claude-sonnet-4-6",
            "haiku": "claude-haiku-4-5-20251001",
        }
        model = model_aliases.get(self.claude_model, self.claude_model)
        cmd = [
            "claude",
            "--model",
            model,
            "--output-format",
            "json",
        ]
        if self.max_turns is not None:
            cmd.extend(["--max-turns", str(self.max_turns)])
        if self.system_prompt_path and self.system_prompt_path.exists():
            prompt = self.system_prompt_path.read_text(encoding="utf-8")
            cmd.extend(["--system-prompt", prompt])
        cmd.extend(["--dangerously-skip-permissions", "-p", _INITIAL_PROMPT])
        return cmd

    def _build_env(self) -> dict[str, str]:
        """Строит окружение для subprocess.

        Наследует текущее окружение и добавляет UAF-специфичные переменные.

        Returns:
            Словарь переменных окружения.
        """
        env = os.environ.copy()
        env.update(
            {
                "MLFLOW_TRACKING_URI": self.mlflow_tracking_uri,
                "MLFLOW_EXPERIMENT_NAME": self.mlflow_experiment_name,
                "UAF_SESSION_ID": self.session_id,
                "UAF_BUDGET_STATUS_FILE": str(self.budget_status_file),
                "UAF_SESSION_DIR": str(self.session_dir),
            }
        )
        return env

    def _read_stdout(self, stdout: IO[str]) -> None:
        """Читает stdout subprocess построчно.

        Вызывает stdout_callback если задан, иначе логирует строки.
        Обновляет внутренний счётчик времени для обнаружения зависания.

        Args:
            stdout: файловый объект stdout subprocess.
        """
        session_log = self.session_dir / "session.log"
        try:
            with session_log.open("a", encoding="utf-8") as log_fh:
                for line in stdout:
                    line = line.rstrip("\n")
                    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
                    log_fh.write(f"{timestamp} INFO [claude] {line}\n")
                    log_fh.flush()

                    if self.stdout_callback is not None:
                        try:
                            self.stdout_callback(line)
                        except Exception as exc:
                            logger.debug("stdout_callback ошибка: %s", exc)
                    else:
                        logger.debug("[claude] %s", line)
        except Exception as exc:
            logger.error("Ошибка чтения stdout Claude Code: %s", exc)

    @staticmethod
    def read_pid_from_file(session_dir: Path) -> int | None:
        """Читает PID из claude_pid.txt если файл существует.

        Args:
            session_dir: директория сессии.

        Returns:
            PID или None если файл не найден или некорректен.
        """
        pid_file = session_dir / "claude_pid.txt"
        if not pid_file.exists():
            return None
        try:
            return int(pid_file.read_text().strip())
        except (ValueError, OSError):
            return None
