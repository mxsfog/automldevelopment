"""Инициализация DVC и автоматические коммиты артефактов для UAF-сессии."""

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

_DVC_SIZE_THRESHOLD_BYTES = 1 * 1024 * 1024  # 1 МБ


class DVCSetup:
    """Инициализирует DVC в рабочей директории и управляет автоматическими коммитами.

    Артефакты > 1 МБ добавляются в DVC, <= 1 МБ — коммитятся напрямую в git.

    Атрибуты:
        work_dir: рабочая директория пользователя (не SESSION_DIR).
        uaf_dir: путь к .uaf/ директории.
        session_id: идентификатор сессии.
    """

    def __init__(self, work_dir: Path, session_id: str) -> None:
        self.work_dir = work_dir
        self.uaf_dir = work_dir / ".uaf"
        self.session_id = session_id
        self._dvc_cache_dir = self.uaf_dir / "dvc-cache"

    def init(self) -> None:
        """Инициализирует DVC если ещё не инициализирован, настраивает local cache."""
        dvc_dir = self.work_dir / ".dvc"
        if not dvc_dir.exists():
            logger.info("Инициализация DVC в %s", self.work_dir)
            self._run(["dvc", "init"])
            self._setup_dvcignore()
            self._setup_gitignore()
        else:
            logger.info("DVC уже инициализирован в %s", self.work_dir)

        self._dvc_cache_dir.mkdir(parents=True, exist_ok=True)
        self._configure_cache()

    def _configure_cache(self) -> None:
        """Настраивает local cache для DVC."""
        try:
            self._run(["dvc", "cache", "dir", str(self._dvc_cache_dir)])
        except subprocess.CalledProcessError as exc:
            logger.warning("Не удалось настроить DVC cache: %s", exc)

    def _setup_dvcignore(self) -> None:
        """Создаёт .dvcignore с исключениями для эфемерных файлов."""
        dvcignore_path = self.work_dir / ".dvcignore"
        lines = [
            "# Автогенерировано UAF",
            ".uaf/mlflow.db",
            ".uaf/mlruns/",
            ".uaf/budget_status.json",
            ".uaf/mlflow_context.json",
            "**/__pycache__/",
            "**/*.pyc",
            "**/.venv/",
            "**/venv/",
        ]
        dvcignore_path.write_text("\n".join(lines) + "\n")

    def _setup_gitignore(self) -> None:
        """Добавляет эфемерные файлы в .gitignore."""
        gitignore_path = self.work_dir / ".gitignore"
        entries_to_add = [
            ".uaf/mlflow.db",
            ".uaf/mlruns/",
            ".uaf/budget_status.json",
            ".uaf/mlflow_context.json",
        ]
        existing = set()
        if gitignore_path.exists():
            existing = set(gitignore_path.read_text().splitlines())

        new_entries = [e for e in entries_to_add if e not in existing]
        if new_entries:
            with gitignore_path.open("a") as fh:
                fh.write("\n# UAF эфемерные файлы\n")
                fh.write("\n".join(new_entries) + "\n")

    def auto_commit_artifact(self, artifact_path: Path, mlflow_run_id: str | None = None) -> None:
        """Автоматически коммитит артефакт: dvc add если > 1 МБ, иначе git add.

        Args:
            artifact_path: путь к артефакту (файл или директория).
            mlflow_run_id: run_id MLflow для cross-referencing в commit message.
        """
        if not artifact_path.exists():
            logger.warning("Артефакт не найден, пропускаем: %s", artifact_path)
            return

        size = self._get_size(artifact_path)
        relative = artifact_path.relative_to(self.work_dir)
        run_ref = f" [mlflow_run_id: {mlflow_run_id}]" if mlflow_run_id else ""

        if size > _DVC_SIZE_THRESHOLD_BYTES:
            logger.info("Артефакт > 1 МБ, добавляем в DVC: %s (%.1f МБ)", relative, size / 1e6)
            self._dvc_add(artifact_path)
            commit_msg = f"session {self.session_id}: dvc add {relative}{run_ref}"
            self._git_commit([str(artifact_path) + ".dvc"], commit_msg)
        else:
            logger.info("Артефакт <= 1 МБ, добавляем в git: %s", relative)
            commit_msg = f"session {self.session_id}: add {relative}{run_ref}"
            self._git_commit([str(artifact_path)], commit_msg)

    def commit_program_md(self, program_md_path: Path, stage: str = "generated") -> None:
        """Коммитит program.md через DVC.

        Args:
            program_md_path: путь к program.md.
            stage: стадия (generated/approved).
        """
        self._dvc_add(program_md_path)
        commit_msg = f"session {self.session_id}: program.md {stage}"
        self._git_commit([str(program_md_path) + ".dvc"], commit_msg)

    def commit_session_step(self, step_dir: Path, step_id: str, mlflow_run_id: str) -> None:
        """Коммитит результаты завершённого шага эксперимента.

        Args:
            step_dir: директория шага (experiments/{step_id}/).
            step_id: идентификатор шага.
            mlflow_run_id: run_id для cross-referencing.
        """
        size = self._get_size(step_dir)
        if size > _DVC_SIZE_THRESHOLD_BYTES:
            self._dvc_add(step_dir)
            dvc_file = str(step_dir) + ".dvc"
            commit_msg = (
                f"session {self.session_id}: step {step_id} complete"
                f" [mlflow_run_id: {mlflow_run_id}]"
            )
            self._git_commit([dvc_file], commit_msg)
        else:
            commit_msg = (
                f"session {self.session_id}: step {step_id} complete"
                f" [mlflow_run_id: {mlflow_run_id}]"
            )
            self._git_add_all(step_dir)
            self._git_commit_staged(commit_msg)

    def commit_final_report(self, report_dir: Path) -> None:
        """Коммитит финальный отчёт.

        Args:
            report_dir: директория с отчётом.
        """
        self._dvc_add(report_dir)
        commit_msg = f"session {self.session_id}: final report"
        self._git_commit([str(report_dir) + ".dvc"], commit_msg)

    def _dvc_add(self, path: Path) -> None:
        """Добавляет файл или директорию в DVC.

        Args:
            path: путь к файлу или директории.
        """
        self._run(["dvc", "add", str(path)])

    def _git_commit(self, files: list[str], message: str) -> None:
        """Добавляет файлы в git и делает коммит.

        Args:
            files: список файлов для git add.
            message: сообщение коммита.
        """
        self._run(["git", "add", *files])
        self._git_commit_staged(message)

    def _git_add_all(self, path: Path) -> None:
        """Добавляет все файлы в директории в git staging.

        Args:
            path: директория.
        """
        self._run(["git", "add", str(path)])

    def _git_commit_staged(self, message: str) -> None:
        """Коммитит то, что уже в staging.

        Args:
            message: сообщение коммита.
        """
        try:
            self._run(["git", "commit", "-m", message])
        except subprocess.CalledProcessError as exc:
            # ничего не staged — не ошибка
            if "nothing to commit" in (exc.output or "") or "nothing added" in (exc.output or ""):
                logger.debug("git commit: нечего коммитить для: %s", message)
            else:
                raise

    def _run(self, cmd: list[str]) -> subprocess.CompletedProcess:  # type: ignore[type-arg]
        """Выполняет команду в рабочей директории.

        Args:
            cmd: команда и аргументы.

        Returns:
            CompletedProcess с результатом выполнения.

        Raises:
            subprocess.CalledProcessError: команда завершилась с ненулевым кодом.
        """
        logger.debug("DVC/git: %s", " ".join(cmd))
        return subprocess.run(
            cmd,
            cwd=str(self.work_dir),
            check=True,
            capture_output=True,
            text=True,
        )

    @staticmethod
    def _get_size(path: Path) -> int:
        """Возвращает размер файла или директории в байтах.

        Args:
            path: путь к файлу или директории.

        Returns:
            Размер в байтах.
        """
        if path.is_file():
            return path.stat().st_size
        total = 0
        for child in path.rglob("*"):
            if child.is_file():
                total += child.stat().st_size
        return total
