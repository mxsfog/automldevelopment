"""Инициализация MLflow для UAF-сессии."""

import json
import logging
import subprocess
import time
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


class MLflowSetup:
    """Инициализирует MLflow (SQLite backend) и создаёт структуру runs для сессии.

    Атрибуты:
        uaf_dir: путь к .uaf/ директории рабочего пространства.
        tracking_uri: URI для MLflow tracking server.
        experiment_name: имя MLflow experiment.
        experiment_id: ID созданного experiment (после init).
        client: MlflowClient для работы с runs.
    """

    def __init__(
        self,
        uaf_dir: Path,
        session_id: str,
        auto_start_server: bool = True,
        server_host: str = "127.0.0.1",
        server_port: int = 5000,
    ) -> None:
        self.uaf_dir = uaf_dir
        self.session_id = session_id
        self.auto_start_server = auto_start_server
        self.server_host = server_host
        self.server_port = server_port

        self.db_path = uaf_dir / "mlflow.db"
        self.artifacts_dir = uaf_dir / "mlruns"
        self.tracking_uri = f"http://{server_host}:{server_port}"
        self.experiment_name = f"uaf/{session_id}"
        self.experiment_id: str | None = None
        self.client: MlflowClient | None = None
        self._server_process: subprocess.Popen | None = None  # type: ignore[type-arg]

    def init(self, task_title: str, budget_mode: str, claude_model: str) -> str:
        """Инициализирует MLflow: запускает сервер, создаёт experiment и Planning Run.

        Args:
            task_title: заголовок задачи из task.yaml.
            budget_mode: режим бюджета (fixed/dynamic).
            claude_model: модель Claude Code для сессии.

        Returns:
            experiment_id созданного experiment.
        """
        self.uaf_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        if self.auto_start_server:
            self._start_server()
            self._wait_for_server()

        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = MlflowClient(tracking_uri=self.tracking_uri)

        experiment = self.client.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            self.experiment_id = self.client.create_experiment(
                name=self.experiment_name,
                artifact_location=str(self.artifacts_dir / self.session_id),
            )
            logger.info(
                "Создан MLflow experiment: %s (id=%s)", self.experiment_name, self.experiment_id
            )
        else:
            self.experiment_id = experiment.experiment_id
            logger.info("Использован существующий MLflow experiment: %s", self.experiment_name)

        self._create_planning_run(
            task_title=task_title, budget_mode=budget_mode, claude_model=claude_model
        )
        self._write_mlflow_context()

        return self.experiment_id

    def _start_server(self) -> None:
        """Запускает mlflow server как фоновый процесс."""
        cmd = [
            "mlflow",
            "server",
            "--backend-store-uri",
            f"sqlite:///{self.db_path}",
            "--default-artifact-root",
            str(self.artifacts_dir),
            "--host",
            self.server_host,
            "--port",
            str(self.server_port),
        ]
        logger.info("Запуск MLflow server: %s", " ".join(cmd))
        self._server_process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def _wait_for_server(self, timeout_seconds: int = 30) -> None:
        """Ожидает готовности MLflow server.

        Args:
            timeout_seconds: максимальное время ожидания в секундах.

        Raises:
            TimeoutError: сервер не поднялся за отведённое время.
        """
        import urllib.request

        health_url = f"http://{self.server_host}:{self.server_port}/health"
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            try:
                urllib.request.urlopen(health_url, timeout=2)
                logger.info("MLflow server готов: %s", self.tracking_uri)
                return
            except Exception:
                time.sleep(1)
        raise TimeoutError(
            f"MLflow server не поднялся за {timeout_seconds} сек: {self.tracking_uri}"
        )

    def _create_planning_run(self, task_title: str, budget_mode: str, claude_model: str) -> None:
        """Создаёт Planning Run с метаданными сессии.

        Args:
            task_title: заголовок задачи.
            budget_mode: режим бюджета.
            claude_model: модель Claude Code.
        """
        assert self.experiment_id is not None
        assert self.client is not None

        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run(
            run_name="planning/initial", experiment_id=self.experiment_id
        ) as run:
            mlflow.log_params(
                {
                    "session_id": self.session_id,
                    "task_title": task_title,
                    "budget_mode": budget_mode,
                    "claude_model": claude_model,
                }
            )
            mlflow.set_tag("type", "planning")
            mlflow.set_tag("approval_status", "pending")
            mlflow.set_tag("session_id", self.session_id)
            self._planning_run_id = run.info.run_id
            logger.info("Создан Planning Run: %s", run.info.run_id)

    def _write_mlflow_context(self) -> None:
        """Записывает mlflow_context.json — файл для Claude Code."""
        context = {
            "tracking_uri": self.tracking_uri,
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "session_id": self.session_id,
            "planning_run_id": getattr(self, "_planning_run_id", None),
        }
        context_path = self.uaf_dir / "mlflow_context.json"
        context_path.write_text(json.dumps(context, indent=2))
        logger.info("Записан mlflow_context.json: %s", context_path)

    def mark_approved(self) -> None:
        """Обновляет тег approval_status в Planning Run на approved."""
        if not hasattr(self, "_planning_run_id"):
            return
        assert self.client is not None
        self.client.set_tag(self._planning_run_id, "approval_status", "approved")
        logger.info("Planning Run помечен как approved")

    def log_approval_metrics(
        self, approval_status: str, approval_time_iso: str, wait_time_seconds: float
    ) -> None:
        """Логирует метрики одобрения в Planning Run.

        Args:
            approval_status: approved/rejected.
            approval_time_iso: время одобрения в ISO формате.
            wait_time_seconds: сколько секунд ждали одобрения.
        """
        if not hasattr(self, "_planning_run_id"):
            return
        assert self.client is not None
        self.client.set_tag(self._planning_run_id, "approval_status", approval_status)
        self.client.set_tag(self._planning_run_id, "approval_time", approval_time_iso)
        self.client.log_metric(self._planning_run_id, "wait_time_seconds", wait_time_seconds)

    def stop_server(self) -> None:
        """Останавливает MLflow server если был запущен."""
        if self._server_process is not None:
            self._server_process.terminate()
            try:
                self._server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._server_process.kill()
            logger.info("MLflow server остановлен")
