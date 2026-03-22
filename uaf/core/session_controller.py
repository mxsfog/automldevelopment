"""ResearchSessionController — state machine UAF-сессии исследования."""

import json
import logging
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import yaml

logger = logging.getLogger(__name__)

# Состояния state machine
SessionState = Literal[
    "IDLE",
    "SETUP",
    "DATA_LOADING",
    "PLANNING",
    "AWAITING_APPROVAL",
    "EXECUTING",
    "ANALYZING",
    "REPORTING",
    "DONE",
    "FAILED",
    "REJECTED",
]

# Допустимые переходы состояний
_TRANSITIONS: dict[str, list[str]] = {
    "IDLE": ["SETUP"],
    "SETUP": ["DATA_LOADING", "FAILED"],
    "DATA_LOADING": ["PLANNING", "FAILED"],
    "PLANNING": ["AWAITING_APPROVAL", "FAILED"],
    "AWAITING_APPROVAL": ["EXECUTING", "REJECTED", "FAILED"],
    "EXECUTING": ["ANALYZING", "FAILED"],
    "ANALYZING": ["REPORTING", "FAILED"],
    "REPORTING": ["DONE", "FAILED"],
    "DONE": [],
    "FAILED": [],
    "REJECTED": [],
}


@dataclass
class SessionStateData:
    """Сериализуемое состояние сессии для --resume.

    Атрибуты:
        session_id: уникальный идентификатор сессии.
        state: текущее состояние state machine.
        work_dir: рабочая директория.
        task_path: путь к task.yaml.
        budget_path: путь к budget.yaml.
        claude_model: модель Claude Code.
        fully_autonomous: режим без approval.
        mlflow_experiment_id: ID MLflow эксперимента.
        mlflow_tracking_uri: URI MLflow сервера.
        created_at: время создания сессии (ISO).
        updated_at: время последнего обновления (ISO).
        approval_result: результат HumanOversightGate.
        error: сообщение об ошибке (если FAILED).
        resume_count: сколько раз сессия возобновлялась.
    """

    session_id: str
    state: str
    work_dir: str
    task_path: str
    budget_path: str | None
    claude_model: str
    fully_autonomous: bool
    mlflow_experiment_id: str | None = None
    mlflow_tracking_uri: str = "http://127.0.0.1:5000"
    created_at: str = ""
    updated_at: str = ""
    approval_result: dict[str, Any] | None = None
    error: str | None = None
    resume_count: int = 0


class ResearchSessionController:
    """Оркестрирует UAF-сессию через state machine.

    Управляет жизненным циклом: IDLE → SETUP → DATA_LOADING → PLANNING
    → AWAITING_APPROVAL → EXECUTING → ANALYZING → REPORTING → DONE/FAILED.

    Сохраняет состояние в SESSION_DIR/session_state.json для поддержки --resume.

    Атрибуты:
        work_dir: рабочая директория пользователя.
        task_path: путь к task.yaml.
        budget_path: путь к budget.yaml (опционально).
        session_id: ID сессии (генерируется если не указан).
        claude_model: модель для Claude Code subprocess.
        fully_autonomous: пропустить HumanOversightGate.
        mlflow_tracking_uri: URI MLflow сервера.
    """

    def __init__(
        self,
        work_dir: Path,
        task_path: Path,
        budget_path: Path | None = None,
        session_id: str | None = None,
        claude_model: str = "claude-opus-4",
        fully_autonomous: bool = False,
        mlflow_tracking_uri: str = "http://127.0.0.1:5000",
        prev_session_id: str | None = None,
    ) -> None:
        self.work_dir = work_dir.resolve()
        self.task_path = task_path.resolve()
        self.budget_path = budget_path.resolve() if budget_path else None
        self.session_id = session_id or self._generate_session_id()
        self.claude_model = claude_model
        self.fully_autonomous = fully_autonomous
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self._prev_session_id = prev_session_id

        self._uaf_dir = work_dir / ".uaf"
        self._session_dir = self._uaf_dir / "sessions" / self.session_id
        self._state_file = self._session_dir / "session_state.json"
        self._budget_status_file = self._uaf_dir / "budget_status.json"

        self._state: SessionState = "IDLE"
        self._task_config: dict[str, Any] = {}
        self._budget_config: dict[str, Any] = {}
        self._mlflow_experiment_id: str | None = None
        self._state_lock = threading.Lock()
        self._budget_controller = None
        self._claude_runner = None

        logger.info(
            "ResearchSessionController инициализирован: session_id=%s, work_dir=%s",
            self.session_id,
            self.work_dir,
        )

    @classmethod
    def from_resume(cls, work_dir: Path, session_id: str) -> "ResearchSessionController":
        """Восстанавливает контроллер из сохранённого состояния.

        Args:
            work_dir: рабочая директория.
            session_id: ID сессии для возобновления.

        Returns:
            ResearchSessionController с восстановленным состоянием.

        Raises:
            FileNotFoundError: если session_state.json не найден.
        """
        uaf_dir = work_dir / ".uaf"
        session_dir = uaf_dir / "sessions" / session_id
        state_file = session_dir / "session_state.json"

        if not state_file.exists():
            raise FileNotFoundError(f"session_state.json не найден: {state_file}")

        data = json.loads(state_file.read_text(encoding="utf-8"))
        saved_state = SessionStateData(**data)

        controller = cls(
            work_dir=Path(saved_state.work_dir),
            task_path=Path(saved_state.task_path),
            budget_path=Path(saved_state.budget_path) if saved_state.budget_path else None,
            session_id=saved_state.session_id,
            claude_model=saved_state.claude_model,
            fully_autonomous=saved_state.fully_autonomous,
            mlflow_tracking_uri=saved_state.mlflow_tracking_uri,
        )
        controller._state = saved_state.state  # type: ignore[assignment]
        controller._mlflow_experiment_id = saved_state.mlflow_experiment_id
        controller._state_data = saved_state
        controller._state_data.resume_count += 1
        logger.info(
            "Сессия %s восстановлена из состояния %s (resume #%d)",
            session_id,
            saved_state.state,
            controller._state_data.resume_count,
        )
        return controller

    def run(self) -> bool:
        """Запускает полный lifecycle сессии.

        Проходит через все состояния от текущего до DONE или FAILED.

        Returns:
            True если сессия завершилась успешно (DONE).
        """
        logger.info("Запуск ResearchSessionController: state=%s", self._state)

        try:
            if self._state == "IDLE":
                self._transition("SETUP")
                self._do_setup()

            if self._state == "SETUP":
                self._transition("DATA_LOADING")
                self._do_data_loading()

            if self._state == "DATA_LOADING":
                self._transition("PLANNING")
                self._do_planning()

            if self._state == "PLANNING":
                self._transition("AWAITING_APPROVAL")
                approved = self._do_approval()
                if not approved:
                    self._transition("REJECTED")
                    logger.info("Сессия отклонена на этапе одобрения")
                    return False

            if self._state == "AWAITING_APPROVAL":
                self._transition("EXECUTING")
                self._do_executing()

            if self._state == "EXECUTING":
                self._transition("ANALYZING")
                self._do_analyzing()

            if self._state == "ANALYZING":
                self._transition("REPORTING")
                self._do_reporting()

            if self._state == "REPORTING":
                self._transition("DONE")
                self._print_session_summary()

            return self._state == "DONE"

        except KeyboardInterrupt:
            logger.warning("Прерывание (KeyboardInterrupt): установка FAILED")
            self._fail("KeyboardInterrupt")
            return False
        except Exception as exc:
            logger.error("Необработанная ошибка в сессии: %s", exc, exc_info=True)
            self._fail(str(exc))
            return False

    def _do_setup(self) -> None:
        """SETUP: создание директорий, инициализация MLflow и DVC."""
        logger.info("[SETUP] Инициализация инфраструктуры")
        self._session_dir.mkdir(parents=True, exist_ok=True)

        self._task_config = self._load_yaml(self.task_path)
        self._budget_config = self._load_yaml(self.budget_path) if self.budget_path else {}

        self._save_state()

        # MLflow setup
        from uaf.integrations.mlflow_setup import MLflowSetup

        task = self._task_config.get("task", {})
        mlflow_setup = MLflowSetup(
            uaf_dir=self._uaf_dir,
            session_id=self.session_id,
            auto_start_server=True,
        )
        self._mlflow_experiment_id = mlflow_setup.init(
            task_title=task.get("title", "UAF Session"),
            budget_mode=self._budget_config.get("budget", {}).get("mode", "fixed"),
            claude_model=self.claude_model,
        )
        logger.info("[SETUP] MLflow experiment: %s", self._mlflow_experiment_id)

        # DVC setup
        from uaf.integrations.dvc_setup import DVCSetup

        dvc = DVCSetup(work_dir=self.work_dir, session_id=self.session_id)
        try:
            dvc.init()
        except Exception as exc:
            logger.warning("[SETUP] DVC init не удался (не критично): %s", exc)

        self._save_state()
        logger.info("[SETUP] завершён")

    def _do_data_loading(self) -> None:
        """DATA_LOADING: загрузка данных, leakage audit, adversarial validation."""
        logger.info("[DATA_LOADING] Загрузка и валидация данных")
        task = self._task_config.get("task", {})
        data_cfg = self._task_config.get("data", {})

        # Поддержка files[role=main] и legacy train_path
        files = data_cfg.get("files", [])
        main_file = next((f for f in files if f.get("role") == "main"), None)
        train_path_str = (
            main_file.get("path") if main_file else data_cfg.get("train_path")
        )
        target_column = data_cfg.get("target_column", task.get("target_column", "target"))

        if not train_path_str:
            logger.info("[DATA_LOADING] train_path не указан, пропуск загрузки данных")
            self._save_state()
            return

        train_path = (self.work_dir / train_path_str).resolve()

        try:
            from uaf.data.loader import DataLoader

            loader = DataLoader(
                train_path=train_path,
                target_column=target_column,
            )
            data_schema = loader.load()
            n_rows = data_schema.splits[0].n_rows if data_schema.splits else 0
            n_features = len(data_schema.features)
            logger.info(
                "[DATA_LOADING] Загружено: %d строк, %d признаков",
                n_rows,
                n_features,
            )
        except Exception as exc:
            logger.warning("[DATA_LOADING] Ошибка загрузки данных: %s", exc)

        self._save_state()
        logger.info("[DATA_LOADING] завершён")

    def _do_planning(self) -> None:
        """PLANNING: подготовка context/ пакета для Claude Code."""
        logger.info("[PLANNING] Подготовка context/ пакета")

        improvement_context_path: Path | None = None
        prev_session_dir: Path | None = None
        if self._prev_session_id:
            prev_session_dir = self._uaf_dir / "sessions" / self._prev_session_id
            improvement_context_path = self._build_improvement_context(prev_session_dir)

        try:
            from uaf.core.program_generator import ProgramMdGenerator

            generator = ProgramMdGenerator(
                session_dir=self._session_dir,
            )
            generator.prepare_context(
                task_path=self.task_path,
                session_id=self.session_id,
                improvement_context_path=improvement_context_path,
                prev_session_dir=prev_session_dir,
            )
            logger.info("[PLANNING] context/ пакет подготовлен")
        except Exception as exc:
            logger.warning("[PLANNING] Ошибка подготовки context/: %s", exc)

        self._save_state()
        logger.info("[PLANNING] завершён")

    def _do_approval(self) -> bool:
        """AWAITING_APPROVAL: HumanOversightGate.

        Returns:
            True если одобрено.
        """
        logger.info("[AWAITING_APPROVAL] Запуск HumanOversightGate")

        program_md_path = self._session_dir / "program.md"
        # Если program.md ещё не существует — используем context/program_md_template.md
        if not program_md_path.exists():
            context_template = self._session_dir / "context" / "program_md_template.md"
            if context_template.exists():
                import shutil

                shutil.copy(context_template, program_md_path)
            else:
                # Создаём минимальный program.md
                self._create_minimal_program_md(program_md_path)

        approval_mode = "fully_autonomous" if self.fully_autonomous else "standard"
        adv_auc = None  # TODO: читать из data_schema если есть

        from uaf.core.oversight import ApprovalResult, HumanOversightGate

        gate = HumanOversightGate(
            program_md_path=program_md_path,
            approval_mode=approval_mode,  # type: ignore[arg-type]
            adversarial_auc=adv_auc,
        )
        result: ApprovalResult = gate.check()

        # Сохраняем результат в state
        approval_dict = {
            "approved": result.approved,
            "modified": result.modified,
            "approval_mode": result.approval_mode,
            "wait_time_seconds": result.wait_time_seconds,
            "edit_rounds": result.edit_rounds,
            "notes": result.notes,
        }
        if hasattr(self, "_state_data"):
            self._state_data.approval_result = approval_dict
        self._save_state()

        if result.approved and self._mlflow_experiment_id:
            try:
                from uaf.integrations.mlflow_setup import MLflowSetup

                mlflow_setup = MLflowSetup(
                    uaf_dir=self._uaf_dir,
                    session_id=self.session_id,
                    auto_start_server=False,
                )
                mlflow_setup.client = None  # будет переиспользовать tracking_uri
                gate.log_to_mlflow(
                    result=result,
                    tracking_uri=self.mlflow_tracking_uri,
                )
            except Exception as exc:
                logger.debug("Не удалось залогировать approval в MLflow: %s", exc)

        logger.info("[AWAITING_APPROVAL] approved=%s", result.approved)
        return result.approved

    def _do_executing(self) -> None:
        """EXECUTING: запуск Claude Code subprocess + BudgetController."""
        logger.info("[EXECUTING] Запуск Claude Code")

        # Инициализация budget_status.json
        from uaf.budget.status_file import BudgetStatusV21, write_budget_status

        initial_status = BudgetStatusV21(session_id=self.session_id)
        write_budget_status(initial_status, self._budget_status_file)

        # Конфигурация бюджета
        from uaf.budget.controller import BudgetConfig, BudgetController

        budget_cfg = self._budget_config.get("budget", {})
        metric_cfg = self._task_config.get("metric", {}) or self._task_config.get("task", {}).get(
            "metric", {}
        )
        budget_config = BudgetConfig(
            mode=budget_cfg.get("mode", "fixed"),
            max_iterations=budget_cfg.get("max_iterations", 20),
            max_time_hours=budget_cfg.get("max_time_hours", 8.0),
            patience=budget_cfg.get("convergence", {}).get("patience", 3),
            min_delta=budget_cfg.get("convergence", {}).get("min_delta", 0.001),
            min_iterations=budget_cfg.get("convergence", {}).get("min_iterations", 3),
            metric_direction=metric_cfg.get("direction", "maximize"),
            metric_name=metric_cfg.get("name", "roi"),
            leakage_sanity_threshold=metric_cfg.get("leakage_sanity_threshold"),
            leakage_soft_warning=metric_cfg.get("leakage_soft_warning"),
        )

        experiment_id = self._mlflow_experiment_id or "0"
        self._budget_controller = BudgetController(
            budget_status_path=self._budget_status_file,
            experiment_id=experiment_id,
            tracking_uri=self.mlflow_tracking_uri,
            config=budget_config,
            session_id=self.session_id,
        )

        # ClaudeCodeRunner
        from uaf.runner.claude_runner import ClaudeCodeRunner

        session_timeout = budget_cfg.get("max_time_hours", 24.0) * 3600 + 300  # +5 мин буфер
        self._claude_runner = ClaudeCodeRunner(
            session_dir=self._session_dir,
            session_id=self.session_id,
            claude_model=self.claude_model,
            mlflow_tracking_uri=self.mlflow_tracking_uri,
            mlflow_experiment_name=f"uaf/{self.session_id}",
            budget_status_file=self._budget_status_file,
            stdout_callback=lambda line: self._budget_controller.update_stdout_time(),  # type: ignore[union-attr]
            timeout_seconds=session_timeout,
            fully_autonomous=self.fully_autonomous,
            on_start=lambda pid: self._budget_controller.set_claude_pid(pid),  # type: ignore[union-attr]
        )

        # Запускаем BudgetController в thread
        self._budget_controller.start(claude_pid=None)

        try:
            return_code = self._claude_runner.run()
            logger.info("[EXECUTING] Claude Code завершился: return_code=%d", return_code)
        except FileNotFoundError:
            logger.error(
                "[EXECUTING] claude CLI не найден в PATH. "
                "Установите Claude Code: https://claude.ai/code"
            )
            raise
        finally:
            self._budget_controller.stop()

        self._save_state()
        logger.info("[EXECUTING] завершён")

    def _do_analyzing(self) -> None:
        """ANALYZING: ResultAnalyzer + SystemErrorAnalyzer + RuffEnforcer."""
        logger.info("[ANALYZING] Post-session анализ")

        if not self._mlflow_experiment_id:
            logger.warning("[ANALYZING] MLflow experiment_id не задан, пропуск анализа")
            self._save_state()
            return

        task = self._task_config.get("task", {})
        target_metric = task.get("metric", {}).get("name", "metric")
        metric_direction = task.get("metric", {}).get("direction", "maximize")

        # RuffEnforcer
        try:
            from uaf.core.ruff_enforcer import RuffEnforcer

            enforcer = RuffEnforcer(session_dir=self._session_dir)
            ruff_report = enforcer.enforce()
            enforcer.log_to_mlflow(
                report=ruff_report,
                tracking_uri=self.mlflow_tracking_uri,
                experiment_id=self._mlflow_experiment_id,
                session_id=self.session_id,
            )
            self._ruff_report = ruff_report
            logger.info(
                "[ANALYZING] RuffEnforcer: clean_rate=%.1f%%",
                ruff_report.clean_rate * 100,
            )
        except Exception as exc:
            logger.warning("[ANALYZING] RuffEnforcer ошибка: %s", exc)
            self._ruff_report = None

        # ResultAnalyzer
        try:
            from uaf.analysis.result_analyzer import ResultAnalyzer

            analyzer = ResultAnalyzer(
                session_id=self.session_id,
                experiment_id=self._mlflow_experiment_id,
                tracking_uri=self.mlflow_tracking_uri,
                session_dir=self._session_dir,
                target_metric=target_metric,
                metric_direction=metric_direction,  # type: ignore[arg-type]
            )
            analysis = analyzer.analyze()
            logger.info(
                "[ANALYZING] ResultAnalyzer: total=%d, completed=%d, failed=%d",
                analysis.total_runs,
                analysis.completed_runs,
                analysis.failed_runs,
            )
        except Exception as exc:
            logger.warning("[ANALYZING] ResultAnalyzer ошибка: %s", exc)

        # SystemErrorAnalyzer
        try:
            from uaf.analysis.system_error_analyzer import SystemErrorAnalyzer

            sys_analyzer = SystemErrorAnalyzer(
                session_id=self.session_id,
                session_dir=self._session_dir,
                experiment_id=self._mlflow_experiment_id,
                tracking_uri=self.mlflow_tracking_uri,
            )
            sys_report = sys_analyzer.analyze()
            logger.info(
                "[ANALYZING] SystemErrorAnalyzer: health=%s", sys_report.overall_health
            )
        except Exception as exc:
            logger.warning("[ANALYZING] SystemErrorAnalyzer ошибка: %s", exc)

        self._auto_save_best_model()

        self._save_state()
        logger.info("[ANALYZING] завершён")

    def _auto_save_best_model(self) -> None:
        """Системный fallback: сохраняет лучшую модель в models/best/.

        Вызывается после ANALYZING. Если Claude Code уже сохранил модель согласно
        Model Artifact Protocol — ничего не делает. Иначе ищет model files в
        experiments/, берёт самый последний, копирует и создаёт минимальный
        metadata.json чтобы следующая сессия могла пропустить Phase 1-3.
        """
        models_best_dir = self._session_dir / "models" / "best"
        metadata_file = models_best_dir / "metadata.json"

        if metadata_file.exists():
            logger.info("[AUTO-SAVE] models/best/metadata.json уже существует — пропуск")
            return

        experiments_dir = self._session_dir / "experiments"
        if not experiments_dir.exists():
            return

        # Ищем model files по расширениям
        _ext_to_framework = {
            ".cbm": "catboost",
            ".lgb": "lgbm",
            ".xgb": "xgboost",
            ".pkl": "sklearn",
            ".joblib": "sklearn",
        }
        candidates: list[tuple[Path, str, float]] = []
        for ext, framework in _ext_to_framework.items():
            for path in experiments_dir.rglob(f"*{ext}"):
                candidates.append((path, framework, path.stat().st_mtime))

        # Также проверяем models/best/ на случай частичного сохранения
        if models_best_dir.exists():
            for ext, framework in _ext_to_framework.items():
                for path in models_best_dir.glob(f"*{ext}"):
                    candidates.append((path, framework, path.stat().st_mtime))

        if not candidates:
            logger.info("[AUTO-SAVE] model files не найдены, пропуск")
            return

        best_path, framework, _ = max(candidates, key=lambda x: x[2])

        models_best_dir.mkdir(parents=True, exist_ok=True)
        target = models_best_dir / best_path.name
        if not target.exists():
            import shutil
            shutil.copy2(best_path, target)

        # Извлекаем feature names из модели
        feature_names: list[str] = []
        try:
            if framework == "catboost":
                from catboost import CatBoostClassifier
                m = CatBoostClassifier()
                m.load_model(str(target))
                feature_names = list(m.feature_names_)
            elif framework == "lgbm":
                import lightgbm as lgb
                m = lgb.Booster(model_file=str(target))
                feature_names = list(m.feature_name())
            elif framework == "xgboost":
                import xgboost as xgb
                m = xgb.Booster()
                m.load_model(str(target))
                feature_names = list(m.feature_names or [])
            elif framework == "sklearn":
                import joblib
                m = joblib.load(target)
                if hasattr(m, "feature_names_in_"):
                    feature_names = list(m.feature_names_in_)
        except Exception as exc:
            logger.warning("[AUTO-SAVE] не удалось извлечь feature names: %s", exc)

        # Берём метрику из session_analysis.json
        best_value: float | None = None
        analysis_file = self._session_dir / "session_analysis.json"
        if analysis_file.exists():
            try:
                analysis_data = json.loads(analysis_file.read_text(encoding="utf-8"))
                best_value = analysis_data.get("best_value")
            except Exception:
                pass

        metric_name = self._task_config.get("metric", {}).get("name", "metric")

        # Проверяем есть ли pipeline.pkl рядом с model file
        pipeline_src = best_path.parent / "pipeline.pkl"
        pipeline_file_name: str | None = None
        if pipeline_src.exists():
            import shutil as _shutil
            pipeline_target = models_best_dir / "pipeline.pkl"
            if not pipeline_target.exists():
                _shutil.copy2(pipeline_src, pipeline_target)
            pipeline_file_name = "pipeline.pkl"
            logger.info("[AUTO-SAVE] pipeline.pkl скопирован из %s", pipeline_src)
        else:
            logger.warning(
                "[AUTO-SAVE] pipeline.pkl не найден рядом с %s — "
                "следующая сессия будет использовать fallback через model_file",
                best_path,
            )

        metadata: dict[str, Any] = {
            "framework": framework,
            "model_file": target.name,
            "pipeline_file": pipeline_file_name,
            "session_id": self.session_id,
            metric_name: best_value,
            "feature_names": feature_names,
            "auto_saved": True,
        }
        metadata_file.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
        logger.info(
            "[AUTO-SAVE] модель сохранена: %s (pipeline=%s, framework=%s, features=%d, %s=%.4f)",
            target.name,
            pipeline_file_name or "нет",
            framework,
            len(feature_names),
            metric_name,
            best_value or 0.0,
        )

    def _do_reporting(self) -> None:
        """REPORTING: ReportGenerator -> LaTeX -> PDF."""
        logger.info("[REPORTING] Генерация отчёта")

        if not self._mlflow_experiment_id:
            logger.warning("[REPORTING] MLflow experiment_id не задан, пропуск генерации отчёта")
            self._save_state()
            return

        task = self._task_config.get("task", {})
        metric_cfg = self._task_config.get("metric", {})
        data_cfg = self._task_config.get("data", {})
        data_files = data_cfg.get("files", [])
        main_file = next((f for f in data_files if f.get("role") == "main"), data_files[0] if data_files else {})
        task_config_flat = {
            "title": task.get("name", task.get("title", f"Session {self.session_id}")),
            "task_type": task.get("type", "---"),
            "target_metric": metric_cfg.get("name", "roi"),
            "metric_direction": metric_cfg.get("direction", "maximize"),
            "target_column": data_cfg.get("target_column", "---"),
            "dataset_path": main_file.get("path", "---"),
            "problem_statement": task.get("description", ""),
            "claude_model": self.claude_model,
        }

        try:
            from uaf.reporting.report_generator import ReportGenerator

            generator = ReportGenerator(
                session_dir=self._session_dir,
                session_id=self.session_id,
                tracking_uri=self.mlflow_tracking_uri,
                experiment_id=self._mlflow_experiment_id,
                task_config=task_config_flat,
            )
            ruff_report = getattr(self, "_ruff_report", None)
            result_path = generator.compile_from_sections(ruff_report=ruff_report)

            if result_path:
                logger.info("[REPORTING] Отчёт создан: %s", result_path)
            else:
                logger.warning("[REPORTING] Отчёт не создан")
        except Exception as exc:
            logger.error("[REPORTING] ReportGenerator ошибка: %s", exc, exc_info=True)

        self._save_state()
        logger.info("[REPORTING] завершён")

    def _print_session_summary(self) -> None:
        """Выводит финальное summary сессии в терминал."""
        report_dir = self._session_dir / "report"
        pdf_files = list(report_dir.glob("*.pdf")) if report_dir.exists() else []
        pdf_path = pdf_files[0] if pdf_files else None

        print("\n" + "=" * 70)
        print(f"  UAF Сессия завершена: {self.session_id}")
        print("=" * 70)
        if pdf_path:
            print(f"  Отчёт: {pdf_path}")
        else:
            tex_files = list(report_dir.glob("*.tex")) if report_dir.exists() else []
            if tex_files:
                print(f"  Отчёт (.tex fallback): {tex_files[0]}")
        print(f"  MLflow UI: {self.mlflow_tracking_uri}")
        print(f"  Experiment: uaf/{self.session_id}")
        print(f"  Session dir: {self._session_dir}")
        print("=" * 70)

    def _transition(self, new_state: str) -> None:
        """Переводит state machine в новое состояние.

        Args:
            new_state: целевое состояние.

        Raises:
            ValueError: если переход недопустим.
        """
        with self._state_lock:
            allowed = _TRANSITIONS.get(self._state, [])
            if new_state not in allowed:
                raise ValueError(
                    f"Недопустимый переход состояния: {self._state} → {new_state}. "
                    f"Допустимые: {allowed}"
                )
            old_state = self._state
            self._state = new_state  # type: ignore[assignment]
            logger.info("Переход состояния: %s → %s", old_state, new_state)
            self._save_state()

    def _fail(self, error: str) -> None:
        """Устанавливает FAILED состояние с сообщением об ошибке.

        Args:
            error: описание ошибки.
        """
        with self._state_lock:
            self._state = "FAILED"
            logger.error("Сессия %s перешла в FAILED: %s", self.session_id, error)
            self._save_state(error=error)

        # Останавливаем BudgetController если запущен
        if self._budget_controller is not None:
            import contextlib

            with contextlib.suppress(Exception):
                self._budget_controller.stop()

    def _save_state(self, error: str | None = None) -> None:
        """Сохраняет текущее состояние в session_state.json.

        Args:
            error: опциональное сообщение об ошибке.
        """
        self._session_dir.mkdir(parents=True, exist_ok=True)
        now = datetime.now(tz=timezone.utc).isoformat()

        data = {
            "session_id": self.session_id,
            "state": self._state,
            "work_dir": str(self.work_dir),
            "task_path": str(self.task_path),
            "budget_path": str(self.budget_path) if self.budget_path else None,
            "claude_model": self.claude_model,
            "fully_autonomous": self.fully_autonomous,
            "mlflow_experiment_id": self._mlflow_experiment_id,
            "mlflow_tracking_uri": self.mlflow_tracking_uri,
            "created_at": getattr(
                getattr(self, "_state_data", None), "created_at", now
            ),
            "updated_at": now,
            "approval_result": getattr(
                getattr(self, "_state_data", None), "approval_result", None
            ),
            "error": error,
            "resume_count": getattr(
                getattr(self, "_state_data", None), "resume_count", 0
            ),
        }
        tmp = self._state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        import os

        os.replace(tmp, self._state_file)

    def _load_yaml(self, path: Path) -> dict[str, Any]:
        """Загружает YAML файл.

        Args:
            path: путь к YAML.

        Returns:
            Словарь с данными.
        """
        try:
            with path.open(encoding="utf-8") as fh:
                return yaml.safe_load(fh) or {}
        except Exception as exc:
            logger.error("Ошибка загрузки YAML %s: %s", path, exc)
            return {}

    def _create_minimal_program_md(self, path: Path) -> None:
        """Создаёт минимальный program.md если context/ не готов.

        Args:
            path: путь для записи program.md.
        """
        task = self._task_config.get("task", {})
        content = f"""# Research Program: {task.get("title", "UAF Session")}

## Metadata
- session_id: {self.session_id}
- created: {datetime.now(tz=timezone.utc).isoformat()}
- approved_by: pending
- approval_time: null
- budget_mode: {self._budget_config.get("budget", {}).get("mode", "fixed")}
- claude_model: {self.claude_model}

## Task Description
{task.get("problem_statement", "Описание задачи не задано.")}

## Research Phases

### Phase 1: Baseline
**Goal:** Установить baseline метрики
**Success Criterion:** Хоть какой-то результат

#### Step 1.1: Baseline
- **Hypothesis:** Baseline модель задаёт нижний предел
- **Method:** DummyClassifier/Regressor
- **Metric:** {task.get("metric", {}).get("name", "metric")}
- **Critical:** true
- **Status:** pending
- **MLflow Run ID:** null
- **Result:**
- **Conclusion:**

## Current Status
- **Active Phase:** Phase 1
- **Completed Steps:** 0/1
- **Best Result:** null
- **Budget Used:** 0%

## Iteration Log

## Final Conclusions

---

## Execution Instructions

### MLflow Logging
Каждый эксперимент обязан использовать MLflow.

### Code Quality
После каждого .py файла: `ruff format <file> && ruff check <file> --fix`

### Budget Check
Проверяй UAF_BUDGET_STATUS_FILE перед каждым экспериментом.
"""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        logger.info("Создан минимальный program.md: %s", path)

    def _build_improvement_context(self, prev_session_dir: Path) -> Path | None:
        """Генерирует improvement_context.md из итогов предыдущей сессии.

        Читает program.md и session_analysis.json из директории предыдущей сессии
        и формирует структурированный контекст для следующей сессии.

        Args:
            prev_session_dir: директория предыдущей сессии.

        Returns:
            Путь к сгенерированному improvement_context.md или None при ошибке.
        """
        if not prev_session_dir.exists():
            logger.warning(
                "[PLANNING] Директория предыдущей сессии не найдена: %s", prev_session_dir
            )
            return None

        prev_session_id = prev_session_dir.name
        program_md_path = prev_session_dir / "program.md"
        analysis_path = prev_session_dir / "session_analysis.json"

        # Извлекаем секции из program.md
        final_conclusions = ""
        iteration_log = ""
        accepted_features = ""

        if program_md_path.exists():
            try:
                content = program_md_path.read_text(encoding="utf-8")
                sections = self._extract_program_md_sections(content)
                final_conclusions = sections.get("Final Conclusions", "").strip()
                iteration_log = sections.get("Iteration Log", "").strip()
                accepted_features = sections.get("Accepted Features", "").strip()
            except Exception as exc:
                logger.warning("[PLANNING] Не удалось прочитать program.md: %s", exc)
        else:
            logger.warning("[PLANNING] program.md не найден в предыдущей сессии: %s", program_md_path)

        # Читаем session_analysis.json
        best_metric_value: float | None = None
        best_metric_name = "metric"
        ranked_runs_text = ""
        failed_runs_text = ""

        if analysis_path.exists():
            try:
                analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
                best_metric_value = analysis.get("best_metric_value")
                best_metric_name = analysis.get("target_metric", "metric")

                ranked_runs = analysis.get("ranked_runs", [])[:5]
                if ranked_runs:
                    lines = []
                    for i, run in enumerate(ranked_runs, 1):
                        run_name = run.get("run_name", run.get("run_id", f"run_{i}"))
                        metric_val = run.get("metric_value", "?")
                        tags = run.get("tags", {})
                        step = tags.get("step", "")
                        step_str = f" (step: {step})" if step else ""
                        lines.append(f"- #{i}: {run_name}{step_str} → {best_metric_name}={metric_val}")
                    ranked_runs_text = "\n".join(lines)

                failed_runs = analysis.get("failed_runs", [])
                if failed_runs:
                    fail_lines = []
                    for run in failed_runs[:5]:
                        run_name = run.get("run_name", run.get("run_id", "unknown"))
                        reason = run.get("failure_reason", run.get("error", "неизвестна"))
                        fail_lines.append(f"- {run_name}: {reason}")
                    failed_runs_text = "\n".join(fail_lines)
            except Exception as exc:
                logger.warning("[PLANNING] Не удалось прочитать session_analysis.json: %s", exc)
        else:
            logger.info("[PLANNING] session_analysis.json не найден: %s", analysis_path)

        # Формируем improvement_context.md
        best_result_str = (
            f"{best_metric_value:.6f}" if best_metric_value is not None else "нет данных"
        )

        lines: list[str] = [
            f"# Previous Session Context: {prev_session_id}",
            "",
            "## Best Results Achieved",
            f"- Best {best_metric_name}: {best_result_str}",
            "",
        ]

        if ranked_runs_text:
            lines += [
                "## Top Runs (do NOT repeat identical configurations)",
                ranked_runs_text,
                "",
            ]

        if failed_runs_text:
            lines += [
                "## Failed Runs (причины провалов)",
                failed_runs_text,
                "",
            ]

        if iteration_log:
            lines += [
                "## What Was Tried (do NOT repeat)",
                iteration_log,
                "",
            ]

        if accepted_features:
            lines += [
                "## Accepted Features",
                accepted_features,
                "",
            ]

        if final_conclusions:
            lines += [
                "## Recommended Next Steps",
                final_conclusions,
                "",
            ]

        ctx_content = "\n".join(lines)

        output_path = self._session_dir / "improvement_context.md"
        self._session_dir.mkdir(parents=True, exist_ok=True)
        output_path.write_text(ctx_content, encoding="utf-8")
        logger.info(
            "[PLANNING] improvement_context.md сгенерирован из сессии %s (%d байт)",
            prev_session_id,
            len(ctx_content),
        )
        return output_path

    @staticmethod
    def _extract_program_md_sections(content: str) -> dict[str, str]:
        """Извлекает именованные секции из program.md.

        Разбивает документ по заголовкам ## и возвращает словарь
        {название_секции: содержимое_секции}.

        Args:
            content: полный текст program.md.

        Returns:
            Словарь секций без заголовочных строк.
        """
        sections: dict[str, str] = {}
        current_name: str | None = None
        current_lines: list[str] = []

        for line in content.splitlines():
            if line.startswith("## "):
                if current_name is not None:
                    sections[current_name] = "\n".join(current_lines)
                current_name = line[3:].strip()
                current_lines = []
            elif current_name is not None:
                current_lines.append(line)

        if current_name is not None:
            sections[current_name] = "\n".join(current_lines)

        return sections

    @staticmethod
    def _generate_session_id() -> str:
        """Генерирует уникальный session_id.

        Returns:
            Строка вида 'YYYYMMDD-HHMMSS-{uuid8}'.
        """
        now = datetime.now(tz=timezone.utc)
        date_part = now.strftime("%Y%m%d-%H%M%S")
        uuid_part = str(uuid.uuid4()).replace("-", "")[:8]
        return f"{date_part}-{uuid_part}"

    @property
    def state(self) -> str:
        """Текущее состояние state machine."""
        return self._state

    @property
    def session_dir(self) -> Path:
        """Директория сессии."""
        return self._session_dir
