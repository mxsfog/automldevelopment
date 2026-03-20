"""ReportGenerator — компиляция LaTeX/PDF из MLflow данных и секций Claude Code."""

import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import jinja2

from uaf.analysis.result_analyzer import SessionAnalysis, load_session_analysis
from uaf.core.ruff_enforcer import RuffReport
from uaf.reporting.latex_templates import (
    ANALYSIS_SECTION,
    CODE_QUALITY,
    EXECUTIVE_SUMMARY,
    EXPERIMENT_RESULTS,
    REPORT_MAIN,
    REPRODUCIBILITY,
    TASK_DESCRIPTION,
)

logger = logging.getLogger(__name__)

# Суффиксы для поиска LaTeX компилятора
_LATEX_COMPILERS = ["tectonic", "pdflatex", "xelatex"]


def _latex_escape(text: str) -> str:
    """Экранирует специальные символы LaTeX.

    Использует однопроходный regex чтобы не ломать уже добавленные escape-последовательности.

    Args:
        text: исходный текст.

    Returns:
        Экранированный текст.
    """
    special: dict[str, str] = {
        "\\": r"\textbackslash{}",
        "{": r"\{",
        "}": r"\}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    pattern = re.compile("|".join(re.escape(k) for k in special))
    return pattern.sub(lambda m: special[m.group()], text)


def _md_to_latex(text: str) -> str:
    """Базовая конвертация Markdown в LaTeX.

    Конвертирует заголовки, жирный, курсив, списки, блоки кода.

    Args:
        text: Markdown текст.

    Returns:
        LaTeX текст (приближённый).
    """
    # Блоки кода
    text = re.sub(
        r"```[a-z]*\n(.*?)```",
        r"\\begin{lstlisting}\n\1\\end{lstlisting}",
        text,
        flags=re.DOTALL,
    )

    # Заголовки
    text = re.sub(r"^### (.+)$", r"\\subsubsection{\1}", text, flags=re.MULTILINE)
    text = re.sub(r"^## (.+)$", r"\\subsection{\1}", text, flags=re.MULTILINE)
    text = re.sub(r"^# (.+)$", r"\\subsection{\1}", text, flags=re.MULTILINE)

    # Жирный и курсив
    text = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", text)
    text = re.sub(r"\*(.+?)\*", r"\\textit{\1}", text)

    # Inline code
    text = re.sub(r"`(.+?)`", r"\\texttt{\1}", text)

    # Маркированные списки
    lines = text.split("\n")
    result = []
    in_list = False
    for line in lines:
        if line.strip().startswith("- ") or line.strip().startswith("* "):
            if not in_list:
                result.append("\\begin{itemize}")
                in_list = True
            result.append("  \\item " + line.strip()[2:])
        else:
            if in_list:
                result.append("\\end{itemize}")
                in_list = False
            result.append(line)
    if in_list:
        result.append("\\end{itemize}")
    text = "\n".join(result)

    return text


def _create_jinja_env() -> jinja2.Environment:
    """Создаёт Jinja2 окружение с кастомными фильтрами.

    Returns:
        Настроенное jinja2.Environment.
    """
    env = jinja2.Environment(
        loader=jinja2.BaseLoader(),
        undefined=jinja2.Undefined,
        trim_blocks=True,
        lstrip_blocks=True,
        block_start_string="<%",
        block_end_string="%>",
        variable_start_string="<<",
        variable_end_string=">>",
        comment_start_string="<#",
        comment_end_string="#>",
    )
    env.filters["latex_escape"] = _latex_escape
    env.filters["md_to_latex"] = _md_to_latex
    env.globals["enumerate"] = enumerate
    return env


class ReportGenerator:
    """Генерирует LaTeX отчёт и компилирует в PDF.

    Читает:
    - Текстовые секции из SESSION_DIR/report/sections/ (сгенерированы Claude Code)
    - session_analysis.json (ResultAnalyzer)
    - ruff_report.json (RuffEnforcer)
    - MLflow runs для таблиц

    Генерирует:
    - matplotlib графики (Metric Progression, Budget Burndown)
    - .tex файл через Jinja2
    - PDF через tectonic или pdflatex

    Атрибуты:
        session_dir: директория сессии.
        session_id: ID сессии.
        tracking_uri: MLflow tracking URI.
        experiment_id: MLflow experiment ID.
        task_config: конфигурация задачи.
    """

    def __init__(
        self,
        session_dir: Path,
        session_id: str,
        tracking_uri: str,
        experiment_id: str,
        task_config: dict[str, Any] | None = None,
    ) -> None:
        self.session_dir = session_dir
        self.session_id = session_id
        self.tracking_uri = tracking_uri
        self.experiment_id = experiment_id
        self.task_config = task_config or {}

        self.report_dir = session_dir / "report"
        self.sections_dir = self.report_dir / "sections"
        self.figures_dir = self.report_dir / "figures"
        self._jinja_env = _create_jinja_env()

    def compile_from_sections(self, ruff_report: RuffReport | None = None) -> Path | None:
        """Полный pipeline генерации отчёта.

        Шаги:
        1. Создание директорий
        2. Загрузка session_analysis.json
        3. Чтение текстовых секций от Claude Code
        4. Генерация matplotlib графиков
        5. Сборка .tex
        6. Компиляция PDF

        Args:
            ruff_report: результат RuffEnforcer (опционально).

        Returns:
            Путь к PDF или .tex как fallback, или None при полном провале.
        """
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.sections_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        logger.info("ReportGenerator: начало генерации отчёта для сессии %s", self.session_id)

        # Загрузка анализа
        analysis = load_session_analysis(self.session_dir)
        if analysis is None:
            logger.warning(
                "session_analysis.json не найден, отчёт будет без analysis секции"
            )

        # Чтение ruff_report.json если ruff_report не передан
        if ruff_report is None:
            ruff_report = self._load_ruff_report()

        # Читаем текстовые секции от Claude Code
        sections_text = self._read_text_sections()

        # Генерируем фигуры
        metric_figure = self._generate_metric_figure(analysis)
        budget_figure = self._generate_budget_figure()
        alert_figure = self._generate_alert_figure()

        # Собираем .tex
        tex_content = self._build_tex(
            analysis=analysis,
            ruff_report=ruff_report,
            sections_text=sections_text,
            metric_figure=metric_figure,
            budget_figure=budget_figure,
            alert_figure=alert_figure,
        )

        tex_path = self.report_dir / "report.tex"
        tex_path.write_text(tex_content, encoding="utf-8")
        logger.info("report.tex записан: %s", tex_path)

        # Компиляция PDF
        pdf_path = self._compile_pdf(tex_path)
        if pdf_path:
            self._log_to_mlflow(pdf_path, tex_path)
            return pdf_path

        logger.warning("PDF не скомпилирован, доступен .tex: %s", tex_path)
        return tex_path

    def _read_text_sections(self) -> dict[str, str]:
        """Читает текстовые секции из SESSION_DIR/report/sections/.

        Returns:
            Словарь {секция: текст}.
        """
        sections: dict[str, str] = {}
        section_files = {
            "executive_summary": "executive_summary.md",
            "analysis_and_findings": "analysis_and_findings.md",
            "recommendations": "recommendations.md",
            "monitoring_conclusions": "monitoring_conclusions.md",
        }
        for key, filename in section_files.items():
            path = self.sections_dir / filename
            if path.exists():
                sections[key] = path.read_text(encoding="utf-8")
                logger.debug("Прочитана секция %s (%d символов)", key, len(sections[key]))
            else:
                sections[key] = ""
        return sections

    def _generate_metric_figure(self, analysis: SessionAnalysis | None) -> Path | None:
        """Генерирует график Metric Progression.

        Args:
            analysis: результат анализа сессии.

        Returns:
            Путь к PDF-графику или None.
        """
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            if analysis is None or not analysis.ranked_runs:
                return None

            target_metric = analysis.target_metric
            # Берём данные из ranked_runs (sorted by start_time original)
            runs_with_metric = [
                r for r in analysis.ranked_runs if target_metric in r.metrics
            ]
            if not runs_with_metric:
                return None

            # Для прогресса нам нужна история — ranked_runs уже отсортированы по метрике,
            # но мы хотим порядок по времени. Используем start_time если доступен.
            sorted_by_time = sorted(runs_with_metric, key=lambda r: r.start_time)
            values = [r.metrics[target_metric] for r in sorted_by_time]
            iterations = list(range(1, len(values) + 1))

            # Накопленный лучший
            if analysis.metric_direction == "maximize":
                running_best = [max(values[: i + 1]) for i in range(len(values))]
            else:
                running_best = [min(values[: i + 1]) for i in range(len(values))]

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(iterations, values, "o-", alpha=0.6, label="Итерация", color="steelblue")
            ax.plot(iterations, running_best, "s--", label="Лучший", color="darkred", linewidth=2)
            ax.set_xlabel("Итерация")
            ax.set_ylabel(target_metric)
            ax.set_title(f"Metric Progression: {target_metric}")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()

            fig_path = self.figures_dir / "metric_progression.pdf"
            fig.savefig(str(fig_path), bbox_inches="tight")
            plt.close(fig)
            logger.info("Metric Progression график: %s", fig_path)
            return fig_path

        except Exception as exc:
            logger.warning("Не удалось сгенерировать metric figure: %s", exc)
            return None

    def _generate_budget_figure(self) -> Path | None:
        """Генерирует график Budget Burndown из budget_status.json.

        Returns:
            Путь к PDF-графику или None.
        """
        try:
            import json

            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            budget_file = self.session_dir.parent / "budget_status.json"
            if not budget_file.exists():
                # Fallback: файл в parent директории
                budget_file = self.session_dir / ".." / "budget_status.json"
            if not budget_file.exists():
                return None

            status = json.loads(budget_file.read_text(encoding="utf-8"))
            iterations_used = status.get("iterations_used", 0)
            iterations_limit = status.get("iterations_limit", None)
            metrics_history = status.get("metrics_history", [])

            if iterations_used == 0:
                return None

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            # Budget usage
            ax1 = axes[0]
            if iterations_limit:
                used_pct = min(iterations_used / iterations_limit * 100, 100)
                remaining_pct = 100 - used_pct
                ax1.bar(
                    ["Использовано", "Остаток"],
                    [used_pct, remaining_pct],
                    color=["#e74c3c", "#2ecc71"],
                )
                ax1.set_ylabel("% итераций")
                ax1.set_title("Бюджет итераций")
                ax1.set_ylim(0, 110)
                for i, v in enumerate([used_pct, remaining_pct]):
                    ax1.text(i, v + 1, f"{v:.1f}%", ha="center")
            else:
                ax1.text(
                    0.5,
                    0.5,
                    f"Итераций: {iterations_used}",
                    ha="center",
                    va="center",
                    transform=ax1.transAxes,
                )
                ax1.set_title("Бюджет итераций (dynamic mode)")

            # Metrics history
            ax2 = axes[1]
            if metrics_history:
                x_vals = range(1, len(metrics_history) + 1)
                ax2.plot(x_vals, metrics_history, "o-", color="steelblue")
                ax2.set_xlabel("Итерация")
                ax2.set_ylabel("Метрика")
                ax2.set_title("Метрика по итерациям")
                ax2.grid(True, alpha=0.3)
            else:
                ax2.text(0.5, 0.5, "Нет данных", ha="center", va="center", transform=ax2.transAxes)
                ax2.set_title("Метрика по итерациям")

            plt.tight_layout()
            fig_path = self.figures_dir / "budget_burndown.pdf"
            fig.savefig(str(fig_path), bbox_inches="tight")
            plt.close(fig)
            logger.info("Budget Burndown график: %s", fig_path)
            return fig_path

        except Exception as exc:
            logger.warning("Не удалось сгенерировать budget figure: %s", exc)
            return None

    def _generate_alert_figure(self) -> Path | None:
        """Генерирует Alert Timeline если есть WARNING+ алерты.

        Returns:
            Путь к PDF-графику или None.
        """
        try:
            import json
            import time

            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            budget_file = self.session_dir.parent / "budget_status.json"
            if not budget_file.exists():
                return None

            status = json.loads(budget_file.read_text(encoding="utf-8"))
            alerts = status.get("alerts", [])

            # Фильтруем WARNING и CRITICAL
            significant = [a for a in alerts if a.get("level") in ("WARNING", "CRITICAL")]
            if not significant:
                return None

            session_start = status.get("timestamp", time.time()) - status.get("time_elapsed", 0)
            alert_times = [(a["timestamp"] - session_start) / 60 for a in significant]
            alert_codes = [a["code"] for a in significant]
            alert_levels = [a["level"] for a in significant]
            colors = ["#e74c3c" if lv == "CRITICAL" else "#f39c12" for lv in alert_levels]

            fig, ax = plt.subplots(figsize=(12, 4))
            y_pos = range(len(significant))
            ax.barh(y_pos, [0.3] * len(significant), left=alert_times, color=colors, height=0.6)
            ax.set_yticks(list(y_pos))
            ax.set_yticklabels(alert_codes, fontsize=9)
            ax.set_xlabel("Время от старта сессии (минуты)")
            ax.set_title("Alert Timeline")

            from matplotlib.patches import Patch

            legend_elements = [
                Patch(facecolor="#e74c3c", label="CRITICAL"),
                Patch(facecolor="#f39c12", label="WARNING"),
            ]
            ax.legend(handles=legend_elements, loc="lower right")
            ax.grid(True, axis="x", alpha=0.3)
            plt.tight_layout()

            fig_path = self.figures_dir / "alert_timeline.pdf"
            fig.savefig(str(fig_path), bbox_inches="tight")
            plt.close(fig)
            logger.info("Alert Timeline график: %s", fig_path)
            return fig_path

        except Exception as exc:
            logger.warning("Не удалось сгенерировать alert figure: %s", exc)
            return None

    def _build_tex(
        self,
        analysis: SessionAnalysis | None,
        ruff_report: RuffReport | None,
        sections_text: dict[str, str],
        metric_figure: Path | None,
        budget_figure: Path | None,
        alert_figure: Path | None,
    ) -> str:
        """Собирает полный .tex файл из секций через Jinja2.

        Args:
            analysis: результат анализа.
            ruff_report: результат ruff.
            sections_text: текстовые секции от Claude Code.
            metric_figure: путь к графику метрики.
            budget_figure: путь к budget burndown графику.
            alert_figure: путь к alert timeline графику.

        Returns:
            Строка .tex содержимого.
        """
        from datetime import datetime

        task_title = self.task_config.get("title", f"Session {self.session_id}")
        target_metric = self.task_config.get("target_metric", "metric")
        target_column = self.task_config.get("target_column", "---")
        task_type = self.task_config.get("task_type", "---")
        dataset_path = self.task_config.get("dataset_path", "---")
        metric_direction = self.task_config.get("metric_direction", "maximize")
        problem_statement = self.task_config.get("problem_statement", "")
        claude_model = self.task_config.get("claude_model", "claude-opus-4")
        random_seed = self.task_config.get("random_seed", None)
        mlflow_experiment = f"uaf/{self.session_id}"
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M UTC")

        # Подготовка данных для секций
        total_runs = analysis.total_runs if analysis else 0
        completed_runs = analysis.completed_runs if analysis else 0
        failed_runs = analysis.failed_runs if analysis else 0
        partial_runs = analysis.partial_runs if analysis else 0
        ranked_runs = analysis.ranked_runs if analysis else []
        all_ranked = analysis.ranked_runs if analysis else []
        failed_runs_list = [r for r in all_ranked if r.status == "failed"]
        best_value = ranked_runs[0].metrics.get(target_metric) if ranked_runs else None
        hypotheses = analysis.hypotheses if analysis else []
        param_correlations = analysis.param_correlations if analysis else []

        # Загрузка all run ids из MLflow
        all_run_ids = self._fetch_all_run_ids()
        requirements_lock = self._read_requirements_lock()
        dvc_commit = self._read_dvc_commit()
        git_sha = self._read_git_sha()

        # Рендеринг каждой секции
        def render(template_str: str, extra_ctx: dict | None = None) -> str:
            ctx = {
                "session_id": self.session_id,
                "task_title": task_title,
                "target_metric": target_metric,
                "target_column": target_column,
                "task_type": task_type,
                "dataset_path": dataset_path,
                "metric_direction": metric_direction,
                "problem_statement": problem_statement,
                "claude_model": claude_model,
                "random_seed": random_seed,
                "mlflow_experiment": mlflow_experiment,
                "created_at": created_at,
                "total_runs": total_runs,
                "completed_runs": completed_runs,
                "failed_runs": failed_runs,
                "partial_runs": partial_runs,
                "ranked_runs": ranked_runs,
                "failed_runs_list": failed_runs_list,
                "best_value": best_value,
                "hypotheses": hypotheses,
                "param_correlations": param_correlations,
                "ruff_report": ruff_report,
                "all_run_ids": all_run_ids,
                "requirements_lock": requirements_lock,
                "dvc_commit": dvc_commit,
                "git_sha": git_sha,
                "metric_figure_path": str(metric_figure) if metric_figure else None,
                "budget_figure_path": str(budget_figure) if budget_figure else None,
                "alert_figure_path": str(alert_figure) if alert_figure else None,
                "executive_summary_text": sections_text.get("executive_summary", ""),
                "analysis_text": sections_text.get("analysis_and_findings", "")
                + "\n\n"
                + sections_text.get("recommendations", ""),
            }
            if extra_ctx:
                ctx.update(extra_ctx)
            tmpl = self._jinja_env.from_string(template_str)
            return tmpl.render(**ctx)

        exec_summary = render(EXECUTIVE_SUMMARY)
        task_desc = render(TASK_DESCRIPTION)
        exp_results = render(EXPERIMENT_RESULTS)
        analysis_sec = render(ANALYSIS_SECTION)
        code_quality = render(CODE_QUALITY)
        reproducibility = render(REPRODUCIBILITY)

        full_tex = render(
            REPORT_MAIN,
            {
                "executive_summary_section": exec_summary,
                "task_description_section": task_desc,
                "experiment_results_section": exp_results,
                "analysis_section": analysis_sec,
                "code_quality_section": code_quality,
                "reproducibility_section": reproducibility,
            },
        )
        return full_tex

    def _compile_pdf(self, tex_path: Path) -> Path | None:
        """Компилирует .tex в PDF через tectonic → pdflatex → fallback.

        Args:
            tex_path: путь к .tex файлу.

        Returns:
            Путь к PDF или None если компиляция не удалась.
        """
        pdf_path = tex_path.with_suffix(".pdf")

        # Пробуем tectonic
        tectonic_bin = shutil.which("tectonic") or str(Path.home() / ".local" / "bin" / "tectonic")
        if Path(tectonic_bin).exists():
            try:
                result = subprocess.run(
                    [tectonic_bin, str(tex_path.resolve())],
                    capture_output=True,
                    text=True,
                    timeout=300,
                    cwd=str(self.report_dir),
                )
                if result.returncode == 0 and pdf_path.exists():
                    logger.info("PDF скомпилирован через tectonic: %s", pdf_path)
                    return pdf_path
                logger.warning(
                    "tectonic завершился с кодом %d: %s",
                    result.returncode,
                    result.stderr[:500],
                )
            except Exception as exc:
                logger.warning("tectonic ошибка: %s", exc)

        # Пробуем pdflatex
        for compiler in ["pdflatex", "xelatex"]:
            if shutil.which(compiler):
                try:
                    for _ in range(2):  # двойной проход для TOC и ссылок
                        result = subprocess.run(
                            [
                                compiler,
                                "-interaction=nonstopmode",
                                "-output-directory",
                                str(self.report_dir),
                                str(tex_path),
                            ],
                            capture_output=True,
                            text=True,
                            timeout=120,
                            cwd=str(self.report_dir),
                        )
                    if pdf_path.exists():
                        logger.info("PDF скомпилирован через %s: %s", compiler, pdf_path)
                        return pdf_path
                    logger.warning("%s завершился без PDF: %s", compiler, result.stderr[:500])
                except Exception as exc:
                    logger.warning("%s ошибка: %s", compiler, exc)

        logger.warning(
            "LaTeX компилятор не найден или не удалось скомпилировать PDF. "
            "Доступен .tex: %s",
            tex_path,
        )
        return None

    def _load_ruff_report(self) -> RuffReport | None:
        """Загружает ruff_report.json если существует.

        Returns:
            RuffReport или None.
        """
        import json

        from uaf.core.ruff_enforcer import RuffFileResult

        ruff_path = self.session_dir / "ruff_report.json"
        if not ruff_path.exists():
            return None
        try:
            data = json.loads(ruff_path.read_text(encoding="utf-8"))
            return RuffReport(
                total_files=data.get("total_files", 0),
                clean_files=data.get("clean_files", 0),
                files_with_unfixable=data.get("files_with_unfixable", 0),
                total_violations=data.get("total_violations", 0),
                clean_rate=data.get("clean_rate", 1.0),
                files=[
                    RuffFileResult(
                        file=Path(f["file"]),
                        formatted=f.get("formatted", True),
                        violations_before_fix=f.get("violations_before_fix", 0),
                        violations_after_fix=f.get("violations_after_fix", 0),
                    )
                    for f in data.get("files", [])
                ],
                ruff_version=data.get("ruff_version", "unknown"),
                target_met=data.get("target_met", True),
            )
        except Exception as exc:
            logger.warning("Не удалось загрузить ruff_report.json: %s", exc)
            return None

    def _fetch_all_run_ids(self) -> list[str]:
        """Возвращает все MLflow run IDs сессии.

        Returns:
            Список run ID строк.
        """
        try:
            from mlflow.tracking import MlflowClient

            client = MlflowClient(tracking_uri=self.tracking_uri)
            runs = client.search_runs(
                experiment_ids=[self.experiment_id],
                filter_string=f"tags.session_id = '{self.session_id}'",
                max_results=500,
            )
            return [r.info.run_id for r in runs]
        except Exception as exc:
            logger.debug("Ошибка получения run IDs: %s", exc)
            return []

    def _read_requirements_lock(self) -> str | None:
        """Читает requirements.lock из SESSION_DIR.

        Returns:
            Содержимое файла или None.
        """
        lock_path = self.session_dir / "requirements.lock"
        if lock_path.exists():
            return lock_path.read_text(encoding="utf-8")
        return None

    def _read_dvc_commit(self) -> str | None:
        """Читает последний DVC-связанный git commit SHA.

        Returns:
            Git SHA или None.
        """
        try:
            result = subprocess.run(
                ["git", "log", "--oneline", "-1", "--", str(self.session_dir)],
                capture_output=True,
                text=True,
                cwd=str(self.session_dir.parent),
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip().split()[0]
        except Exception:
            pass
        return None

    def _read_git_sha(self) -> str | None:
        """Читает текущий HEAD git SHA.

        Returns:
            Git SHA или None.
        """
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=str(self.session_dir),
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()[:12]
        except Exception:
            pass
        return None

    def _log_to_mlflow(self, pdf_path: Path, tex_path: Path) -> None:
        """Логирует артефакты отчёта в MLflow.

        Args:
            pdf_path: путь к PDF.
            tex_path: путь к .tex.
        """
        try:
            import mlflow

            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(f"uaf/{self.session_id}")

            with mlflow.start_run(
                run_name="report/final",
                experiment_id=self.experiment_id,
            ):
                mlflow.set_tag("type", "report")
                mlflow.set_tag("session_id", self.session_id)
                mlflow.set_tag("status", "complete")
                if pdf_path.exists():
                    mlflow.log_artifact(str(pdf_path), artifact_path="report")
                if tex_path.exists():
                    mlflow.log_artifact(str(tex_path), artifact_path="report")
                # Логируем figures
                for fig in self.figures_dir.glob("*.pdf"):
                    mlflow.log_artifact(str(fig), artifact_path="report/figures")

            logger.info("Отчёт залогирован в MLflow")
        except Exception as exc:
            logger.warning("Не удалось залогировать отчёт в MLflow: %s", exc)
