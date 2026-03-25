"""Custom MCP tools для Claude Agent SDK.

4 tools: save_pipeline, check_budget, get_experiment_memory, log_experiment_result.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd
from claude_agent_sdk import create_sdk_mcp_server, tool
from mlflow.tracking import MlflowClient

from uaf import MLFLOW_DEFAULT_URI

logger = logging.getLogger(__name__)


def create_uaf_tools(
    session_dir: Path,
    budget_status_file: Path,
    mlflow_tracking_uri: str = MLFLOW_DEFAULT_URI,
    experiment_id: str | None = None,
    target_metric: str = "metric",
    train_data_path: Path | None = None,
) -> Any:
    """Создаёт MCP сервер с UAF custom tools.

    Args:
        session_dir: директория сессии
        budget_status_file: путь к budget_status.json
        mlflow_tracking_uri: URI MLflow tracking server
        experiment_id: ID MLflow эксперимента
        target_metric: имя целевой метрики
        train_data_path: путь к train данным (для валидации pipeline)
    """
    sample_data: pd.DataFrame | None = None
    if train_data_path and train_data_path.exists():
        try:
            sample_data = pd.read_csv(train_data_path, nrows=5)
        except Exception:
            logger.warning("Не удалось загрузить sample_data из %s", train_data_path)

    mlflow_client = MlflowClient(mlflow_tracking_uri) if mlflow_tracking_uri else None

    @tool(
        "save_pipeline",
        "Save best pipeline for chain continuation. MANDATORY before finishing session.",
        {"metric_value": float, "metric_name": str, "framework": str},
    )
    async def save_pipeline(args: dict[str, Any]) -> dict[str, Any]:
        models_dir = session_dir / "models" / "best"
        models_dir.mkdir(parents=True, exist_ok=True)
        pipeline_path = models_dir / "pipeline.pkl"

        if not pipeline_path.exists():
            return _error("pipeline.pkl not found. Save model with joblib.dump() first.")

        # Валидация: pickle загружается и predict работает
        try:
            import joblib

            pipeline = joblib.load(pipeline_path)
            if sample_data is not None:
                test_pred = pipeline.predict(sample_data)
                if len(test_pred) != len(sample_data):
                    return _error(
                        f"predict returned {len(test_pred)} rows, expected {len(sample_data)}"
                    )
        except Exception as exc:
            return _error(f"pipeline.pkl corrupted or incompatible: {exc}")

        # Cross-check metric с MLflow
        warning = ""
        if mlflow_client and experiment_id:
            try:
                runs = mlflow_client.search_runs(
                    experiment_ids=[experiment_id],
                    order_by=[f"metrics.{args['metric_name']} DESC"],
                    max_results=1,
                )
                if runs:
                    mlflow_best = runs[0].data.metrics.get(args["metric_name"])
                    if mlflow_best is not None:
                        diff = abs(args["metric_value"] - mlflow_best)
                        if diff > 0.01:
                            warning = (
                                f" WARNING: reported {args['metric_name']}="
                                f"{args['metric_value']:.4f} differs from MLflow best"
                                f" {mlflow_best:.4f} by {diff:.4f}"
                            )
            except Exception:
                pass

        metadata = {
            args["metric_name"]: args["metric_value"],
            "framework": args["framework"],
            "session_id": session_dir.name,
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        (models_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False)
        )
        msg = (
            f"Pipeline saved and validated: {args['metric_name']}={args['metric_value']:.4f}"
        )
        logger.info("[TOOL] %s%s", msg, warning)
        return _success(msg + warning)

    @tool("check_budget", "Check remaining budget. Call before stopping.", {})
    async def check_budget(args: dict[str, Any]) -> dict[str, Any]:
        if not budget_status_file.exists():
            return _success(json.dumps({"can_stop": False, "message": "Budget file not found"}))

        try:
            status = json.loads(budget_status_file.read_text())
        except Exception:
            return _success(json.dumps({"can_stop": False, "message": "Budget file unreadable"}))

        iterations_used = status.get("iterations_used", 0)
        iterations_limit = status.get("iterations_limit", 999)
        budget_fraction = status.get("budget_fraction_used", 0.0)
        time_fraction = status.get("time_elapsed", 0) / max(
            status.get("time_limit", 1), 1
        )
        hard_stop = status.get("hard_stop", False)

        can_stop = (
            iterations_used >= iterations_limit
            or budget_fraction >= 0.95
            or time_fraction >= 0.95
            or hard_stop
        )

        result = {
            "iterations_remaining": iterations_limit - iterations_used,
            "time_remaining_hours": round(
                max(0, status.get("time_limit", 0) - status.get("time_elapsed", 0)) / 3600,
                2,
            ),
            "budget_fraction": round(budget_fraction, 3),
            "hard_stop": hard_stop,
            "can_stop": can_stop,
            "message": (
                "You may stop." if can_stop
                else "Budget remaining. Generate new hypothesis."
            ),
        }
        return _success(json.dumps(result))

    @tool("get_experiment_memory", "Get structured history of all experiments.", {})
    async def get_experiment_memory(args: dict[str, Any]) -> dict[str, Any]:
        if not mlflow_client or not experiment_id:
            return _success(json.dumps({"experiments": [], "total": 0}))

        try:
            runs = mlflow_client.search_runs(
                experiment_ids=[experiment_id],
                order_by=["start_time DESC"],
                max_results=50,
            )
        except Exception as exc:
            return _error(f"MLflow query failed: {exc}")

        memory: list[dict[str, Any]] = []
        strategy_counts: dict[str, int] = {}
        best_metric: float | None = None

        for run in runs:
            tags = run.data.tags
            if tags.get("type") in ("planning", "ruff_report", "report"):
                continue

            metric_val = run.data.metrics.get(target_metric)
            if metric_val is not None and (best_metric is None or metric_val > best_metric):
                best_metric = metric_val

            strategy = tags.get("strategy_category", "unknown")
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

            memory.append({
                "step": tags.get("step", "?"),
                "hypothesis": tags.get("hypothesis", ""),
                "status": tags.get("status", run.info.status),
                "strategy_category": strategy,
                "metric": metric_val,
                "conclusion": tags.get("conclusion", ""),
                "run_name": run.info.run_name,
            })

        result = {
            "experiments": memory,
            "total": len(memory),
            "best_metric": best_metric,
            "strategy_distribution": strategy_counts,
        }
        return _success(json.dumps(result, default=str))

    @tool(
        "log_experiment_result",
        "Log structured result of current experiment step.",
        {
            "step": str,
            "hypothesis": str,
            "status": str,
            "strategy_category": str,
            "metric_value": float,
            "conclusion": str,
        },
    )
    async def log_experiment_result(args: dict[str, Any]) -> dict[str, Any]:
        # Post-hoc leakage validation
        warning = ""
        if args.get("metric_value") and mlflow_client and experiment_id:
            try:
                runs = mlflow_client.search_runs(
                    experiment_ids=[experiment_id],
                    filter_string="tags.type = 'experiment'",
                    max_results=5,
                )
                cv_metrics = [
                    r.data.metrics.get("cv_mean_roi", r.data.metrics.get("cv_mean", 0))
                    for r in runs
                    if r.data.metrics.get("cv_mean_roi") or r.data.metrics.get("cv_mean")
                ]
                if cv_metrics:
                    avg_cv = sum(cv_metrics) / len(cv_metrics)
                    if avg_cv > 0 and args["metric_value"] > 3 * avg_cv:
                        warning = (
                            f" LEAKAGE SUSPECT: metric {args['metric_value']:.4f}"
                            f" > 3x avg CV ({avg_cv:.4f}). Investigate."
                        )
            except Exception:
                pass

        msg = (
            f"Logged step {args['step']}: {args['status']}"
            f" ({target_metric}={args['metric_value']:.4f})"
        )
        logger.info("[TOOL] %s%s", msg, warning)
        return _success(msg + warning)

    server = create_sdk_mcp_server(
        "uaf-tools",
        tools=[save_pipeline, check_budget, get_experiment_memory, log_experiment_result],
    )
    return server


def _success(text: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}]}


def _error(text: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": f"ERROR: {text}"}], "isError": True}
