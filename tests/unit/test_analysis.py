"""Тест регрессии: best_value в _analysis_to_dict."""

from unittest.mock import MagicMock

from uaf.analysis.result_analyzer import _analysis_to_dict


def test_best_value_from_ranked_runs() -> None:
    """best_value берётся из ranked_runs[0].metrics."""
    analysis = MagicMock()
    analysis.session_id = "test"
    analysis.total_runs = 3
    analysis.completed_runs = 3
    analysis.failed_runs = 0
    analysis.partial_runs = 0
    analysis.target_metric = "roi"
    analysis.metric_direction = "maximize"
    analysis.has_predictions = False
    analysis.segment_analysis_note = ""

    run = MagicMock()
    run.run_id = "r1"
    run.run_name = "step1"
    run.status = "FINISHED"
    run.metrics = {"roi": 5.5, "auc": 0.8}
    run.params = {}
    analysis.ranked_runs = [run]

    analysis.metric_profile = None
    analysis.failure_analysis = MagicMock()
    analysis.failure_analysis.total_failed = 0
    analysis.failure_analysis.by_category = {}
    analysis.failure_analysis.systemic_category = None
    analysis.failure_analysis.examples = []
    analysis.param_correlations = []
    analysis.hypotheses = []

    result = _analysis_to_dict(analysis)
    assert result["best_value"] == 5.5


def test_best_value_from_metric_profile() -> None:
    """Fallback: best_value из metric_profile.best."""
    analysis = MagicMock()
    analysis.session_id = "test"
    analysis.total_runs = 1
    analysis.completed_runs = 1
    analysis.failed_runs = 0
    analysis.partial_runs = 0
    analysis.target_metric = "roi"
    analysis.metric_direction = "maximize"
    analysis.has_predictions = False
    analysis.segment_analysis_note = ""

    run = MagicMock()
    run.run_id = "r1"
    run.run_name = "step1"
    run.status = "FINISHED"
    run.metrics = {"auc": 0.8}  # нет roi
    run.params = {}
    analysis.ranked_runs = [run]

    analysis.metric_profile = MagicMock()
    analysis.metric_profile.metric_name = "roi"
    analysis.metric_profile.mean = 3.0
    analysis.metric_profile.std = 1.0
    analysis.metric_profile.best = 4.2
    analysis.metric_profile.worst = 1.5
    analysis.metric_profile.count = 5

    analysis.failure_analysis = MagicMock()
    analysis.failure_analysis.total_failed = 0
    analysis.failure_analysis.by_category = {}
    analysis.failure_analysis.systemic_category = None
    analysis.failure_analysis.examples = []
    analysis.param_correlations = []
    analysis.hypotheses = []

    result = _analysis_to_dict(analysis)
    assert result["best_value"] == 4.2


def test_best_value_none_no_runs() -> None:
    """Нет runs → best_value = None."""
    analysis = MagicMock()
    analysis.session_id = "test"
    analysis.total_runs = 0
    analysis.completed_runs = 0
    analysis.failed_runs = 0
    analysis.partial_runs = 0
    analysis.target_metric = "roi"
    analysis.metric_direction = "maximize"
    analysis.has_predictions = False
    analysis.segment_analysis_note = ""
    analysis.ranked_runs = []
    analysis.metric_profile = None
    analysis.failure_analysis = MagicMock()
    analysis.failure_analysis.total_failed = 0
    analysis.failure_analysis.by_category = {}
    analysis.failure_analysis.systemic_category = None
    analysis.failure_analysis.examples = []
    analysis.param_correlations = []
    analysis.hypotheses = []

    result = _analysis_to_dict(analysis)
    assert result["best_value"] is None
