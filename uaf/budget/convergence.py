"""Алгоритмы проверки конвергенции метрики для dynamic budget mode."""

import logging

logger = logging.getLogger(__name__)


def check_convergence(
    metrics_history: list[float],
    patience: int,
    min_delta: float,
    min_iterations: int,
) -> bool:
    """Проверяет конвергенцию метрики по относительному изменению.

    Конвергенция = все последние `patience` дельт меньше `min_delta`
    AND накоплено минимум `min_iterations` точек.

    Args:
        metrics_history: история значений метрики по итерациям.
        patience: количество последних итераций для анализа.
        min_delta: минимальное относительное изменение метрики.
        min_iterations: минимальное число итераций до проверки.

    Returns:
        True если метрика сошлась.
    """
    if len(metrics_history) < min_iterations:
        return False
    if len(metrics_history) < patience + 1:
        return False

    recent = metrics_history[-patience - 1 :]
    deltas = [
        abs(recent[i] - recent[i - 1]) / (abs(recent[i - 1]) + 1e-10)
        for i in range(1, len(recent))
    ]
    converged = all(d < min_delta for d in deltas)
    if converged:
        logger.info(
            "Конвергенция обнаружена: последние %d дельт=%s (порог %.4f)",
            patience,
            [f"{d:.6f}" for d in deltas],
            min_delta,
        )
    return converged


def check_convergence_with_llm_signal(
    metrics_history: list[float],
    patience: int,
    min_delta: float,
    min_iterations: int,
    llm_convergence_signal: float,
    llm_signal_threshold: float = 0.9,
    llm_consecutive_required: int = 2,
    llm_consecutive_count: int = 0,
) -> tuple[bool, str]:
    """Комбинированная проверка конвергенции: алгоритм + LLM сигнал.

    Args:
        metrics_history: история метрики.
        patience: patience для алгоритмической проверки.
        min_delta: порог изменения для алгоритмической проверки.
        min_iterations: минимальное число итераций.
        llm_convergence_signal: сигнал от Claude Code (0.0-1.0).
        llm_signal_threshold: порог для LLM сигнала.
        llm_consecutive_required: сколько раз подряд LLM сигнал должен быть выше порога.
        llm_consecutive_count: текущий счётчик последовательных LLM сигналов.

    Returns:
        Tuple (converged, reason) — флаг конвергенции и причина.
    """
    if len(metrics_history) >= min_iterations:
        algo_converged = check_convergence(
            metrics_history=metrics_history,
            patience=patience,
            min_delta=min_delta,
            min_iterations=min_iterations,
        )
        if algo_converged:
            return True, "metric_convergence"

    if (
        llm_convergence_signal >= llm_signal_threshold
        and llm_consecutive_count >= llm_consecutive_required
        and len(metrics_history) >= min_iterations
    ):
        logger.info(
            "Конвергенция по LLM сигналу: signal=%.2f (порог=%.2f), consecutive=%d",
            llm_convergence_signal,
            llm_signal_threshold,
            llm_consecutive_count,
        )
        return True, "llm_signal"

    return False, ""
