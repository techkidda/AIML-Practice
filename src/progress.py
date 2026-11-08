from __future__ import annotations

from collections.abc import Sequence


def estimate_remaining_seconds(
    completed_model_seconds: Sequence[float],
    remaining_models: int,
) -> float | None:
    if not completed_model_seconds:
        return None
    average_seconds = sum(completed_model_seconds) / len(completed_model_seconds)
    return average_seconds * max(remaining_models, 0)


def format_duration(seconds: float) -> str:
    rounded_seconds = max(0, round(seconds))
    minutes, remaining_seconds = divmod(rounded_seconds, 60)
    if minutes:
        return f"{minutes}m {remaining_seconds}s"
    return f"{remaining_seconds}s"
