from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from typing import Any

import numpy as np
from fitparse import FitFile


@dataclass
class ParsedFitMetrics:
    total_distance: float | None = None
    elevation_gain: float | None = None
    average_hr: float | None = None
    aerobic_decoupling: float | None = None
    athlete_weight: float | None = None
    final_cadence: float | None = None
    pace_min_km: float | None = None
    age: int | None = None


def parse_fit_metrics(content: bytes) -> ParsedFitMetrics:
    fit = FitFile(BytesIO(content))
    session = _first_message(fit, "session")

    total_distance_m = _message_value(session, "total_distance")
    elevation_gain_m = _message_value(session, "total_ascent")
    avg_speed_m_s = _first_available(session, ["enhanced_avg_speed", "avg_speed"])
    average_hr = _message_value(session, "avg_heart_rate")
    final_cadence = _first_available(session, ["avg_running_cadence", "avg_cadence"])

    athlete_weight = None
    age = None

    user_profile = _first_message(fit, "user_profile")
    if user_profile is not None:
        athlete_weight = _message_value(user_profile, "weight")

        birth_year = _message_value(user_profile, "birth_year")
        if isinstance(birth_year, (int, float)) and birth_year > 1900:
            age = datetime.now().year - int(birth_year)

    total_distance = float(total_distance_m) / 1000.0 if _is_positive_number(total_distance_m) else None
    elevation_gain = float(elevation_gain_m) if elevation_gain_m is not None else None
    session_speed_kmh = float(avg_speed_m_s) * 3.6 if _is_positive_number(avg_speed_m_s) else None
    pace_min_km = (60.0 / session_speed_kmh) if _is_positive_number(session_speed_kmh) else None
    aerobic_decoupling = _compute_aerobic_decoupling(fit)

    return ParsedFitMetrics(
        total_distance=total_distance,
        elevation_gain=elevation_gain,
        average_hr=float(average_hr) if average_hr is not None else None,
        aerobic_decoupling=aerobic_decoupling,
        athlete_weight=float(athlete_weight) if athlete_weight is not None else None,
        final_cadence=float(final_cadence) if final_cadence is not None else None,
        pace_min_km=pace_min_km,
        age=age,
    )


def _compute_aerobic_decoupling(fit: FitFile) -> float | None:
    rows: list[tuple[float, float]] = []
    for record in fit.get_messages("record"):
        speed_m_s = _first_available(record, ["enhanced_speed", "speed"])
        hr = _message_value(record, "heart_rate")
        if not _is_positive_number(speed_m_s) or not _is_positive_number(hr):
            continue
        speed_kmh = float(speed_m_s) * 3.6
        rows.append((speed_kmh, float(hr)))

    if len(rows) < 20:
        return None

    arr = np.array(rows, dtype=float)
    split = len(arr) // 2
    first_half = arr[:split]
    second_half = arr[split:]

    ef1 = np.mean(first_half[:, 0] / first_half[:, 1])
    ef2 = np.mean(second_half[:, 0] / second_half[:, 1])
    if ef1 == 0:
        return None
    return float((ef2 - ef1) / ef1 * 100.0)


def _first_message(fit: FitFile, message_name: str):
    for msg in fit.get_messages(message_name):
        return msg
    return None


def _message_value(message: Any, field_name: str):
    if message is None:
        return None
    return message.get_value(field_name)


def _first_available(message: Any, field_names: list[str]):
    for field_name in field_names:
        value = _message_value(message, field_name)
        if value is not None:
            return value
    return None


def _is_positive_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and float(value) > 0
