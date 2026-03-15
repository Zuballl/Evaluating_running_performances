from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler


@dataclass
class PCAExplainer:
    feature_names: list[str]
    scaler: MinMaxScaler | StandardScaler
    pca: PCA
    reference_scores: np.ndarray
    reference_score_min: float
    reference_score_max: float
    preprocessing: str
    clip_lower: np.ndarray | None
    clip_upper: np.ndarray | None


@dataclass
class PCAExplanation:
    score: float
    raw_score: float
    percentile: float
    contributions: pd.DataFrame


def fit_pca_explainer(
    df_numeric: pd.DataFrame,
    n_components: int = 1,
    preprocessing: str = "minmax",
    clip_quantile_low: float = 0.01,
    clip_quantile_high: float = 0.99,
) -> PCAExplainer:
    if n_components != 1:
        raise ValueError("XAI module currently supports only n_components=1.")
    if preprocessing not in {"minmax", "standard_clip"}:
        raise ValueError("Unsupported preprocessing. Use 'minmax' or 'standard_clip'.")
    if not (0.0 <= clip_quantile_low < clip_quantile_high <= 1.0):
        raise ValueError("clip quantiles must satisfy 0 <= low < high <= 1")

    feature_names = df_numeric.columns.tolist()
    clip_lower: np.ndarray | None = None
    clip_upper: np.ndarray | None = None

    if preprocessing == "standard_clip":
        clip_lower = df_numeric.quantile(clip_quantile_low).to_numpy(dtype=float)
        clip_upper = df_numeric.quantile(clip_quantile_high).to_numpy(dtype=float)
        clipped = _clip_frame(df_numeric, clip_lower, clip_upper)
        scaler = StandardScaler()
        scaled = scaler.fit_transform(clipped)
    else:
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df_numeric)

    pca = PCA(n_components=n_components)
    pca.fit(scaled)
    reference_scores = pca.transform(scaled).flatten()

    return PCAExplainer(
        feature_names=feature_names,
        scaler=scaler,
        pca=pca,
        reference_scores=reference_scores,
        reference_score_min=float(np.min(reference_scores)),
        reference_score_max=float(np.max(reference_scores)),
        preprocessing=preprocessing,
        clip_lower=clip_lower,
        clip_upper=clip_upper,
    )


def score_samples(explainer: PCAExplainer, samples: pd.DataFrame) -> np.ndarray:
    aligned = _align_features(samples, explainer.feature_names)
    prepared = _apply_preprocessing(aligned, explainer)
    scaled = explainer.scaler.transform(prepared)
    return explainer.pca.transform(scaled).flatten()


def explain_sample(explainer: PCAExplainer, sample: pd.Series | Mapping[str, float] | pd.DataFrame) -> PCAExplanation:
    sample_df = _to_single_row_dataframe(sample, explainer.feature_names)

    prepared = _apply_preprocessing(sample_df, explainer)
    scaled_row = explainer.scaler.transform(prepared)[0]
    centered_row = scaled_row - explainer.pca.mean_
    component = explainer.pca.components_[0]

    contribution_values = centered_row * component
    raw_score = float(np.sum(contribution_values))
    score = _normalize_score(raw_score, explainer.reference_score_min, explainer.reference_score_max)
    percentile = _score_percentile(raw_score, explainer.reference_scores)

    abs_values = np.abs(contribution_values)
    abs_sum = float(np.sum(abs_values))
    if abs_sum > 0:
        percent_abs = abs_values / abs_sum * 100.0
    else:
        percent_abs = np.zeros_like(abs_values)

    contributions = pd.DataFrame(
        {
            "feature": explainer.feature_names,
            "centered_scaled_value": centered_row,
            "pca_weight": component,
            "contribution": contribution_values,
            "contribution_pct_abs": percent_abs,
        }
    ).sort_values("contribution_pct_abs", ascending=False)

    return PCAExplanation(
        score=score,
        raw_score=raw_score,
        percentile=percentile,
        contributions=contributions.reset_index(drop=True),
    )


def _align_features(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    return df[feature_names].copy()


def _apply_preprocessing(df: pd.DataFrame, explainer: PCAExplainer) -> pd.DataFrame:
    if explainer.preprocessing == "standard_clip":
        if explainer.clip_lower is None or explainer.clip_upper is None:
            raise ValueError("Explainer is missing clip bounds for standard_clip preprocessing.")
        return _clip_frame(df, explainer.clip_lower, explainer.clip_upper)
    return df


def _clip_frame(df: pd.DataFrame, clip_lower: np.ndarray, clip_upper: np.ndarray) -> pd.DataFrame:
    clipped = df.to_numpy(dtype=float)
    clipped = np.clip(clipped, clip_lower, clip_upper)
    return pd.DataFrame(clipped, columns=df.columns, index=df.index)


def _to_single_row_dataframe(
    sample: pd.Series | Mapping[str, float] | pd.DataFrame,
    feature_names: list[str],
) -> pd.DataFrame:
    if isinstance(sample, pd.DataFrame):
        if len(sample) != 1:
            raise ValueError("Expected exactly one row for explanation.")
        row_df = sample.copy()
    elif isinstance(sample, pd.Series):
        row_df = sample.to_frame().T
    else:
        row_df = pd.DataFrame([dict(sample)])

    return _align_features(row_df, feature_names)


def _score_percentile(score: float, reference_scores: np.ndarray) -> float:
    if reference_scores.size == 0:
        return 0.0
    below_or_equal = float(np.mean(reference_scores <= score))
    return below_or_equal * 100.0


def _normalize_score(score: float, reference_min: float, reference_max: float) -> float:
    if reference_max <= reference_min:
        return 0.0
    normalized = (score - reference_min) / (reference_max - reference_min)
    return float(np.clip(normalized * 10.0, 0.0, 10.0))
