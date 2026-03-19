from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_METRICS_PATH = ROOT_DIR / "outputs" / "metrics" / "model_comparison.csv"
DEFAULT_METRICWISE_PATH = ROOT_DIR / "outputs" / "metrics" / "model_comparison_by_metric.csv"
DEFAULT_METRIC_AGREEMENT_PATH = ROOT_DIR / "outputs" / "metrics" / "metric_agreement_by_model.csv"
TOP_FEATURES_COUNT = 3
METRIC_KEYS = ("spearman", "kendall", "mutual_info", "permutation_importance")
DEFAULT_METRIC_WEIGHTS = {
    "spearman": 0.25,
    "kendall": 0.2,
    "mutual_info": 0.25,
    "permutation_importance": 0.3,
}


def _safe_spearman(left: pd.Series, right: pd.Series) -> float:
    value = left.corr(right, method="spearman")
    if pd.isna(value):
        return 0.0
    return float(value)


def _safe_kendall(left: pd.Series, right: pd.Series) -> float:
    value, _ = kendalltau(left, right)
    if pd.isna(value):
        return 0.0
    return float(value)


def _safe_mutual_information(scores, df_numeric: pd.DataFrame) -> dict[str, float]:
    x = df_numeric.fillna(0.0).to_numpy(dtype=float)
    y = pd.Series(scores).fillna(0.0).to_numpy(dtype=float)
    try:
        mi = mutual_info_regression(x, y, random_state=42)
    except Exception:
        mi = np.zeros(len(df_numeric.columns), dtype=float)
    return {column: float(max(value, 0.0)) for column, value in zip(df_numeric.columns, mi)}


def _safe_permutation_importance(
    scores,
    df_numeric: pd.DataFrame,
    sample_size_cap: int = 5000,
    n_estimators: int = 80,
    max_depth: int = 8,
    n_repeats: int = 5,
) -> dict[str, float]:
    sample_size = min(len(df_numeric), sample_size_cap)
    sample_df = df_numeric.sample(n=sample_size, random_state=42) if len(df_numeric) > sample_size else df_numeric
    y = pd.Series(scores, index=df_numeric.index).loc[sample_df.index].fillna(0.0).to_numpy(dtype=float)
    x = sample_df.fillna(0.0).to_numpy(dtype=float)

    if np.allclose(y, y[0]):
        return {column: 0.0 for column in df_numeric.columns}

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x, y)
    permutation = permutation_importance(
        model,
        x,
        y,
        n_repeats=n_repeats,
        random_state=42,
        n_jobs=-1,
        scoring="neg_mean_squared_error",
    )
    importances = np.maximum(permutation.importances_mean, 0.0)
    return {column: float(value) for column, value in zip(df_numeric.columns, importances)}


def _build_metric_scores(
    scores,
    df_numeric: pd.DataFrame,
    *,
    perm_sample_size_cap: int = 5000,
    perm_n_estimators: int = 80,
    perm_max_depth: int = 8,
    perm_n_repeats: int = 5,
) -> dict[str, dict[str, float]]:
    score_series = pd.Series(scores)

    spearman = {}
    kendall = {}
    for column in df_numeric.columns:
        spearman[column] = abs(_safe_spearman(score_series, df_numeric[column]))
        kendall[column] = abs(_safe_kendall(score_series, df_numeric[column]))

    mutual_info = _safe_mutual_information(scores, df_numeric)
    perm_importance = _safe_permutation_importance(
        scores,
        df_numeric,
        sample_size_cap=perm_sample_size_cap,
        n_estimators=perm_n_estimators,
        max_depth=perm_max_depth,
        n_repeats=perm_n_repeats,
    )
    return {
        "spearman": spearman,
        "kendall": kendall,
        "mutual_info": mutual_info,
        "permutation_importance": perm_importance,
    }


def _normalize_weights(metric_weights: dict[str, float] | None) -> dict[str, float]:
    weights = dict(DEFAULT_METRIC_WEIGHTS)
    if metric_weights is not None:
        for key, value in metric_weights.items():
            if key in METRIC_KEYS:
                weights[key] = max(float(value), 0.0)

    total = sum(weights.values())
    if total <= 0:
        return dict(DEFAULT_METRIC_WEIGHTS)
    return {key: value / total for key, value in weights.items()}


def _metric_ranks(metric_scores: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    ranks: dict[str, dict[str, float]] = {}
    for metric_name, values in metric_scores.items():
        rank_series = pd.Series(values, dtype=float).rank(ascending=False, method="average")
        ranks[metric_name] = {feature: float(rank) for feature, rank in rank_series.items()}
    return ranks


def _rank_score(rank_value: float, n_features: int) -> float:
    if n_features <= 1:
        return 1.0
    return 1.0 - (rank_value - 1.0) / (n_features - 1.0)


def _bootstrap_rank_stability(
    scores,
    df_numeric: pd.DataFrame,
    *,
    metric_weights: dict[str, float],
    n_bootstrap: int,
    top_k: int,
    random_state: int,
    bootstrap_sample_size: int,
) -> dict[str, dict[str, float]]:
    features = list(df_numeric.columns)
    if n_bootstrap <= 0:
        return {
            feature: {
                "top_k_frequency": 0,
                "top_k_probability": 0.0,
                "median_rank": 0.0,
                "iqr_rank": 0.0,
            }
            for feature in features
        }

    rng = np.random.default_rng(seed=random_state)
    n = len(df_numeric)
    draw_size = min(max(1, bootstrap_sample_size), n)
    feature_rank_history = {feature: [] for feature in features}
    top_k_hits = {feature: 0 for feature in features}

    score_array = np.asarray(scores, dtype=float)
    for _ in range(n_bootstrap):
        sampled_idx = rng.integers(low=0, high=n, size=draw_size)
        boot_df = df_numeric.iloc[sampled_idx].reset_index(drop=True)
        boot_scores = score_array[sampled_idx]

        boot_metrics = _build_metric_scores(
            boot_scores,
            boot_df,
            perm_sample_size_cap=min(1500, draw_size),
            perm_n_estimators=30,
            perm_max_depth=6,
            perm_n_repeats=2,
        )
        boot_ranks = _metric_ranks(boot_metrics)

        aggregated = []
        for feature in features:
            combined_rank_score = 0.0
            for metric_name in METRIC_KEYS:
                combined_rank_score += metric_weights[metric_name] * _rank_score(boot_ranks[metric_name][feature], len(features))
            aggregated.append((feature, combined_rank_score))
        aggregated.sort(key=lambda item: item[1], reverse=True)

        for rank_position, (feature, _) in enumerate(aggregated, start=1):
            feature_rank_history[feature].append(rank_position)
            if rank_position <= top_k:
                top_k_hits[feature] += 1

    stability = {}
    for feature in features:
        rank_values = np.array(feature_rank_history[feature], dtype=float)
        stability[feature] = {
            "top_k_frequency": int(top_k_hits[feature]),
            "top_k_probability": float(top_k_hits[feature] / n_bootstrap),
            "median_rank": float(np.median(rank_values)),
            "iqr_rank": float(np.percentile(rank_values, 75) - np.percentile(rank_values, 25)),
        }
    return stability


def rank_features_for_scores(
    scores,
    df_numeric: pd.DataFrame,
    top_n: int = TOP_FEATURES_COUNT,
    *,
    metric_weights: dict[str, float] | None = None,
    bootstrap_repeats: int = 0,
    bootstrap_sample_size: int = 3000,
    random_state: int = 42,
) -> list[dict[str, float | str]]:
    weights = _normalize_weights(metric_weights)
    metric_scores = _build_metric_scores(scores, df_numeric)
    metric_ranks = _metric_ranks(metric_scores)
    stability_stats = _bootstrap_rank_stability(
        scores,
        df_numeric,
        metric_weights=weights,
        n_bootstrap=bootstrap_repeats,
        top_k=top_n,
        random_state=random_state,
        bootstrap_sample_size=bootstrap_sample_size,
    )

    n_features = len(df_numeric.columns)

    rows = []
    for feature in df_numeric.columns:
        rank_spearman = metric_ranks["spearman"][feature]
        rank_kendall = metric_ranks["kendall"][feature]
        rank_mi = metric_ranks["mutual_info"][feature]
        rank_perm = metric_ranks["permutation_importance"][feature]

        combined = (
            weights["spearman"] * _rank_score(rank_spearman, n_features)
            + weights["kendall"] * _rank_score(rank_kendall, n_features)
            + weights["mutual_info"] * _rank_score(rank_mi, n_features)
            + weights["permutation_importance"] * _rank_score(rank_perm, n_features)
        )
        stability = stability_stats[feature]
        rows.append(
            {
                "feature": feature,
                "spearman": round(metric_scores["spearman"][feature], 6),
                "kendall": round(metric_scores["kendall"][feature], 6),
                "mutual_info": round(metric_scores["mutual_info"][feature], 6),
                "permutation_importance": round(metric_scores["permutation_importance"][feature], 6),
                "rank_spearman": round(rank_spearman, 3),
                "rank_kendall": round(rank_kendall, 3),
                "rank_mutual_info": round(rank_mi, 3),
                "rank_permutation_importance": round(rank_perm, 3),
                "combined_impact": float(combined),
                "top_k_frequency": int(stability["top_k_frequency"]),
                "top_k_probability": round(float(stability["top_k_probability"]), 6),
                "median_rank": round(float(stability["median_rank"]), 3),
                "iqr_rank": round(float(stability["iqr_rank"]), 3),
            }
        )

    rows.sort(key=lambda item: item["combined_impact"], reverse=True)
    return rows[:top_n]



def _build_row(
    name: str,
    architecture: str,
    mse: float,
    scores,
    df_numeric: pd.DataFrame,
    notes: str,
    *,
    metric_weights: dict[str, float] | None = None,
    bootstrap_repeats: int = 0,
    bootstrap_sample_size: int = 3000,
    random_state: int = 42,
) -> dict[str, object]:
    top_features = rank_features_for_scores(
        scores,
        df_numeric,
        top_n=TOP_FEATURES_COUNT,
        metric_weights=metric_weights,
        bootstrap_repeats=bootstrap_repeats,
        bootstrap_sample_size=bootstrap_sample_size,
        random_state=random_state,
    )
    latent_score_quality = sum(feature["combined_impact"] for feature in top_features) / max(len(top_features), 1)

    row = {
        "Model": name,
        "Architecture": architecture,
        "Reconstruction MSE": round(float(mse), 6),
        "Latent Score Quality": round(latent_score_quality, 6),
        "Notes": notes,
    }

    for index, feature in enumerate(top_features, start=1):
        row[f"Top {index} Feature"] = feature["feature"]
        row[f"Top {index} Spearman"] = feature["spearman"]
        row[f"Top {index} Kendall"] = feature["kendall"]
        row[f"Top {index} Mutual Info"] = feature["mutual_info"]
        row[f"Top {index} Permutation Importance"] = feature["permutation_importance"]
        row[f"Top {index} Median Rank"] = feature["median_rank"]
        row[f"Top {index} Rank IQR"] = feature["iqr_rank"]
        row[f"Top {index} P(Top3)"] = feature["top_k_probability"]
        row[f"Top {index} Combined Impact"] = round(float(feature["combined_impact"]), 6)

    return row


def build_comparison_table(
    autoencoder_results,
    pca_result,
    vae_result,
    df_numeric: pd.DataFrame,
    *,
    metric_weights: dict[str, float] | None = None,
    bootstrap_repeats: int = 0,
    bootstrap_sample_size: int = 3000,
    random_state: int = 42,
) -> pd.DataFrame:
    rows = []
    for result in autoencoder_results:
        rows.append(
            _build_row(
                name=result.name,
                architecture=result.architecture,
                mse=result.mse,
                scores=result.scores,
                df_numeric=df_numeric,
                notes="Autoencoder bottleneck score",
                metric_weights=metric_weights,
                bootstrap_repeats=bootstrap_repeats,
                bootstrap_sample_size=bootstrap_sample_size,
                random_state=random_state,
            )
        )

    rows.append(
        _build_row(
            name=pca_result.name,
            architecture=pca_result.architecture,
            mse=pca_result.mse,
            scores=pca_result.scores,
            df_numeric=df_numeric,
            notes=f"Explained variance={pca_result.explained_variance_ratio:.6f}",
            metric_weights=metric_weights,
            bootstrap_repeats=bootstrap_repeats,
            bootstrap_sample_size=bootstrap_sample_size,
            random_state=random_state,
        )
    )

    rows.append(
        _build_row(
            name=vae_result.name,
            architecture=vae_result.architecture,
            mse=vae_result.mse,
            scores=vae_result.scores,
            df_numeric=df_numeric,
            notes=f"KL proxy={vae_result.kl_loss:.6f}",
            metric_weights=metric_weights,
            bootstrap_repeats=bootstrap_repeats,
            bootstrap_sample_size=bootstrap_sample_size,
            random_state=random_state,
        )
    )

    return pd.DataFrame(rows)


def _build_metricwise_rows(
    name: str,
    architecture: str,
    scores,
    df_numeric: pd.DataFrame,
    top_n: int,
) -> list[dict[str, object]]:
    metric_scores = _build_metric_scores(scores, df_numeric)
    rows: list[dict[str, object]] = []

    for metric_name in METRIC_KEYS:
        sorted_features = sorted(
            metric_scores[metric_name].items(),
            key=lambda item: item[1],
            reverse=True,
        )
        for position, (feature, score_value) in enumerate(sorted_features[:top_n], start=1):
            rows.append(
                {
                    "Model": name,
                    "Architecture": architecture,
                    "Metric": metric_name,
                    "Rank": position,
                    "Feature": feature,
                    "Metric Score": round(float(score_value), 6),
                }
            )
    return rows


def build_metricwise_table(
    autoencoder_results,
    pca_result,
    vae_result,
    df_numeric: pd.DataFrame,
    *,
    top_n: int = TOP_FEATURES_COUNT,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for result in autoencoder_results:
        rows.extend(
            _build_metricwise_rows(
                name=result.name,
                architecture=result.architecture,
                scores=result.scores,
                df_numeric=df_numeric,
                top_n=top_n,
            )
        )

    rows.extend(
        _build_metricwise_rows(
            name=pca_result.name,
            architecture=pca_result.architecture,
            scores=pca_result.scores,
            df_numeric=df_numeric,
            top_n=top_n,
        )
    )
    rows.extend(
        _build_metricwise_rows(
            name=vae_result.name,
            architecture=vae_result.architecture,
            scores=vae_result.scores,
            df_numeric=df_numeric,
            top_n=top_n,
        )
    )

    return pd.DataFrame(rows)


def _build_metric_agreement_rows(
    name: str,
    architecture: str,
    scores,
    df_numeric: pd.DataFrame,
) -> list[dict[str, object]]:
    metric_scores = _build_metric_scores(scores, df_numeric)
    rank_vectors = {
        metric_name: pd.Series(values, dtype=float).rank(ascending=False, method="average")
        for metric_name, values in metric_scores.items()
    }

    rows: list[dict[str, object]] = []
    for metric_a in METRIC_KEYS:
        for metric_b in METRIC_KEYS:
            corr = rank_vectors[metric_a].corr(rank_vectors[metric_b], method="spearman")
            rows.append(
                {
                    "Model": name,
                    "Architecture": architecture,
                    "Metric A": metric_a,
                    "Metric B": metric_b,
                    "Rank Correlation": 0.0 if pd.isna(corr) else round(float(corr), 6),
                }
            )
    return rows


def build_metric_agreement_table(
    autoencoder_results,
    pca_result,
    vae_result,
    df_numeric: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for result in autoencoder_results:
        rows.extend(
            _build_metric_agreement_rows(
                name=result.name,
                architecture=result.architecture,
                scores=result.scores,
                df_numeric=df_numeric,
            )
        )

    rows.extend(
        _build_metric_agreement_rows(
            name=pca_result.name,
            architecture=pca_result.architecture,
            scores=pca_result.scores,
            df_numeric=df_numeric,
        )
    )
    rows.extend(
        _build_metric_agreement_rows(
            name=vae_result.name,
            architecture=vae_result.architecture,
            scores=vae_result.scores,
            df_numeric=df_numeric,
        )
    )

    return pd.DataFrame(rows)


def save_comparison_table(df: pd.DataFrame, output_path: Path = DEFAULT_METRICS_PATH) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def save_metricwise_table(df: pd.DataFrame, output_path: Path = DEFAULT_METRICWISE_PATH) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def save_metric_agreement_table(df: pd.DataFrame, output_path: Path = DEFAULT_METRIC_AGREEMENT_PATH) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path
