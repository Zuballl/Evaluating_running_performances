from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_METRICS_PATH = ROOT_DIR / "outputs" / "metrics" / "model_comparison.csv"
TOP_FEATURES_COUNT = 3


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


def _safe_permutation_importance(scores, df_numeric: pd.DataFrame) -> dict[str, float]:
    sample_size = min(len(df_numeric), 5000)
    sample_df = df_numeric.sample(n=sample_size, random_state=42) if len(df_numeric) > sample_size else df_numeric
    y = pd.Series(scores, index=df_numeric.index).loc[sample_df.index].fillna(0.0).to_numpy(dtype=float)
    x = sample_df.fillna(0.0).to_numpy(dtype=float)

    if np.allclose(y, y[0]):
        return {column: 0.0 for column in df_numeric.columns}

    model = RandomForestRegressor(
        n_estimators=80,
        max_depth=8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x, y)
    permutation = permutation_importance(
        model,
        x,
        y,
        n_repeats=5,
        random_state=42,
        n_jobs=-1,
        scoring="neg_mean_squared_error",
    )
    importances = np.maximum(permutation.importances_mean, 0.0)
    return {column: float(value) for column, value in zip(df_numeric.columns, importances)}


def _normalize_metric(values: dict[str, float]) -> dict[str, float]:
    max_value = max(values.values()) if values else 0.0
    if max_value <= 0:
        return {key: 0.0 for key in values}
    return {key: float(value / max_value) for key, value in values.items()}


def rank_features_for_scores(scores, df_numeric: pd.DataFrame, top_n: int = TOP_FEATURES_COUNT) -> list[dict[str, float | str]]:
    score_series = pd.Series(scores)

    spearman = {}
    kendall = {}
    for column in df_numeric.columns:
        spearman[column] = abs(_safe_spearman(score_series, df_numeric[column]))
        kendall[column] = abs(_safe_kendall(score_series, df_numeric[column]))

    mutual_info = _safe_mutual_information(scores, df_numeric)
    perm_importance = _safe_permutation_importance(scores, df_numeric)

    spearman_norm = _normalize_metric(spearman)
    kendall_norm = _normalize_metric(kendall)
    mi_norm = _normalize_metric(mutual_info)
    perm_norm = _normalize_metric(perm_importance)

    rows = []
    for feature in df_numeric.columns:
        combined = (spearman_norm[feature] + kendall_norm[feature] + mi_norm[feature] + perm_norm[feature]) / 4
        rows.append(
            {
                "feature": feature,
                "spearman": round(spearman[feature], 6),
                "kendall": round(kendall[feature], 6),
                "mutual_info": round(mutual_info[feature], 6),
                "permutation_importance": round(perm_importance[feature], 6),
                "combined_impact": float(combined),
            }
        )

    rows.sort(key=lambda item: item["combined_impact"], reverse=True)
    return rows[:top_n]



def _build_row(name: str, architecture: str, mse: float, scores, df_numeric: pd.DataFrame, notes: str) -> dict[str, object]:
    top_features = rank_features_for_scores(scores, df_numeric, top_n=TOP_FEATURES_COUNT)
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
        row[f"Top {index} Combined Impact"] = round(float(feature["combined_impact"]), 6)

    return row


def build_comparison_table(autoencoder_results, pca_result, vae_result, df_numeric: pd.DataFrame) -> pd.DataFrame:
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
        )
    )

    return pd.DataFrame(rows)


def save_comparison_table(df: pd.DataFrame, output_path: Path = DEFAULT_METRICS_PATH) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path
