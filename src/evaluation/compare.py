from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_METRICS_PATH = ROOT_DIR / "outputs" / "metrics" / "model_comparison.csv"


def _safe_spearman(left: pd.Series, right: pd.Series) -> float:
    value = left.corr(right, method="spearman")
    if pd.isna(value):
        return 0.0
    return float(value)


def _build_row(name: str, architecture: str, mse: float, scores, df_numeric: pd.DataFrame, notes: str) -> dict[str, object]:
    pace_corr = _safe_spearman(pd.Series(scores), df_numeric["pace_min_km"])
    hr_corr = _safe_spearman(pd.Series(scores), df_numeric["average_hr"])
    latent_score_quality = (abs(pace_corr) + abs(hr_corr)) / 2

    return {
        "Model": name,
        "Architecture": architecture,
        "Reconstruction MSE": round(float(mse), 6),
        "Latent Score vs Pace (Spearman)": round(pace_corr, 6),
        "Latent Score vs HR (Spearman)": round(hr_corr, 6),
        "Latent Score Quality": round(latent_score_quality, 6),
        "Notes": notes,
    }


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
