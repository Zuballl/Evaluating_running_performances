from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_METRICS_PATH = ROOT_DIR / "outputs" / "metrics" / "model_comparison.csv"


def build_comparison_table(autoencoder_results, include_vae: bool, include_tsne: bool) -> pd.DataFrame:
    rows = []
    for result in autoencoder_results:
        rows.append(
            {
                "Model Approach": result.name,
                "Architecture": result.architecture,
                "MSE / Metric": f"{result.mse:.6f}",
                "Best For": "Autoencoder ranking",
            }
        )

    if include_vae:
        rows.append(
            {
                "Model Approach": "vae",
                "Architecture": "Probabilistic latent space",
                "MSE / Metric": "N/A (KL + recon)",
                "Best For": "Smooth latent distribution",
            }
        )

    if include_tsne:
        rows.append(
            {
                "Model Approach": "tsne",
                "Architecture": "Manifold learning",
                "MSE / Metric": "N/A",
                "Best For": "Cluster visualization",
            }
        )

    return pd.DataFrame(rows)


def save_comparison_table(df: pd.DataFrame, output_path: Path = DEFAULT_METRICS_PATH) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path
