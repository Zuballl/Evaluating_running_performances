from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


ROOT_DIR = Path(__file__).resolve().parents[2]
PLOTS_DIR = ROOT_DIR / "outputs" / "plots"


sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 300


def plot_mse_comparison(autoencoder_results, output_path: Path | None = None) -> Path:
    if output_path is None:
        output_path = PLOTS_DIR / "mse_comparison.png"

    model_names = [result.name for result in autoencoder_results]
    mse_values = [result.mse for result in autoencoder_results]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=model_names, y=mse_values, hue=model_names, palette="viridis", legend=False)
    plt.title("Porównanie Błędu Rekonstrukcji (MSE)")
    plt.ylabel("Mean Squared Error")

    for index, value in enumerate(mse_values):
        ax.text(index, value + (max(mse_values) * 0.01), f"{value:.6f}", ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def plot_model_agreement(simple_scores, deep_scores, output_path: Path | None = None) -> Path:
    if output_path is None:
        output_path = PLOTS_DIR / "model_agreement.png"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 8))
    plt.scatter(simple_scores, deep_scores, alpha=0.4, c=deep_scores, cmap="coolwarm", edgecolors="w", linewidth=0.5)
    plt.title("Korelacja Wyników: Simple vs Deep Autoencoder")
    plt.xlabel("Performance Score (Simple)")
    plt.ylabel("Performance Score (Deep)")
    plt.colorbar(label="Skala Performance Score")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path
