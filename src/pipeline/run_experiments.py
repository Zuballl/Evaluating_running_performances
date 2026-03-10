import argparse
from pathlib import Path

from src.data.read_data import PROCESSED_DATA_PATH, load_clean_numeric_data, prepare_clean_data
from src.evaluation.compare import build_comparison_table, save_comparison_table
from src.models.autoencoder_models import run_autoencoder_comparison
from src.models.tsne_model import run_tsne
from src.models.vae_model import run_vae
from src.visualization.plots import plot_model_agreement, plot_mse_comparison


def parse_args():
    parser = argparse.ArgumentParser(description="Run performance-score experiments.")
    parser.add_argument("--sample-size", type=int, default=10000, help="Number of rows used for model experiments.")
    parser.add_argument("--skip-cleaning", action="store_true", help="Use existing clean dataset instead of regenerating it.")
    parser.add_argument("--run-vae", action="store_true", help="Run VAE model.")
    parser.add_argument("--run-tsne", action="store_true", help="Run t-SNE model.")
    parser.add_argument("--with-plots", action="store_true", help="Generate plots for autoencoder comparison.")
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.skip_cleaning or not PROCESSED_DATA_PATH.exists():
        prepare_clean_data()

    df_numeric = load_clean_numeric_data(PROCESSED_DATA_PATH, sample_size=args.sample_size)

    autoencoder_results, _ = run_autoencoder_comparison(df_numeric)

    if args.run_vae:
        vae_scores = run_vae(df_numeric)
        print(f"VAE complete. Scores shape: {vae_scores.shape}")

    if args.run_tsne:
        tsne_df = run_tsne(df_numeric)
        tsne_output = Path("outputs/metrics/tsne_scores.csv")
        tsne_output.parent.mkdir(parents=True, exist_ok=True)
        tsne_df[["tsne_score", "x", "y"]].to_csv(tsne_output, index=False)
        print(f"t-SNE complete. Saved: {tsne_output}")

    comparison_df = build_comparison_table(autoencoder_results, include_vae=args.run_vae, include_tsne=args.run_tsne)
    comparison_path = save_comparison_table(comparison_df)

    if args.with_plots:
        simple_scores = autoencoder_results[0].scores
        deep_scores = autoencoder_results[2].scores
        mse_plot = plot_mse_comparison(autoencoder_results)
        agreement_plot = plot_model_agreement(simple_scores, deep_scores)
        print(f"Saved plots: {mse_plot}, {agreement_plot}")

    print(f"Saved comparison table: {comparison_path}")
    print(comparison_df.to_markdown(index=False))


if __name__ == "__main__":
    main()
