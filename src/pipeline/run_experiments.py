import argparse
from pathlib import Path
import random

import numpy as np
import torch

from sklearn.model_selection import train_test_split

from src.data.read_data import PROCESSED_DATA_PATH, load_clean_numeric_data, prepare_clean_data
from src.evaluation.compare import (
    build_comparison_table,
    build_metric_agreement_table,
    build_metricwise_table,
    save_comparison_table,
    save_metric_agreement_table,
    save_metricwise_table,
    rank_features_for_scores,
)
from src.models.autoencoder_models import run_autoencoder_comparison
from src.models.pca_model import run_pca
from src.models.vae_model import run_vae
import pandas as pd
from src.visualization.plots import (
    plot_metric_agreement_heatmaps,
    plot_metricwise_consensus,
    plot_metricwise_top3_heatmaps,
    plot_model_agreement,
    plot_mse_comparison,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run performance-score experiments.")
    parser.add_argument("--sample-size", type=int, default=1000000, help="Number of rows used for model experiments.")
    parser.add_argument("--skip-cleaning", action="store_true", help="Use existing clean dataset instead of regenerating it.")
    parser.add_argument("--with-plots", action="store_true", help="Generate plots for autoencoder comparison.")
    parser.add_argument("--ae-epochs", type=int, default=20, help="Training epochs for each autoencoder variant.")
    parser.add_argument("--ae-batch-size", type=int, default=2048, help="Batch size for autoencoder training.")
    parser.add_argument("--ae-patience", type=int, default=3, help="Early stopping patience for autoencoders.")
    parser.add_argument("--vae-epochs", type=int, default=50, help="Training epochs for VAE.")
    parser.add_argument("--vae-batch-size", type=int, default=32, help="Batch size for VAE training.")
    parser.add_argument("--vae-patience", type=int, default=5, help="Early stopping patience for VAE.")
    parser.add_argument("--bootstrap-repeats", type=int, default=30, help="Bootstrap repeats for feature-ranking stability.")
    parser.add_argument("--bootstrap-sample-size", type=int, default=3000, help="Sample size used in each bootstrap repeat.")
    parser.add_argument("--weight-spearman", type=float, default=0.25, help="Weight for Spearman rank contribution.")
    parser.add_argument("--weight-kendall", type=float, default=0.2, help="Weight for Kendall rank contribution.")
    parser.add_argument("--weight-mi", type=float, default=0.25, help="Weight for Mutual Information rank contribution.")
    parser.add_argument("--weight-perm", type=float, default=0.3, help="Weight for permutation-importance rank contribution.")
    parser.add_argument("--best-model", type=str, default="auto", choices=["auto", "simple", "medium", "deep"], help="Autoencoder model to use for athlete profiling: auto=select by MSE (default), simple/medium/deep=specific model.")
    parser.add_argument("--all-ae-profiles", action="store_true", help="Generate profile plots for all 3 autoencoder variants (instead of just best model).")
    parser.add_argument("--random-seed", type=int, default=42, help="Global random seed for reproducible runs.")
    return parser.parse_args()


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def select_best_autoencoder(autoencoder_results, best_model_arg="auto"):
    """
    Select which autoencoder to use for athlete profiling.
    
    Args:
        autoencoder_results: list of [simple, medium, deep] AutoencoderResult
        best_model_arg: "auto"=select by MSE (lowest), "simple"/"medium"/"deep"=explicit choice
    
    Returns:
        (selected_result, model_name, selected_idx)
    """
    if best_model_arg == "auto":
        # Select based on lowest MSE
        best_idx = min(range(len(autoencoder_results)), 
                      key=lambda i: autoencoder_results[i].mse)
        selected = autoencoder_results[best_idx]
        print(f"\n🤖 Selected {selected.name} (MSE={selected.mse:.6f}) - lowest reconstruction error")
        return selected, selected.name, best_idx
    else:
        # Explicit model selected
        idx = {"simple": 0, "medium": 1, "deep": 2}[best_model_arg]
        selected = autoencoder_results[idx]
        print(f"\n🤖 Using explicitly selected {selected.name} (MSE={selected.mse:.6f})")
        return selected, selected.name, idx

def get_extreme_athletes(
    df_numeric,
    scores,
    *,
    metric_weights=None,
    bootstrap_repeats: int = 0,
    bootstrap_sample_size: int = 3000,
    random_state: int = 42,
):
    """
    Extracts top 3 and bottom 3 athletes based on scores.
    Returns both the extreme athletes DataFrame and the top 3 features.

    Feature ranking is based on weighted rank aggregation with optional bootstrap stability.
    """
    # Dodajemy wyniki do kopii danych, aby móc je posortować
    df_with_scores = df_numeric.copy()
    df_with_scores["performance_score"] = scores

    # Sortujemy: od najlepszych do najgorszych
    df_sorted = df_with_scores.sort_values(by="performance_score", ascending=False)

    # Wybieramy Top 3 i Bottom 3
    top_3 = df_sorted.head(3).copy()
    bottom_3 = df_sorted.tail(3).copy()

    # Nadajemy etykiety dla wykresu
    top_3["Athlete Label"] = [f"Lider {i+1}" for i in range(3)]
    bottom_3["Athlete Label"] = [f"Outsider {i+1}" for i in reversed(range(3))]

    # Łączymy w jeden DataFrame do wykresu
    extreme_df = pd.concat([top_3, bottom_3])

    # Obliczamy TOP 3 features na podstawie agregacji rang z opcjonalnym bootstrapem
    top_3_features = rank_features_for_scores(
        scores,
        df_numeric,
        top_n=3,
        metric_weights=metric_weights,
        bootstrap_repeats=bootstrap_repeats,
        bootstrap_sample_size=bootstrap_sample_size,
        random_state=random_state,
    )
    top_3_feature_names = [feature["feature"] for feature in top_3_features]

    print("\nTOP 3 features by aggregated rank impact:")
    for feature in top_3_features:
        print(
            f"  {feature['feature']}: combined={feature['combined_impact']:.6f}, "
            f"spearman={feature['spearman']:.6f}, kendall={feature['kendall']:.6f}, "
            f"mi={feature['mutual_info']:.6f}, perm={feature['permutation_importance']:.6f}, "
            f"p_top3={feature['top_k_probability']:.3f}, median_rank={feature['median_rank']:.2f}, iqr={feature['iqr_rank']:.2f}"
        )

    # Mapujemy nazwy kolumn
    column_mapping = {
        "pace_min_km": "Pace [min/km]",
        "average_hr": "Average Heart Rate [bpm]",
        "elevation_gain": "Elevation Gain [m]",
        "total_distance": "Total Distance [km]",
        "final_cadence": "Final Cadence [spm]",
        "aerobic_decoupling": "Aerobic Decoupling [%]",
        "age": "Age [years]",
        "athlete_weight": "Athlete Weight [kg]",
    }

    renamed_extreme_df = extreme_df.rename(columns=column_mapping)
    
    # Note: df_numeric is already in original units (no denormalization needed)
    # The scaler is used internally for model training, not for visualization
    
    mapped_features = [column_mapping.get(feat, feat) for feat in top_3_feature_names]
    
    return renamed_extreme_df, mapped_features


def get_top_features_per_model(
    model_results,
    df_numeric,
    *,
    metric_weights=None,
    bootstrap_repeats: int = 0,
    bootstrap_sample_size: int = 3000,
    random_state: int = 42,
):
    """
    Oblicza TOP 3 features dla każdego modelu na podstawie agregacji rang.

    Returns:
        dict: {model_name -> [(feature_name, combined_impact), ...]}
    """
    top_features_per_model = {}

    for result in model_results:
        top_features = rank_features_for_scores(
            result.scores,
            df_numeric,
            top_n=3,
            metric_weights=metric_weights,
            bootstrap_repeats=bootstrap_repeats,
            bootstrap_sample_size=bootstrap_sample_size,
            random_state=random_state,
        )
        top_3 = [(feature["feature"], float(feature["combined_impact"])) for feature in top_features]
        top_features_per_model[result.name] = top_3

    return top_features_per_model

def main():
    args = parse_args()
    set_global_seed(args.random_seed)
    metric_weights = {
        "spearman": args.weight_spearman,
        "kendall": args.weight_kendall,
        "mutual_info": args.weight_mi,
        "permutation_importance": args.weight_perm,
    }

    if not args.skip_cleaning or not PROCESSED_DATA_PATH.exists():
        prepare_clean_data()

    df_numeric = load_clean_numeric_data(PROCESSED_DATA_PATH, sample_size=args.sample_size)

    df_train, df_test = train_test_split(df_numeric, test_size=0.2, random_state=42)
    print(f"Train/test split: {len(df_train)} train rows, {len(df_test)} test rows")

    print("running autoencoder comparison...")
    autoencoder_results, _, _ = run_autoencoder_comparison(
        df_train,
        df_test,
        df_numeric,
        ae_epochs=args.ae_epochs,
        ae_batch_size=args.ae_batch_size,
        ae_patience=args.ae_patience,
    )
    print("Autoencoder comparison complete.")

    pca_result, _ = run_pca(df_train, df_test, df_numeric)
    print(f"PCA complete. Scores shape: {pca_result.scores.shape}")

    vae_result = run_vae(
        df_train,
        df_test,
        df_numeric,
        epochs=args.vae_epochs,
        batch_size=args.vae_batch_size,
        patience=args.vae_patience,
        verbose=True,
    )
    print(f"VAE complete. Scores shape: {vae_result.scores.shape}")

    comparison_df = build_comparison_table(
        autoencoder_results,
        pca_result,
        vae_result,
        df_numeric,
        metric_weights=metric_weights,
        bootstrap_repeats=args.bootstrap_repeats,
        bootstrap_sample_size=args.bootstrap_sample_size,
        random_state=args.random_seed,
    )
    comparison_path = save_comparison_table(comparison_df)
    metricwise_df = build_metricwise_table(
        autoencoder_results,
        pca_result,
        vae_result,
        df_numeric,
    )
    metricwise_path = save_metricwise_table(metricwise_df)
    metricwise_heatmap_plot = plot_metricwise_top3_heatmaps(metricwise_df)
    metricwise_consensus_plot = plot_metricwise_consensus(metricwise_df)
    metric_agreement_df = build_metric_agreement_table(
        autoencoder_results,
        pca_result,
        vae_result,
        df_numeric,
    )
    metric_agreement_path = save_metric_agreement_table(metric_agreement_df)
    metric_agreement_plot = plot_metric_agreement_heatmaps(metric_agreement_df)

    # 1. Wybieramy najlepszy model (auto/simple/medium/deep)
    best_model_results, best_model_name, _ = select_best_autoencoder(
        autoencoder_results, 
        best_model_arg=args.best_model
    )
    
    # 2. Generujemy profile plots
    from src.visualization.plots import plot_athlete_profiles
    profile_plots = []
    
    if args.all_ae_profiles:
        # Generujemy dla wszystkich 3 modeli AE
        print("\nGenerowanie profile plots dla wszystkich 3 wariantów autoencoder'ów...")
        for ae_result in autoencoder_results:
            extreme_athletes_df, top_features = get_extreme_athletes(
                df_numeric,
                ae_result.scores,
                metric_weights=metric_weights,
                bootstrap_repeats=args.bootstrap_repeats,
                bootstrap_sample_size=args.bootstrap_sample_size,
                random_state=args.random_seed,
            )
            plot_path = plot_athlete_profiles(
                extreme_athletes_df, 
                features_to_plot=top_features,
                output_path=Path(__file__).parent.parent.parent / "outputs" / "plots" / f"athlete_profiles_{ae_result.name}.png"
            )
            profile_plots.append(plot_path)
            print(f"  ✓ {ae_result.name}: {plot_path}")
    else:
        # Tylko najlepszy model
        print(f"\nIdentyfikacja liderów i outsiderów (model: {best_model_name})...")
        extreme_athletes_df, top_features = get_extreme_athletes(
            df_numeric,
            best_model_results.scores,
            metric_weights=metric_weights,
            bootstrap_repeats=args.bootstrap_repeats,
            bootstrap_sample_size=args.bootstrap_sample_size,
            random_state=args.random_seed,
        )
        profile_plot = plot_athlete_profiles(extreme_athletes_df, features_to_plot=top_features)
        profile_plots.append(profile_plot)
    
    # Reszta Twoich wykresów...
    model_results = [*autoencoder_results, pca_result, vae_result]
    mse_plot = plot_mse_comparison(model_results)
    
    # Nowy wykres z TOP features per model
    top_features_dict = get_top_features_per_model(
        model_results,
        df_numeric,
        metric_weights=metric_weights,
        bootstrap_repeats=args.bootstrap_repeats,
        bootstrap_sample_size=args.bootstrap_sample_size,
        random_state=args.random_seed,
    )
    from src.visualization.plots import plot_top_features_per_model
    top_features_plot = plot_top_features_per_model(top_features_dict)
    
    model_scores = {
        autoencoder_results[0].name: autoencoder_results[0].scores,
        autoencoder_results[1].name: autoencoder_results[1].scores,
        autoencoder_results[2].name: autoencoder_results[2].scores,
        pca_result.name: pca_result.scores,
        vae_result.name: vae_result.scores,
    }
    agreement_plot = plot_model_agreement(model_scores)
    
    profile_plots_str = ", ".join(str(p) for p in profile_plots)
    print(
        "\n✅ Wszystkie wizualizacje gotowe: \n"
        f"  Profile plots: {profile_plots_str}\n"
        f"  MSE comparison: {mse_plot}\n"
        f"  Top features: {top_features_plot}\n"
        f"  Model agreement: {agreement_plot}\n"
        f"  Metric-wise heatmaps: {metricwise_heatmap_plot}\n"
        f"  Metric-wise consensus: {metricwise_consensus_plot}\n"
        f"  Metric agreement matrix: {metric_agreement_plot}"
    )

    print(f"\nSaved comparison table: {comparison_path}")
    print(f"Saved metric-wise ranking table: {metricwise_path}")
    print(f"Saved metric-agreement table: {metric_agreement_path}")
    print(comparison_df.to_markdown(index=False))


if __name__ == "__main__":
    main()
