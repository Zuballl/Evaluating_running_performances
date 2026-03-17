import argparse
from pathlib import Path

from sklearn.model_selection import train_test_split

from src.data.read_data import PROCESSED_DATA_PATH, load_clean_numeric_data, prepare_clean_data
from src.evaluation.compare import build_comparison_table, save_comparison_table, rank_features_for_scores
from src.models.autoencoder_models import run_autoencoder_comparison
from src.models.pca_model import run_pca
from src.models.vae_model import run_vae
import pandas as pd
from src.visualization.plots import plot_latent_score_comparison, plot_model_agreement, plot_mse_comparison


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
    return parser.parse_args()

def get_extreme_athletes(df_numeric, scores):
    """
    Extracts top 3 and bottom 3 athletes based on scores.
    Returns both the extreme athletes DataFrame and the top 3 features by correlation.
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

    # Obliczamy TOP 3 features na podstawie łączonego wpływu (Spearman + Kendall + MI + permutation importance)
    top_3_features = rank_features_for_scores(scores, df_numeric, top_n=3)
    top_3_feature_names = [feature["feature"] for feature in top_3_features]
    
    print("\nTOP 3 features by combined impact on scores:")
    for feature in top_3_features:
        print(
            f"  {feature['feature']}: combined={feature['combined_impact']:.6f}, "
            f"spearman={feature['spearman']:.6f}, kendall={feature['kendall']:.6f}, "
            f"mi={feature['mutual_info']:.6f}, perm={feature['permutation_importance']:.6f}"
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
        "is_male": "Sex (Male=1, Female=0)",
        "athlete_weight": "Athlete Weight [kg]",
    }

    renamed_extreme_df = extreme_df.rename(columns=column_mapping)
    mapped_features = [column_mapping.get(feat, feat) for feat in top_3_feature_names]
    
    return renamed_extreme_df, mapped_features


def get_top_features_per_model(model_results, df_numeric):
    """
    Oblicza TOP 3 features dla każdego modelu na podstawie łączonego wpływu.
    
    Returns:
        dict: {model_name -> [(feature_name, combined_impact), ...]}
    """
    top_features_per_model = {}
    
    for result in model_results:
        top_features = rank_features_for_scores(result.scores, df_numeric, top_n=3)
        top_3 = [(feature["feature"], float(feature["combined_impact"])) for feature in top_features]
        top_features_per_model[result.name] = top_3
    
    return top_features_per_model

def main():
    args = parse_args()

    if not args.skip_cleaning or not PROCESSED_DATA_PATH.exists():
        prepare_clean_data()

    df_numeric = load_clean_numeric_data(PROCESSED_DATA_PATH, sample_size=args.sample_size)

    df_train, df_test = train_test_split(df_numeric, test_size=0.2, random_state=42)
    print(f"Train/test split: {len(df_train)} train rows, {len(df_test)} test rows")

    print("running autoencoder comparison...")
    autoencoder_results, _ = run_autoencoder_comparison(
        df_train,
        df_test,
        df_numeric,
        ae_epochs=args.ae_epochs,
        ae_batch_size=args.ae_batch_size,
        ae_patience=args.ae_patience,
    )
    print("Autoencoder comparison complete.")

    pca_result = run_pca(df_train, df_test, df_numeric)
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

    comparison_df = build_comparison_table(autoencoder_results, pca_result, vae_result, df_numeric)
    comparison_path = save_comparison_table(comparison_df)

    # 1. Pobieramy wyniki z najlepszego modelu (Deep Autoencoder - index 2)
    best_model_results = autoencoder_results[2]
    
    # 2. Wybieramy skrajne przypadki na podstawie rzeczywistych danych
    print("Identyfikacja liderów i outsiderów...")
    extreme_athletes_df, top_features = get_extreme_athletes(df_numeric, best_model_results.scores)
    
    # 3. Generujemy profilowy wykres słupkowy
    from src.visualization.plots import plot_athlete_profiles
    profile_plot = plot_athlete_profiles(extreme_athletes_df, features_to_plot=top_features)
    
    # Reszta Twoich wykresów...
    model_results = [*autoencoder_results, pca_result, vae_result]
    mse_plot = plot_mse_comparison(model_results)
    
    # Nowy wykres z TOP features per model
    top_features_dict = get_top_features_per_model(model_results, df_numeric)
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
    
    print(
        "Wszystkie wizualizacje gotowe: "
        f"{profile_plot}, {mse_plot}, {top_features_plot}, {agreement_plot}"
    )

    print(f"Saved comparison table: {comparison_path}")
    print(comparison_df.to_markdown(index=False))


if __name__ == "__main__":
    main()
