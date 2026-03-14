import argparse
from pathlib import Path

from src.data.read_data import PROCESSED_DATA_PATH, load_clean_numeric_data, prepare_clean_data
from src.evaluation.compare import build_comparison_table, save_comparison_table
from src.models.autoencoder_models import run_autoencoder_comparison
from src.models.tsne_model import run_tsne
from src.models.vae_model import run_vae
import pandas as pd
from src.visualization.plots import plot_model_agreement, plot_mse_comparison


def parse_args():
    parser = argparse.ArgumentParser(description="Run performance-score experiments.")
    parser.add_argument("--sample-size", type=int, default=1000000, help="Number of rows used for model experiments.")
    parser.add_argument("--skip-cleaning", action="store_true", help="Use existing clean dataset instead of regenerating it.")
    parser.add_argument("--run-vae", action="store_true", help="Run VAE model.")
    parser.add_argument("--run-tsne", action="store_true", help="Run t-SNE model.")
    parser.add_argument("--with-plots", action="store_true", help="Generate plots for autoencoder comparison.")
    return parser.parse_args()

def get_extreme_athletes(df_numeric, scores):
    # Dodajemy wyniki do kopii danych, aby móc je posortować
    df_with_scores = df_numeric.copy()
    df_with_scores['performance_score'] = scores
    
    # Sortujemy: od najlepszych do najgorszych
    df_sorted = df_with_scores.sort_values(by='performance_score', ascending=False)
    
    # Wybieramy Top 3 i Bottom 2
    top_3 = df_sorted.head(3).copy()
    bottom_2 = df_sorted.tail(2).copy()
    
    # Nadajemy etykiety dla wykresu
    top_3['Athlete Label'] = [f"Lider {i+1}" for i in range(3)]
    bottom_2['Athlete Label'] = [f"Outsider {i+1}" for i in reversed(range(2))]
    
    # Łączymy w jeden DataFrame do wykresu
    extreme_df = pd.concat([top_3, bottom_2])
    
    # Mapujemy Twoje nazwy kolumn na te używane w funkcji plot_athlete_profiles
    # Dostosuj nazwy 'average_speed' itp. do tych, które masz w df_numeric
    column_mapping = {
        'average_speed': 'Average Speed [km/h]',
        'average_hr': 'Average Heart Rate [bpm]',
        'elevation_gain': 'Average Power [W]' # Jeśli nie masz mocy, używamy przewyższenia
    }
    
    return extreme_df.rename(columns=column_mapping)

def main():
    args = parse_args()

    if not args.skip_cleaning or not PROCESSED_DATA_PATH.exists():
        prepare_clean_data()

    df_numeric = load_clean_numeric_data(PROCESSED_DATA_PATH, sample_size=args.sample_size)

    print("running autoencoder comparison...")
    autoencoder_results, _ = run_autoencoder_comparison(df_numeric)
    print("Autoencoder comparison complete.")

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

    # 1. Pobieramy wyniki z najlepszego modelu (Deep Autoencoder - index 2)
    best_model_results = autoencoder_results[2]
    
    # 2. Wybieramy skrajne przypadki na podstawie rzeczywistych danych
    print("Identyfikacja liderów i outsiderów...")
    extreme_athletes_df = get_extreme_athletes(df_numeric, best_model_results.scores)
    
    # 3. Generujemy profilowy wykres słupkowy
    from src.visualization.plots import plot_athlete_profiles
    profile_plot = plot_athlete_profiles(extreme_athletes_df)
    
    # Reszta Twoich wykresów...
    simple_scores = autoencoder_results[0].scores
    deep_scores = autoencoder_results[2].scores
    plot_mse_comparison(autoencoder_results)
    plot_model_agreement(simple_scores, deep_scores)
    
    print(f"Wszystkie wizualizacje gotowe, w tym: {profile_plot}")

    print(f"Saved comparison table: {comparison_path}")
    print(comparison_df.to_markdown(index=False))


if __name__ == "__main__":
    main()
