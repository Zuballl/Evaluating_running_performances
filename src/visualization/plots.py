from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
PLOTS_DIR = ROOT_DIR / "outputs" / "plots"


sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 300


def plot_mse_comparison(
    model_results,
    output_path: Path | None = None,
) -> Path:
    if output_path is None:
        output_path = PLOTS_DIR / "mse_comparison.png"

    model_names = [result.name for result in model_results]
    mse_values = [result.mse for result in model_results]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=model_names, y=mse_values, hue=model_names, palette="viridis", legend=False)
    plt.title("Porównanie rekonstrukcji modeli (MSE)")
    plt.ylabel("Reconstruction MSE")

    for index, value in enumerate(mse_values):
        ax.text(index, value + (max(mse_values) * 0.01), f"{value:.6f}", ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def plot_latent_score_comparison(comparison_df: pd.DataFrame, output_path: Path | None = None) -> Path:
    if output_path is None:
        output_path = PLOTS_DIR / "latent_score_comparison.png"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_df = comparison_df[["Model", "Latent Score vs Pace (Spearman)", "Latent Score vs HR (Spearman)"]].melt(
        id_vars="Model",
        var_name="Metric",
        value_name="Value",
    )

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=plot_df, x="Model", y="Value", hue="Metric", palette="crest")
    plt.title("Porównanie jakości latent score")
    plt.ylabel("Spearman correlation")
    plt.xlabel("Model")
    plt.axhline(0, color="black", linewidth=0.8)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def plot_model_agreement(model_scores: dict[str, object], output_path: Path | None = None) -> Path:
    if output_path is None:
        output_path = PLOTS_DIR / "model_agreement.png"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    scores_df = pd.DataFrame(model_scores)
    corr = scores_df.corr(method="spearman")

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True)
    plt.title("Macierz zgodności latent score (Spearman)")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def plot_athlete_profiles(comparison_data: pd.DataFrame, output_path: Path | None = None) -> Path:
    """
    Tworzy wizualizację profili skrajnych zawodników (rzeczywiste wartości).

    Argumenty:
        comparison_data (pd.DataFrame): DataFrame zawierający dane porównawcze.
            Oczekuje kolumn: 'Athlete Label', 'Average Speed [km/h]',
            'Elevation Gain [m]', 'Average Heart Rate [bpm]'.
            Kolumna 'Athlete Label' powinna zawierać opisy (np. 'Lider 1', 'Outsider 1').
        output_path (Path | None): Ścieżka do zapisu pliku PNG. Jeśli None,
            używa domyślnej lokalizacji w PLOTS_DIR.

    Zwraca:
        Path: Ścieżka do zapisu wygenerowanego pliku PNG.
    """
    if output_path is None:
        output_path = PLOTS_DIR / "athlete_profiles.png"

    # Przygotowanie danych do wizualizacji - transpozycja, aby uzyskać format long
    # co jest preferowane przez Seaborn (dla łatwego grupowania i definiowania osi)
    # columns_to_plot to lista metryk do wykreślenia
    columns_to_plot = ['Average Speed [km/h]', 'Elevation Gain [m]', 'Average Heart Rate [bpm]']

    # Transformacja danych do formatu "długiego" (long)
    # co ułatwia pracę z seaborn i matplotlib dla zagnieżdżonych wykresów
    data_long = pd.melt(comparison_data,
                         id_vars=['Athlete Label'],
                         value_vars=columns_to_plot,
                         var_name='Metric',
                         value_name='Value')

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Inicjalizacja figury matplotlib
    # Używamy subplots, aby łatwo kontrolować poszczególne wykresy barplot
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18, 6), sharex=True)

    # Lista metryk i jednostek do tytułów osi Y (opcjonalne)
    metrics_titles = [
        "Prędkość Średnia [km/h]",
        "Przewyższenie [m]",
        "Tętno Średnie [bpm]"
    ]

    # Lista metryk do iteracji po kolumnach
    for i, (metric, title) in enumerate(zip(columns_to_plot, metrics_titles)):
        # Przypisanie osi dla konkretnej metryki
        ax = axs[i]

        # Wybierz dane dla danej metryki
        metric_data = data_long[data_long['Metric'] == metric]

        # Tworzenie barplotu dla danej metryki
        # sns.barplot(x=labels, y=metric_data['Value'], ax=ax, hue=labels, palette="muted", legend=False)
        sns.barplot(data=metric_data, x='Athlete Label', y='Value', ax=ax, hue='Athlete Label', palette="muted", legend=False)

        # Ustawienia osi i tytułów
        ax.set_title(title)
        ax.set_xlabel("") # Ukryj etykietę osi X dla przejrzystości
        ax.set_ylabel("") # Ukryj etykietę osi Y, jednostki są w tytule

        # Dodaj etykiety wartości na górze słupków
        for bar in ax.patches:
            # Uzyskaj wartość dla paska
            value = bar.get_height()
            # Dodaj tekst nad paskiem, wyrównany do środka i pogrubiony
            ax.text(bar.get_x() + bar.get_width() / 2, # x-coordinate of the text
                    value + (max(metric_data['Value']) * 0.01), # y-coordinate of the text
                    f"{value:.1f}", # formatted value string
                    ha='center', # horizontal alignment
                    fontweight="bold") # font weight

        # Opcjonalnie: dostosuj zakres osi Y dla lepszej prezentacji danych (np. dla tętna)
        if metric == 'Average Heart Rate [bpm]':
             ax.set_ylim(0, 160) # Na przykład dla tętna
        elif metric == 'Average Speed [km/h]':
            ax.set_ylim(0, 35)
        elif metric == 'Elevation Gain [m]':
            ax.set_ylim(0, 1200)

    # Tytuł całej figury (u góry)
    fig.suptitle("Fizjologiczny Profil Skrajnych Zawodników (Wartości Rzeczywiste)", fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Dostosuj layout dla lepszej prezentacji, pozostawiając miejsce na suptitle
    plt.savefig(output_path)
    plt.close()
    return output_path