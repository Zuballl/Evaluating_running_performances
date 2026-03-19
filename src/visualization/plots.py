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

    impact_columns = [
        column
        for column in comparison_df.columns
        if column.startswith("Top ") and column.endswith("Combined Impact")
    ]
    if not impact_columns:
        raise ValueError("No dynamic combined impact columns found in comparison DataFrame.")

    plot_df = comparison_df[["Model", *impact_columns]].melt(
        id_vars="Model",
        var_name="Metric",
        value_name="Value",
    )

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=plot_df, x="Model", y="Value", hue="Metric", palette="crest")
    plt.title("Porównanie jakości latent score (agregacja rang)")
    plt.ylabel("Aggregated rank impact")
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


def plot_athlete_profiles(comparison_data: pd.DataFrame, features_to_plot: list[str] | None = None, output_path: Path | None = None) -> Path:
    """
    Tworzy wizualizację profili skrajnych zawodników (rzeczywiste wartości).

    Argumenty:
        comparison_data (pd.DataFrame): DataFrame zawierający dane porównawcze.
            Oczekuje kolumny 'Athlete Label'.
        features_to_plot (list[str] | None): Lista cech do wykreślenia. 
            Jeśli None, używa domyślnych: ["Pace [min/km]", "Elevation Gain [m]", "Average Heart Rate [bpm]"]
        output_path (Path | None): Ścieżka do zapisu pliku PNG. Jeśli None,
            używa domyślnej lokalizacji w PLOTS_DIR.

    Zwraca:
        Path: Ścieżka do zapisu wygenerowanego pliku PNG.
    """
    if output_path is None:
        output_path = PLOTS_DIR / "athlete_profiles.png"

    if features_to_plot is None:
        features_to_plot = ['Pace [min/km]', 'Elevation Gain [m]', 'Average Heart Rate [bpm]']

    # Weryfikuj, że wszystkie features istnieją w DataFrame
    missing_features = [f for f in features_to_plot if f not in comparison_data.columns]
    if missing_features:
        raise ValueError(f"Missing features in DataFrame: {missing_features}")

    # Transformacja danych do formatu "długiego" (long)
    data_long = pd.melt(comparison_data,
                         id_vars=['Athlete Label'],
                         value_vars=features_to_plot,
                         var_name='Metric',
                         value_name='Value')

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Inicjalizacja figury matplotlib
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(22, 7), sharex=True)

    # Mapowanie feature names na bardziej czytalne nazwy (opcjonalnie)
    feature_titles = {
        'Pace [min/km]': 'Tempo Biegowe [min/km]',
        'Elevation Gain [m]': 'Przewyższenie [m]',
        'Average Heart Rate [bpm]': 'Tętno Średnie [bpm]',
        'Total Distance [km]': 'Całkowita Odległość [km]',
        'Final Cadence [spm]': 'Ostateczna Kadencja [spm]',
        'Aerobic Decoupling [%]': 'Aerobic Decoupling [%]',
        'Age [years]': 'Wiek [lata]',
        'Athlete Weight [kg]': 'Waga Zawodnika [kg]',
    }

    for i, metric in enumerate(features_to_plot):
        ax = axs[i]
        metric_data = data_long[data_long['Metric'] == metric]

        sns.barplot(data=metric_data, x='Athlete Label', y='Value', ax=ax, hue='Athlete Label', palette="muted", legend=False)

        # Tytuł metryki
        metric_title = feature_titles.get(metric, metric)
        ax.set_title(metric_title)
        ax.set_xlabel("")
        ax.set_ylabel("")

        # Dodaj etykiety wartości na górze słupków
        for bar in ax.patches:
            value = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2,
                    value + (max(metric_data['Value']) * 0.01),
                    f"{value:.1f}",
                    ha='center',
                    fontweight="bold")

        # Dostosuj zakresy osi Y
        y_min = max(metric_data['Value'].min() * 0.8, 0)
        y_max = metric_data['Value'].max() * 1.15
        ax.set_ylim(y_min, y_max)

    # Tytuł całej figury (u góry)
    fig.suptitle("Fizjologiczny Profil Skrajnych Zawodników (Wartości Rzeczywiste)", fontsize=16)

    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    return output_path


def plot_top_features_per_model(top_features_dict: dict[str, list], output_path: Path | None = None) -> Path:
    """
    Pokazuje TOP 3 features dla każdego modelu z kolorami medalowymi (🥇🥈🥉).
    
    Argumenty:
        top_features_dict: {model_name -> [(feature_name, abs_correlation), ...]}
        output_path: Ścieżka do pliku wyjściowego
    
    Zwraca:
        Path: Ścieżka do zapisu pliku
    """
    if output_path is None:
        output_path = PLOTS_DIR / "top_features_per_model.png"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Mapowanie nazw features na czytalne nazwy
    feature_labels = {
        "pace_min_km": "Pace [min/km]",
        "average_hr": "Average HR [bpm]",
        "elevation_gain": "Elevation Gain [m]",
        "total_distance": "Total Distance [km]",
        "final_cadence": "Final Cadence [spm]",
        "aerobic_decoupling": "Aerobic Decoupling [%]",
        "age": "Age [years]",
        "athlete_weight": "Weight [kg]",
    }
    
    # Medal colors: Gold, Silver, Bronze
    medal_colors = ["#FFD700", "#C0C0C0", "#CD7F32"]
    medal_labels = ["1st", "2nd", "3rd"]  # Text labels instead of emojis

    models = list(top_features_dict.keys())
    num_models = len(models)
    
    # Each model has different top features, so do not share Y axis labels.
    fig, axs = plt.subplots(nrows=1, ncols=num_models, figsize=(22, 6), sharey=False)
    if num_models == 1:
        axs = [axs]

    for idx, model_name in enumerate(models):
        ax = axs[idx]
        features = top_features_dict[model_name]
        
        # Extract feature names and values with consistent ordering (position 0, 1, 2)
        feature_data = [(feature_labels.get(f[0], f[0]), 0.0 if pd.isna(f[1]) else float(f[1])) for f in features]
        feature_names = [item[0] for item in feature_data]
        correlations = [item[1] for item in feature_data]
        
        # Barplot with medal colors - reverse order so #1 is at top
        y_pos = range(len(feature_names))
        bars = ax.barh(y_pos, correlations, color=medal_colors, edgecolor="black", linewidth=1.5)
        
        # Set tick labels explicitly 
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_names, fontweight="bold")
        
        # Add medal symbols and values on bars
        for i, (y, corr) in enumerate(zip(y_pos, correlations)):
            medal = medal_labels[i]
            ax.text(corr + 0.02, y, f"[{medal}] {corr:.4f}", va="center", fontweight="bold", fontsize=10)
        
        ax.set_xlabel("Aggregated Rank Score", fontweight="bold")
        ax.set_title(model_name, fontweight="bold", fontsize=12)
        max_corr = max(correlations) if correlations else 0.0
        ax.set_xlim(0, (max_corr * 1.35) if max_corr > 0 else 1.0)
        ax.grid(axis="x", alpha=0.3, linestyle="--")

    fig.suptitle("TOP 3 Features per Model (Medal Ranking by Aggregated Rank)", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def plot_metricwise_top3_heatmaps(metricwise_df: pd.DataFrame, output_path: Path | None = None) -> Path:
    """
    Pokazuje heatmapy TOP-3 per metryka dla każdego modelu.

    Argumenty:
        metricwise_df: DataFrame z kolumnami [Model, Metric, Rank, Feature, Metric Score]
        output_path: Ścieżka do pliku wyjściowego

    Zwraca:
        Path: Ścieżka do zapisu pliku
    """
    if output_path is None:
        output_path = PLOTS_DIR / "metricwise_top3_heatmaps.png"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    metric_labels = {
        "spearman": "Spearman",
        "kendall": "Kendall",
        "mutual_info": "Mutual Info",
        "permutation_importance": "Permutation",
    }
    feature_labels = {
        "pace_min_km": "Pace",
        "average_hr": "Avg HR",
        "elevation_gain": "Elevation",
        "total_distance": "Distance",
        "final_cadence": "Cadence",
        "aerobic_decoupling": "Decoupling",
        "age": "Age",
        "athlete_weight": "Weight",
    }
    metric_order = ["Spearman", "Kendall", "Mutual Info", "Permutation"]
    feature_order = [
        "Pace",
        "Avg HR",
        "Elevation",
        "Distance",
        "Cadence",
        "Decoupling",
        "Age",
        "Weight",
    ]

    df = metricwise_df.copy()
    df["Metric"] = df["Metric"].map(lambda x: metric_labels.get(x, x))
    df["Feature"] = df["Feature"].map(lambda x: feature_labels.get(x, x))

    models = list(df["Model"].unique())
    n_models = len(models)
    ncols = 2
    nrows = (n_models + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 4 * nrows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for idx, model in enumerate(models):
        ax = axes[idx]
        model_df = df[df["Model"] == model]
        pivot = model_df.pivot_table(
            index="Metric",
            columns="Feature",
            values="Metric Score",
            aggfunc="mean",
            fill_value=0.0,
        )
        pivot = pivot.reindex(index=metric_order, columns=feature_order, fill_value=0.0)
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".3f",
            cmap="YlOrRd",
            cbar=False,
            linewidths=0.5,
            linecolor="white",
            ax=ax,
        )
        ax.set_title(model, fontweight="bold")
        ax.set_xlabel("Feature")
        ax.set_ylabel("Metric")
        ax.tick_params(axis="x", rotation=35)

    for idx in range(n_models, len(axes)):
        axes[idx].axis("off")

    fig.suptitle("TOP-3 Feature Scores Per Metric (By Model)", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def plot_metricwise_consensus(metricwise_df: pd.DataFrame, output_path: Path | None = None) -> Path:
    """
    Pokazuje konsensus metryk: ile razy cecha pojawia się w TOP-3,
    rozbite na metryki (stacked bar).
    """
    if output_path is None:
        output_path = PLOTS_DIR / "metricwise_consensus.png"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    metric_labels = {
        "spearman": "Spearman",
        "kendall": "Kendall",
        "mutual_info": "Mutual Info",
        "permutation_importance": "Permutation",
    }
    feature_labels = {
        "pace_min_km": "Pace",
        "average_hr": "Avg HR",
        "elevation_gain": "Elevation",
        "total_distance": "Distance",
        "final_cadence": "Cadence",
        "aerobic_decoupling": "Decoupling",
        "age": "Age",
        "athlete_weight": "Weight",
    }
    metric_order = ["Spearman", "Kendall", "Mutual Info", "Permutation"]
    feature_order = [
        "Pace",
        "Avg HR",
        "Elevation",
        "Distance",
        "Cadence",
        "Decoupling",
        "Age",
        "Weight",
    ]

    df = metricwise_df.copy()
    df["Metric"] = df["Metric"].map(lambda x: metric_labels.get(x, x))
    df["Feature"] = df["Feature"].map(lambda x: feature_labels.get(x, x))

    counts = (
        df.groupby(["Feature", "Metric"], as_index=False)
        .size()
        .rename(columns={"size": "Count"})
    )
    pivot = counts.pivot(index="Feature", columns="Metric", values="Count").fillna(0)
    pivot = pivot.reindex(index=feature_order, columns=metric_order, fill_value=0)

    ax = pivot.plot(
        kind="bar",
        stacked=True,
        figsize=(12, 7),
        colormap="tab20",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_title("Feature Consensus Across Metrics (Top-3 Frequency)", fontweight="bold")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Count of appearances in Top-3")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.25, linestyle="--")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def plot_metric_agreement_heatmaps(metric_agreement_df: pd.DataFrame, output_path: Path | None = None) -> Path:
    """
    Heatmapy zgodności metryk (korelacja rang) dla każdego modelu.
    """
    if output_path is None:
        output_path = PLOTS_DIR / "metric_agreement_heatmaps.png"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    metric_labels = {
        "spearman": "Spearman",
        "kendall": "Kendall",
        "mutual_info": "Mutual Info",
        "permutation_importance": "Permutation",
    }
    metric_order = ["Spearman", "Kendall", "Mutual Info", "Permutation"]

    df = metric_agreement_df.copy()
    df["Metric A"] = df["Metric A"].map(lambda x: metric_labels.get(x, x))
    df["Metric B"] = df["Metric B"].map(lambda x: metric_labels.get(x, x))

    models = list(df["Model"].unique())
    n_models = len(models)
    ncols = 2
    nrows = (n_models + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(13, 5 * nrows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for idx, model in enumerate(models):
        ax = axes[idx]
        model_df = df[df["Model"] == model]
        pivot = model_df.pivot_table(
            index="Metric A",
            columns="Metric B",
            values="Rank Correlation",
            aggfunc="mean",
            fill_value=0.0,
        )
        pivot = pivot.reindex(index=metric_order, columns=metric_order, fill_value=0.0)
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            cbar=(idx == 0),
            square=True,
            linewidths=0.5,
            linecolor="white",
            ax=ax,
        )
        ax.set_title(model, fontweight="bold")
        ax.set_xlabel("Metric B")
        ax.set_ylabel("Metric A")
        ax.tick_params(axis="x", rotation=35)

    for idx in range(n_models, len(axes)):
        axes[idx].axis("off")

    fig.suptitle("Metric Agreement Matrix (Spearman Correlation of Feature Ranks)", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path