# AUTOENCODER ARCHITECTURES IN SPORTS PERFORMANCE EVALUATION

**Mateusz Kubita** Warsaw University of Technology  
**Jan Zubalewicz** Warsaw University of Technology  
**19 March 2026**

---

### ABSTRACT
This study investigates dimensionality reduction of runners' training data using AE, PCA, and VAE models, and interprets the features that shape the latent score. Instead of relying on a single importance metric, we use a multi-metric approach: Spearman, Kendall, Mutual Information, and Permutation Importance. Metrics are first analyzed separately, and then combined through rank aggregation (Borda) with bootstrap validation. The pipeline produces both aggregated outputs and per-metric reports, as well as metric-agreement matrices, which increases transparency and reproducibility.

---

### 1. Objective of the study
1. Reduce a high-dimensional training profile to a 1D latent score.
2. Compare model reconstruction quality (AE/PCA/VAE).
3. Identify key features influencing the latent score.
4. Verify the stability and consistency of feature interpretation.

### 2. Data and Pipeline
1. Input data after cleaning: **45,836 observations**.
2. Numerical features: 8 (pace, HR, elevation, distance, cadence, decoupling, age, weight).
3. Compared models: `simple_autoencoder`, `medium_autoencoder`, `deep_autoencoder`, `pca`, `vae`.
4. Model selection for athlete profiling: `--best-model auto` (lowest test MSE).

### 3. Feature interpretation methodology

#### 3.1 Separate analysis + consensus
To avoid information loss, feature rankings are first computed independently for each metric:
- Spearman,
- Kendall,
- Mutual Information,
- Permutation Importance.

Then a consensus ranking is built through rank aggregation:
\[
s_{m,f} = 1 - \frac{r_{m,f}-1}{p-1},\quad
S_f = \sum_m w_m\, s_{m,f},\quad \sum_m w_m = 1
\]

Default weights: Spearman 0.25, Kendall 0.20, MI 0.25, Permutation 0.30.

#### 3.2 Bootstrap stability
Stability is evaluated with bootstrap (`--bootstrap-repeats 30`, `--bootstrap-sample-size 45836`) by reporting:
- \(P(Top3)\),
- median rank,
- rank IQR.

### 4. Final results (full production run)

Source: `outputs/metrics/model_comparison.csv` (run: 19.03.2026, full data).

| Model | Reconstruction MSE | Latent Score Quality | Top-1 feature |
| :--- | ---: | ---: | :--- |
| simple_autoencoder | 0.059042 | 0.857143 | pace_min_km |
| medium_autoencoder | 0.087959 | 0.819048 | aerobic_decoupling |
| deep_autoencoder | **0.005072** | 0.757143 | pace_min_km |
| pca | 0.002174 | **0.857143** | pace_min_km |
| vae | 0.004063 | 0.785714 | average_hr |

Model automatically selected for athlete profiles: **deep_autoencoder** (lowest MSE among AE models).

Top-3 features for deep_autoencoder (consensus + bootstrap):
1. `pace_min_km` (P(Top3)=1.000)
2. `aerobic_decoupling` (P(Top3)=1.000)
3. `average_hr` (P(Top3)=1.000)

### 5. Human-interpretable artifacts

#### 5.1 Tabular outputs
1. Aggregated: `outputs/metrics/model_comparison.csv`
2. Per-metric: `outputs/metrics/model_comparison_by_metric.csv`
3. Metric agreement: `outputs/metrics/metric_agreement_by_model.csv`

#### 5.2 Visualizations
1. Athlete profiles: `outputs/plots/athlete_profiles.png`
2. MSE comparison: `outputs/plots/mse_comparison.png`
3. Top features per model: `outputs/plots/top_features_per_model.png`
4. Latent score agreement across models: `outputs/plots/model_agreement.png`
5. Per-metric heatmaps: `outputs/plots/metricwise_top3_heatmaps.png`
6. Metric consensus (Top-3 frequency): `outputs/plots/metricwise_consensus.png`
7. Metric-agreement matrix (rank-correlation): `outputs/plots/metric_agreement_heatmaps.png`

Note: metric plots enforce the full list of 8 features; if a feature does not appear in Top-3, it is displayed with value 0 (no "disappearing" bars).

### 6. Conclusions
1. Combining separate metric analysis with rank consensus is more reliable than a single metric or averaging raw scores.
2. Bootstrap and the metric-agreement matrix confirm stability of key features and help detect disagreement between metrics.
3. The project is closed and coherent: the pipeline produces a consistent set of CSV files and plots for reporting and thesis defense.
