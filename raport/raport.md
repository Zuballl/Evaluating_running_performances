# ARCHITEKTURY AUTOENKODERÓW W OCENIE WYDAJNOŚCI SPORTOWEJ

**Mateusz Kubita** Politechnika Warszawska  
**Jan Zubalewicz** Politechnika Warszawska  
**19 marca 2026**

---

### ABSTRAKT
W pracy badano redukcję wymiarowości danych treningowych biegaczy z użyciem modeli AE, PCA i VAE oraz interpretację cech latent score. Zamiast pojedynczej metryki istotności zastosowano podejście wielometryczne: Spearman, Kendall, Mutual Information i Permutation Importance. Najpierw analizowano metryki oddzielnie, a następnie budowano konsensus przez agregację rang (Borda) z walidacją bootstrap. Pipeline generuje zarówno wyniki zagregowane, jak i raporty per-metryka oraz macierze zgodności metryk, co zwiększa transparentność i replikowalność analizy.

---

### 1. Cel pracy
1. Zredukować wielowymiarowy profil treningowy do 1D latent score.
2. Porównać jakość rekonstrukcji modeli (AE/PCA/VAE).
3. Zidentyfikować kluczowe cechy wpływające na latent score.
4. Zweryfikować stabilność i spójność interpretacji cech.

### 2. Dane i pipeline
1. Dane wejściowe po czyszczeniu: **45 836 obserwacji**.
2. Cechy numeryczne: 8 (pace, HR, elevation, distance, cadence, decoupling, age, weight).
3. Porównywane modele: `simple_autoencoder`, `medium_autoencoder`, `deep_autoencoder`, `pca`, `vae`.
4. Selekcja modelu do profilu zawodników: `--best-model auto` (najniższe MSE na teście).

### 3. Metodologia interpretacji cech

#### 3.1 Analiza oddzielna + konsensus
W celu uniknięcia utraty informacji najpierw liczono rankingi cech osobno dla każdej metryki:
- Spearman,
- Kendall,
- Mutual Information,
- Permutation Importance.

Następnie tworzono ranking konsensusowy przez agregację rang:
\[
s_{m,f} = 1 - \frac{r_{m,f}-1}{p-1},\quad
S_f = \sum_m w_m\, s_{m,f},\quad \sum_m w_m = 1
\]

Domyślne wagi: Spearman 0.25, Kendall 0.20, MI 0.25, Permutation 0.30.

#### 3.2 Stabilność bootstrap
Stabilność oceniano bootstrapem (`--bootstrap-repeats 30`, `--bootstrap-sample-size 45836`) raportując:
- \(P(Top3)\),
- medianę rangi,
- IQR rangi.

### 4. Wyniki końcowe (pełny run produkcyjny)

Źródło: `outputs/metrics/model_comparison.csv` (run: 19.03.2026, full data).

| Model | Reconstruction MSE | Latent Score Quality | Top-1 cecha |
| :--- | ---: | ---: | :--- |
| simple_autoencoder | 0.059042 | 0.857143 | pace_min_km |
| medium_autoencoder | 0.087959 | 0.819048 | aerobic_decoupling |
| deep_autoencoder | **0.005072** | 0.757143 | pace_min_km |
| pca | 0.002174 | **0.857143** | pace_min_km |
| vae | 0.004063 | 0.785714 | average_hr |

Model wybrany automatycznie do profilu zawodników: **deep_autoencoder** (najniższe MSE wśród AE).

Top-3 cech dla deep_autoencoder (konsensus + bootstrap):
1. `pace_min_km` (P(Top3)=1.000)
2. `aerobic_decoupling` (P(Top3)=1.000)
3. `average_hr` (P(Top3)=1.000)

### 5. Artefakty interpretowalne dla człowieka

#### 5.1 Wyniki tabelaryczne
1. Agregacja: `outputs/metrics/model_comparison.csv`
2. Per-metryka: `outputs/metrics/model_comparison_by_metric.csv`
3. Zgodność metryk: `outputs/metrics/metric_agreement_by_model.csv`

#### 5.2 Wizualizacje
1. Profile zawodników: `outputs/plots/athlete_profiles.png`
2. Porównanie MSE: `outputs/plots/mse_comparison.png`
3. Top cechy per model: `outputs/plots/top_features_per_model.png`
4. Zgodność latent score modeli: `outputs/plots/model_agreement.png`
5. Heatmapy per-metryka: `outputs/plots/metricwise_top3_heatmaps.png`
6. Konsensus metryk (Top-3 frequency): `outputs/plots/metricwise_consensus.png`
7. Macierz zgodności metryk (rank-correlation): `outputs/plots/metric_agreement_heatmaps.png`

Uwaga: wykresy metryk wymuszają pełną listę 8 cech; jeśli cecha nie pojawia się w Top-3, jest pokazywana z wartością 0 (brak „znikających” słupków).

### 6. Wnioski
1. Połączenie analizy oddzielnej metryk i konsensusu rang jest bardziej wiarygodne niż pojedyncza metryka lub uśrednianie surowych wartości.
2. Bootstrap i macierz zgodności metryk potwierdzają stabilność kluczowych cech oraz ułatwiają wykrywanie rozbieżności między metrykami.
3. Projekt jest domknięty: pipeline produkuje spójny zestaw CSV + wykresów do raportowania i obrony wyników.
