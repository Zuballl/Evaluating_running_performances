# Evaluating Running Performances

Projekt do oceny wydajności biegowej na podstawie danych treningowych. Repozytorium łączy:
- pipeline badawczy (AE/PCA/VAE + interpretacja cech),
- backend API (FastAPI) do scoringu aktywności FIT,
- frontend (React + Vite) do wygodnej prezentacji wyniku i wkładów cech,
- raporty końcowe w wersji PL i EN.

## Co robi projekt

1. Czyści dane i buduje zbiór numeryczny (8 cech).
2. Trenuje i porównuje modele: `simple_autoencoder`, `medium_autoencoder`, `deep_autoencoder`, `pca`, `vae`.
3. Wyznacza latent score i ranking cech metodą multi-metric:
   - Spearman,
   - Kendall,
   - Mutual Information,
   - Permutation Importance.
4. Buduje konsensus rang (Borda-like) i ocenia stabilność bootstrapem.
5. Generuje artefakty CSV + wykresy do raportu.

## Struktura repo

- `src/` - kod pipeline, modeli, ewaluacji, wizualizacji, API
- `data/` - dane wejściowe i przetworzone
- `outputs/metrics/` - tabele wynikowe
- `outputs/plots/` - wykresy
- `raport/` - raporty końcowe:
  - `raport.md` (PL)
  - `raport_en.md` (EN)
- `frontend/` - aplikacja React (Vite)

## Wymagania

- Python 3.11+
- Node.js 18+
- npm 9+

## Szybki start (Python)

W katalogu głównym repo:

```bash
python3 -m venv .venv311
source .venv311/bin/activate
pip install --upgrade pip
pip install numpy pandas scikit-learn matplotlib seaborn torch fastapi uvicorn pydantic python-multipart fitparse
```

## Uruchomienie pełnego pipeline

```bash
source .venv311/bin/activate
python3 -m src.pipeline.run_experiments \
  --bootstrap-repeats 30 \
  --bootstrap-sample-size 45836 \
  --random-seed 42
```

Najważniejsze argumenty CLI:
- `--sample-size` - liczba rekordów do eksperymentu
- `--ae-epochs`, `--vae-epochs` - liczba epok treningu
- `--best-model auto|simple|medium|deep` - wybór modelu AE do profili zawodników
- `--all-ae-profiles` - generowanie profili dla wszystkich AE
- `--weight-spearman`, `--weight-kendall`, `--weight-mi`, `--weight-perm` - wagi metryk
- `--random-seed` - reprodukowalność

## Główne artefakty wynikowe

Po runie pipeline znajdziesz:

### Tabele
- `outputs/metrics/model_comparison.csv`
- `outputs/metrics/model_comparison_by_metric.csv`
- `outputs/metrics/metric_agreement_by_model.csv`

### Wykresy
- `outputs/plots/athlete_profiles.png`
- `outputs/plots/mse_comparison.png`
- `outputs/plots/top_features_per_model.png`
- `outputs/plots/model_agreement.png`
- `outputs/plots/metricwise_top3_heatmaps.png`
- `outputs/plots/metricwise_consensus.png`
- `outputs/plots/metric_agreement_heatmaps.png`

## Uruchomienie API (FastAPI)

```bash
source .venv311/bin/activate
uvicorn src.backend.app:app --reload
```

API domyślnie działa pod `http://127.0.0.1:8000`.

Przydatne endpointy:
- `GET /health`
- `POST /predict-json`
- `POST /predict-fit`

## Uruchomienie frontendu

W osobnym terminalu:

```bash
cd frontend
npm install
npm run dev
```

Frontend domyślnie woła backend pod `http://127.0.0.1:8000`.
Możesz to zmienić przez zmienną środowiskową `VITE_API_BASE_URL`.

## Raporty

- Wersja polska: `raport/raport.md`
- English version: `raport/raport_en.md`

## Reproducibility checklist

Aby odtworzyć wyniki raportu:
1. Użyj pełnych danych (bez redukcji `--sample-size`).
2. Ustaw `--bootstrap-repeats 30` i `--bootstrap-sample-size 45836`.
3. Ustaw `--random-seed 42`.
4. Sprawdź, czy świeżo wygenerowano pliki w `outputs/metrics/` i `outputs/plots/`.
