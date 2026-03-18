#!/usr/bin/env python
import pandas as pd
from sklearn.model_selection import train_test_split
from src.data.read_data import load_clean_numeric_data, PROCESSED_DATA_PATH
from src.models.autoencoder_models import run_autoencoder_comparison
from src.pipeline.run_experiments import get_extreme_athletes

df_numeric = load_clean_numeric_data(PROCESSED_DATA_PATH)
df_train, df_test = train_test_split(df_numeric, test_size=0.2, random_state=42)

print("Running autoencoder comparison...")
autoencoder_results, _, ae_scaler = run_autoencoder_comparison(
    df_train, df_test, df_numeric
)
best_model_results = autoencoder_results[2]

print("\nGetting extreme athletes...")
extreme_athletes_df, top_features = get_extreme_athletes(
    df_numeric, best_model_results.scores, scaler=ae_scaler
)

print("\n" + "="*60)
print("EXTREME ATHLETES WITH DENORMALIZED VALUES")
print("="*60)
print(extreme_athletes_df[["Athlete Label", "Average Heart Rate [bpm]", "Final Cadence [spm]", "Pace [min/km]"]])

print("\n" + "="*60)
print("VALUE RANGES (should be physiologically plausible)")
print("="*60)
for col in ["Average Heart Rate [bpm]", "Final Cadence [spm]", "Pace [min/km]"]:
    if col in extreme_athletes_df.columns:
        min_val = extreme_athletes_df[col].min()
        max_val = extreme_athletes_df[col].max()
        print(f"{col:30s}: min={min_val:8.2f}, max={max_val:8.2f}")

print("\n" + "="*60)
print("PHYSIOLOGICAL RANGES FOR REFERENCE")
print("="*60)
print("Heart Rate [bpm]:              60-180 (realistic range)")
print("Final Cadence [spm]:           160-180 (running step rate)")
print("Pace [min/km]:                 3-7 (typical running pace)")
print("="*60)
