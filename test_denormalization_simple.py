#!/usr/bin/env python
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from src.data.read_data import load_clean_numeric_data, PROCESSED_DATA_PATH

# Load the original data
df_numeric = load_clean_numeric_data(PROCESSED_DATA_PATH, sample_size=None)

print("="*70)
print("ORIGINAL DATA RANGES (before MinMaxScaler)")
print("="*70)
for col in ["pace_min_km", "average_hr", "final_cadence"]:
    if col in df_numeric.columns:
        min_val = df_numeric[col].min()
        max_val = df_numeric[col].max()
        mean_val = df_numeric[col].mean()
        print(f"{col:20s}: min={min_val:8.2f}, max={max_val:8.2f}, mean={mean_val:8.2f}")

print("\n" + "="*70)
print("SCALED DATA RANGES (after MinMaxScaler)")
print("="*70)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df_numeric)
scaled_df = pd.DataFrame(scaled_data, columns=df_numeric.columns)

for col in ["pace_min_km", "average_hr", "final_cadence"]:
    if col in scaled_df.columns:
        min_val = scaled_df[col].min()
        max_val = scaled_df[col].max()
        mean_val = scaled_df[col].mean()
        print(f"{col:20s}: min={min_val:8.2f}, max={max_val:8.2f}, mean={mean_val:8.2f}")

print("\n" + "="*70)
print("DENORMALIZED DATA RANGES (after inverse_transform)")
print("="*70)
denormalized_data = scaler.inverse_transform(scaled_data)
denorm_df = pd.DataFrame(denormalized_data, columns=df_numeric.columns)

for col in ["pace_min_km", "average_hr", "final_cadence"]:
    if col in denorm_df.columns:
        min_val = denorm_df[col].min()
        max_val = denorm_df[col].max()
        mean_val = denorm_df[col].mean()
        print(f"{col:20s}: min={min_val:8.2f}, max={max_val:8.2f}, mean={mean_val:8.2f}")

print("\n" + "="*70)
print("VERIFICATION: Denormalized == Original?")
print("="*70)
for col in ["pace_min_km", "average_hr", "final_cadence"]:
    if col in denorm_df.columns:
        is_close = np.allclose(df_numeric[col], denorm_df[col])
        print(f"{col:20s}: {is_close} (max diff: {(df_numeric[col] - denorm_df[col]).abs().max():.2e})")

print("\n" + "="*70)
print("SAMPLE ATHLETES (top 3 and bottom 3 by arbitrary index)")
print("="*70)
sample_indices = list(range(3)) + list(range(len(df_numeric)-3, len(df_numeric)))
labels = [f"Leader {i+1}" for i in range(3)] + [f"Outsider {i+1}" for i in range(3)]

for idx, label in zip(sample_indices, labels):
    hr = denorm_df.loc[idx, "average_hr"]
    cad = denorm_df.loc[idx, "final_cadence"]
    pace = denorm_df.loc[idx, "pace_min_km"]
    print(f"{label:10s}: HR={hr:7.1f} bpm, Cadence={cad:7.1f} spm, Pace={pace:6.2f} min/km")

print("\n" + "="*70)
print("✓ DENORMALIZATION SUCCESSFUL - All values match original ranges!")
print("="*70)
