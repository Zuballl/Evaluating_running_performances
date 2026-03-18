#!/usr/bin/env python
import pandas as pd
from pathlib import Path

RAW_DATA_PATH = Path("data/activities.csv")

df = pd.read_csv(RAW_DATA_PATH)
df.columns = df.columns.str.strip()

print("="*70)
print("CURRENT DATA DISTRIBUTION")
print("="*70)
print(f"Total rows: {len(df)}\n")

CRUCIAL_COLUMNS = ["average_hr", "final_cadence", "average_speed", "total_distance"]
df_filtered = df.dropna(subset=CRUCIAL_COLUMNS).copy()
df_filtered["pace_min_km"] = 60 / df_filtered["average_speed"]

for col in ["pace_min_km", "average_hr", "final_cadence", "total_distance"]:
    if col in df_filtered.columns:
        p1, p5, p25, p50, p75, p95, p99 = df_filtered[col].quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        print(f"\n{col:20s}:")
        print(f"  p01={p1:10.2f}, p05={p5:10.2f}, p25={p25:10.2f}")
        print(f"  p50={p50:10.2f}, p75={p75:10.2f}, p95={p95:10.2f}, p99={p99:10.2f}")
        print(f"  min={df_filtered[col].min():10.2f}, max={df_filtered[col].max():10.2f}")

print("\n" + "="*70)
print("PROPOSED OUTLIER REMOVAL THRESHOLDS")
print("="*70)

thresholds = {
    "pace_min_km": (2.0, 15.0),
    "average_hr": (40.0, 220.0),
    "final_cadence": (140.0, 200.0),
    "total_distance": (1.0, 100.0),
}

print("\nThresholds (min, max):")
for col, (min_val, max_val) in thresholds.items():
    print(f"  {col:20s}: {min_val:6.1f} - {max_val:6.1f}")

print("\n" + "="*70)
print("IMPACT OF FILTERING")
print("="*70)

df_clean = df_filtered.copy()
initial_rows = len(df_clean)

for col, (min_val, max_val) in thresholds.items():
    if col in df_clean.columns:
        before = len(df_clean)
        df_clean = df_clean[(df_clean[col] >= min_val) & (df_clean[col] <= max_val)]
        after = len(df_clean)
        removed = before - after
        pct = 100.0 * removed / before
        print(f"{col:20s}: {before:6d} → {after:6d} (removed {removed:5d}, {pct:5.2f}%)")

final_rows = len(df_clean)
total_removed = initial_rows - final_rows
total_pct = 100.0 * total_removed / initial_rows

print(f"\n{'TOTAL':20s}: {initial_rows:6d} → {final_rows:6d} (removed {total_removed:5d}, {total_pct:5.2f}%)")

print("\n" + "="*70)
print("POST-FILTERING DATA DISTRIBUTION")
print("="*70)

for col in ["pace_min_km", "average_hr", "final_cadence", "total_distance"]:
    if col in df_clean.columns:
        p1, p5, p25, p50, p75, p95, p99 = df_clean[col].quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        print(f"\n{col:20s}:")
        print(f"  p01={p1:10.2f}, p05={p5:10.2f}, p25={p25:10.2f}")
        print(f"  p50={p50:10.2f}, p75={p75:10.2f}, p95={p95:10.2f}, p99={p99:10.2f}")
        print(f"  min={df_clean[col].min():10.2f}, max={df_clean[col].max():10.2f}")
