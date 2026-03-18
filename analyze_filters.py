#!/usr/bin/env python
import pandas as pd

df_raw = pd.read_csv("data/activities.csv")
df_raw.columns = df_raw.columns.str.strip()

print(f"Raw data: {len(df_raw)} rows")

# Apply same logic as prepare_clean_data
CRUCIAL_COLUMNS = ["average_hr", "final_cadence", "average_speed", "total_distance"]
df = df_raw.dropna(subset=CRUCIAL_COLUMNS).copy()
print(f"After dropping nulls in crucial columns: {len(df)} rows")

df["pace_min_km"] = 60 / df["average_speed"]
df["aerobic_decoupling"] = df["aerobic_decoupling"].fillna(df["aerobic_decoupling"].median())

print(f"Before filters: {len(df)} rows\n")

# Apply filters one by one to see which removes most
df_test = df.copy()
removed_by_pace = df_test[(df_test["pace_min_km"] < 2.0) | (df_test["pace_min_km"] > 15.0)]
print(f"Removed by pace (2.0-15.0): {len(removed_by_pace)} rows ({100*len(removed_by_pace)/len(df_test):.2f}%)")

df_test = df[(df["pace_min_km"] >= 2.0) & (df["pace_min_km"] <= 15.0)]
removed_by_hr = df_test[(df_test["average_hr"] < 40.0) | (df_test["average_hr"] > 220.0)]
print(f"Removed by HR (40-220): {len(removed_by_hr)} rows ({100*len(removed_by_hr)/len(df_test):.2f}%)")

df_test = df_test[(df_test["average_hr"] >= 40.0) & (df_test["average_hr"] <= 220.0)]
removed_by_cad = df_test[(df_test["final_cadence"] < 60.0) | (df_test["final_cadence"] > 180.0)]
print(f"Removed by cadence (60-180): {len(removed_by_cad)} rows ({100*len(removed_by_cad)/len(df_test):.2f}%)")

df_test = df_test[(df_test["final_cadence"] >= 60.0) & (df_test["final_cadence"] <= 180.0)]
removed_by_decay = df_test[(df_test["aerobic_decoupling"] < -150.0) | (df_test["aerobic_decoupling"] > 150.0)]
print(f"Removed by aerobic_decoupling (-150 to 150): {len(removed_by_decay)} rows ({100*len(removed_by_decay)/len(df_test):.2f}%)")

df_final = df_test[(df_test["aerobic_decoupling"] >= -150.0) & (df_test["aerobic_decoupling"] <= 150.0)]
print(f"\n{'='*70}")
print(f"After all filters: {len(df_final)} rows")
print(f"Total removed: {len(df) - len(df_final)} rows ({100*(len(df)-len(df_final))/len(df):.2f}%)")
print(f"Retained: {100*len(df_final)/len(df):.2f}%")
print(f"{'='*70}")

print("\nPOST-FILTER VALUE RANGES:")
print("="*70)
for col in ["pace_min_km", "average_hr", "final_cadence", "aerobic_decoupling"]:
    print(f"\n{col:25s}:")
    print(f"  Range: {df_final[col].min():8.2f} - {df_final[col].max():8.2f}")
    print(f"  Mean: {df_final[col].mean():8.2f}, Median: {df_final[col].median():8.2f}")
