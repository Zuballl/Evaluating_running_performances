#!/usr/bin/env python3
import pandas as pd

df = pd.read_csv("data/activities.csv")
df.columns = df.columns.str.strip()

ad = df["aerobic_decoupling"]

# Pokazuję sample negatywnych wartości
print("Sample of NEGATIVE aerobic_decoupling values:")
print("="*70)
neg_vals = ad[ad < 0].dropna().head(20).values
for val in neg_vals:
    print(f"  {val:.2e}")

print("\n" + "="*70)
print("Unique NEGATIVE threshold values:")
print("="*70)

# Sprawdzam czy są stałe progi
neg_unique = sorted(ad[ad < 0].unique())
print(f"Total unique negative values: {len(neg_unique)}")

# Sprawdzam czy są gęste wartości ujemne
print("\nMost common negative values:")
neg_counts = ad[ad < 0].value_counts().head(10)
for val, count in neg_counts.items():
    print(f"  {val:.2e}: {count} rows")

# Może -1 oznacza brak pomiaru?
print(f"\nValues == -1.0: {(ad == -1.0).sum()}")
print(f"Values < -1.0: {(ad < -1.0).sum()}")

# Sprawdzam co jest między -1 a 0
print(f"\nValues between -1 and 0 (exclusive): {((ad > -1.0) & (ad < 0)).sum()}")

print("\n" + "="*70)
print("Normalized distribution (without negatives):")
print("="*70)
ad_positive = ad[ad >= 0]
print(f"Rows with aerobic_decoupling >= 0: {len(ad_positive)}")
print(f"Min: {ad_positive.min():.2f}")
print(f"Max: {ad_positive.max():.2f}")
print(f"Mean: {ad_positive.mean():.2f}")
print(f"Median: {ad_positive.median():.2f}")
