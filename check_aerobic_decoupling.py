#!/usr/bin/env python3
import pandas as pd
import numpy as np

df = pd.read_csv("data/activities.csv")
df.columns = df.columns.str.strip()

print("="*70)
print("ANALIZA AEROBIC_DECOUPLING")
print("="*70)

# Sprawdzam wszystkie wartości
ad_col = df["aerobic_decoupling"]
print(f"\nTotal rows: {len(df)}")
print(f"Non-null: {ad_col.notna().sum()}")
print(f"NaN: {ad_col.isna().sum()}")
print(f"Dtype: {ad_col.dtype}")

# Wartości skrajne
print(f"\nValue statistics:")
print(f"  Min: {ad_col.min()}")
print(f"  Max: {ad_col.max()}")
print(f"  Mean: {ad_col.mean()}")
print(f"  Median: {ad_col.median()}")
print(f"  StdDev: {ad_col.std()}")

# Percentyle
print(f"\nPercentiles:")
for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    val = ad_col.quantile(p/100)
    print(f"  p{p:2d}: {val:10.2f}")

# Ile wartości poza 0-100
print(f"\nValue ranges:")
print(f"  < 0: {(ad_col < 0).sum()} rows ({100*(ad_col < 0).sum()/ad_col.notna().sum():.2f}%)")
print(f"  0-100: {((ad_col >= 0) & (ad_col <= 100)).sum()} rows ({100*((ad_col >= 0) & (ad_col <= 100)).sum()/ad_col.notna().sum():.2f}%)")
print(f"  > 100: {(ad_col > 100).sum()} rows ({100*(ad_col > 100).sum()/ad_col.notna().sum():.2f}%)")

# Sprawdzam czy valores większe są wielokrotnie większe
large_vals = ad_col[ad_col > 100]
print(f"\nValues > 100 statistics:")
print(f"  Count: {len(large_vals)}")
if len(large_vals) > 0:
    print(f"  Min: {large_vals.min()}")
    print(f"  Max: {large_vals.max()}")
    print(f"  Mean: {large_vals.mean()}")

# Sprawdzam rozkład
print(f"\nDistribution:")
print(f"  Values in 0-10: {((ad_col >= 0) & (ad_col <= 10)).sum()}")
print(f"  Values in 10-50: {((ad_col > 10) & (ad_col <= 50)).sum()}")
print(f"  Values in 50-100: {((ad_col > 50) & (ad_col <= 100)).sum()}")
print(f"  Values in 100-500: {((ad_col > 100) & (ad_col <= 500)).sum()}")
print(f"  Values in 500-1000: {((ad_col > 500) & (ad_col <= 1000)).sum()}")
print(f"  Values > 1000: {(ad_col > 1000).sum()}")

# Pokazuję sample
print(f"\nSample values > 100:")
samples = ad_col[ad_col > 100].dropna().head(20)
for idx, val in samples.items():
    print(f"  Row {idx}: {val:.2f}")
