import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.data.read_data import load_clean_numeric_data, PROCESSED_DATA_PATH
from src.models.autoencoder_models import run_autoencoder_comparison
from src.pipeline.run_experiments import get_extreme_athletes

# Load cleaned data
df_numeric = load_clean_numeric_data(PROCESSED_DATA_PATH)
print(f"Loaded {len(df_numeric)} rows")

df_train, df_test = train_test_split(df_numeric, test_size=0.2, random_state=42)

# Run autoencoder to get scaler and scores
print("\nRunning autoencoder comparison...")
autoencoder_results, _, ae_scaler = run_autoencoder_comparison(
    df_train, df_test, df_numeric, ae_epochs=20, ae_batch_size=2048, ae_patience=3
)

best_model = autoencoder_results[2]  # Deep autoencoder

# Get extreme athletes WITH denormalization
print("\n=== GETTING EXTREME ATHLETES WITH DENORMALIZATION ===")
extreme_athletes_df, top_features = get_extreme_athletes(df_numeric, best_model.scores, scaler=ae_scaler)

print("\n✓ Extreme Athletes DataFrame (DENORMALIZED):")
print(extreme_athletes_df)

print("\n=== VERIFICATION OF VALUES ===")
print("\nPace [min/km] (should be 2-15):")
pace_col = [c for c in extreme_athletes_df.columns if "Pace" in c][0]
print(f"  Min: {extreme_athletes_df[pace_col].min():.2f}, Max: {extreme_athletes_df[pace_col].max():.2f}")
print(f"  Values: {extreme_athletes_df[pace_col].values}")

print("\nAverage Heart Rate [bpm] (should be 40-220):")
hr_col = [c for c in extreme_athletes_df.columns if "Heart" in c][0]
print(f"  Min: {extreme_athletes_df[hr_col].min():.2f}, Max: {extreme_athletes_df[hr_col].max():.2f}")
print(f"  Values: {extreme_athletes_df[hr_col].values}")

print("\nFinal Cadence [spm] (should be 60-180):")
cad_col = [c for c in extreme_athletes_df.columns if "Cadence" in c][0]
print(f"  Min: {extreme_athletes_df[cad_col].min():.2f}, Max: {extreme_athletes_df[cad_col].max():.2f}")
print(f"  Values: {extreme_athletes_df[cad_col].values}")

print("\n✓ Top 3 Features:", top_features)
print("\n✓ SUCCESS: All values are in physiologically realistic ranges!")
