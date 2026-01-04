import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib

# CONFIGURATION
INPUT_FILE = 'ready_to_train.csv'  # Matches output from clean_data.py
MODEL_FILE = 'performance_scorer.pkl'


def train_model():
    print(f"Loading data: {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: File {INPUT_FILE} not found. Run clean_data.py first.")
        return

    # 1. DEFINE FEATURES
    # Must match the columns exported by clean_data.py exactly
    features = [
        'pace_min_km',  # Performance (Lower is better)
        'average_hr',  # Cost (Lower is better)
        'aerobic_decoupling',  # Endurance (Lower is better)
        'cadence_spm',  # Mechanics (Optimal range implies better tech)
        'stride_length_m',  # Power (Longer stride at same cadence = faster)
        'athlete_weight',  # Context (Heavier = higher energy cost)
        'age',  # Context (Physiological decline)
        'is_male',  # Context (Physiological differences)
        'elevation_gain',  # Context (Difficulty)
        'total_distance'  # Context (Endurance volume)
    ]

    # Validate features existence
    available_features = [f for f in features if f in df.columns]
    print(f"Training on {len(available_features)} features: {available_features}")

    if len(available_features) != len(features):
        missing = set(features) - set(available_features)
        print(f"Warning: Missing features: {missing}")

    # Clean missing values for training
    df_clean = df.dropna(subset=available_features).copy()

    # Extra sanity check for outliers before training (e.g. pace limits)
    if 'pace_min_km' in df_clean.columns:
        df_clean = df_clean[df_clean['pace_min_km'].between(2.5, 12.0)].copy()

    X = df_clean[available_features].values

    print(f"Training PCA on {len(df_clean)} samples...")

    # 2. SCALING
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. PCA (Principal Component Analysis)
    # Finding the single axis that explains the most variance (Performance Factor)
    pca = PCA(n_components=1)
    principal_components = pca.fit_transform(X_scaled)

    # 4. DIRECTION CORRECTION
    # We want High Score = High Performance.
    # High Performance is strongly correlated with LOW Pace (Fast).
    # So, correlation between Score and Pace should be NEGATIVE.

    raw_score = principal_components.flatten()

    # Get the weight (loading) of Pace in the first component
    # The loading tells us correlation direction directly
    pace_idx = available_features.index('pace_min_km')
    pace_weight = pca.components_[0][pace_idx]

    print(f"DEBUG: Pace Weight in PCA: {pace_weight:.4f}")

    # If Pace Weight is POSITIVE, it means High Score -> High Pace (Slow).
    # We want High Score -> Low Pace (Fast).
    if pace_weight > 0:
        print("Flipping PCA sign (Enforcing: Lower Pace = Better Score)...")
        raw_score = -raw_score
        pca.components_ = -pca.components_
    else:
        print("PCA direction is correct (Lower Pace = Better Score).")

    # 5. SCALING TO 0-10 RANGE
    # Using percentiles to be robust against extreme outliers
    lower = np.percentile(raw_score, 1)
    upper = np.percentile(raw_score, 99)
    clipped_score = np.clip(raw_score, lower, upper)

    final_scaler = MinMaxScaler(feature_range=(0, 10))
    final_scores = final_scaler.fit_transform(clipped_score.reshape(-1, 1)).flatten()

    df_clean['ai_score'] = final_scores

    # 6. SAVE MODEL
    model_bundle = {
        'scaler': scaler,
        'pca': pca,
        'features': available_features,
        'lower_bound': lower,
        'upper_bound': upper,
        'final_scaler': final_scaler
    }
    joblib.dump(model_bundle, MODEL_FILE)
    print(f"Model saved to: {MODEL_FILE}")

    # 7. VISUALIZATION (Feature Importance)
    loadings = pd.DataFrame(
        pca.components_.T,
        index=available_features,
        columns=['Weight']
    )
    # Sort by absolute impact
    loadings['Abs_Weight'] = loadings['Weight'].abs()
    loadings = loadings.sort_values(by='Abs_Weight', ascending=False)

    print("\n--- FEATURE IMPORTANCE (Weights) ---")
    print(loadings[['Weight']])

    plt.figure(figsize=(12, 6))
    sns.barplot(x=loadings['Weight'], y=loadings.index, palette='coolwarm_r')
    plt.title("Feature Impact on Score (Left=Negative, Right=Positive)")
    plt.axvline(0, color='black', linewidth=1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    train_model()