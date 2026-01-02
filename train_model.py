import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib

# CONFIGURATION
INPUT_FILE = 'ready_to_train.csv'
MODEL_FILE = 'performance_scorer.pkl'


def train_smart_model():
    print(f"Loading data: {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print("File ready_to_train.csv not found")
        return

    # 1. CHANGE: REMOVING 'average_hr'
    # We keep only efficiency_index, which "penalizes" high heart rate relative to pace.
    features = [
        'pace_min_km',       # Pace (LOWER is better)
        # 'average_hr',      # REMOVED - it confused the model (promoted intensity)
        'efficiency_index',  # Efficiency (Now the only guardrail for HR)
        'cadence_spm',       # Cadence
        'stride_length_m',   # Stride Length
        'elevation_gain',    # Elevation Gain
        'total_distance'     # Total Distance
    ]

    available_features = [f for f in features if f in df.columns]
    print(f"Training on SMART feature set: {available_features}")

    df_clean = df.dropna(subset=available_features).copy()
    # Filter for reasonable paces
    df_clean = df_clean[df_clean['pace_min_km'].between(2.5, 12.0)].copy()

    X = df_clean[available_features].values

    # 2. PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=1)
    principal_components = pca.fit_transform(X_scaled)

    # 3. DIRECTION CORRECTION
    # Check correlation with pace.
    # We expect: High Score = Low Pace (min/km). So correlation must be NEGATIVE.
    # If correlation is positive (score increases with time per km), flip the sign.

    raw_score = principal_components.flatten()
    correlation = np.corrcoef(raw_score, df_clean['pace_min_km'])[0, 1]

    if correlation > 0:
        print("Flipping PCA sign (alignment: faster = better)...")
        raw_score = -raw_score
        pca.components_ = -pca.components_

    # 4. SCALING
    lower = np.percentile(raw_score, 1)
    upper = np.percentile(raw_score, 99)
    clipped_score = np.clip(raw_score, lower, upper)

    final_scaler = MinMaxScaler(feature_range=(0, 10))
    final_scores = final_scaler.fit_transform(clipped_score.reshape(-1, 1)).flatten()

    df_clean['ai_score'] = final_scores

    # 5. SAVING
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

    # 6. VISUALIZATION
    loadings = pd.DataFrame(
        pca.components_.T,
        index=available_features,
        columns=['Weight']
    )
    loadings = loadings.sort_values(by='Weight', ascending=False)

    print("\n--------- NEW WEIGHTS ---------")
    print(loadings)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=loadings['Weight'], y=loadings.index, palette='coolwarm_r')
    plt.title("SMART Model: Efficiency should be high, Pace low")
    plt.axvline(0, color='black')
    plt.show()


if __name__ == "__main__":
    train_smart_model()