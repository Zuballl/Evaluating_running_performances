import fitparse
import pandas as pd
import numpy as np
import joblib
import os
import sys

# --- CONFIGURATION ---
MODEL_FILE = 'performance_scorer.pkl'


def format_pace(decimal_pace):
    """Converts 5.5 min/km to 5:30 min/km"""
    minutes = int(decimal_pace)
    seconds = int((decimal_pace - minutes) * 60)
    return f"{minutes}:{seconds:02d}"


def get_file_from_user():
    """Asks user for filename or reads from command line args."""
    # 1. Check if file was provided as command line argument
    if len(sys.argv) > 1:
        candidate = sys.argv[1]
        # Remove potential quotes from drag & drop
        candidate = candidate.replace("'", "").replace('"', "").strip()
        if os.path.exists(candidate):
            return candidate
        else:
            print(f"File provided in argument not found: {candidate}")

    # 2. Interactive input loop
    while True:
        print("\nDrag & Drop your .FIT file here (or type the name):")
        filename = input(">>> ").strip()

        # Clean quotes
        filename = filename.replace("'", "").replace('"', "")

        if not filename:
            continue

        if os.path.exists(filename):
            return filename
        else:
            print(f"File not found: {filename}")
            print("(Ensure the file is in the folder or provide full path)")


def get_run_metrics(fit_file_path):
    print(f"Reading file: {fit_file_path}")
    try:
        fitfile = fitparse.FitFile(fit_file_path)
    except Exception as e:
        print(f"Error opening FIT file: {e}")
        return None

    records = []
    for record in fitfile.get_messages('record'):
        records.append(record.get_values())

    if not records: return None
    df = pd.DataFrame(records)

    # Check if we have enhanced_speed
    if 'enhanced_speed' not in df.columns:
        print("No speed data found.")
        return None

    # Activity filter (running > 0.5 m/s)
    df_active = df[df['enhanced_speed'] > 0.5].copy()
    if df_active.empty:
        print("No active running data found (speed > 0.5 m/s).")
        return None

    metrics = {}

    # 1. SPEED AND PACE
    avg_speed_kmh = df_active['enhanced_speed'].mean() * 3.6
    metrics['pace_min_km'] = 60 / avg_speed_kmh

    # 2. HEART RATE AND EFFICIENCY
    avg_hr = df_active['heart_rate'].mean() if 'heart_rate' in df_active.columns else df['heart_rate'].mean()
    if pd.isna(avg_hr) or avg_hr == 0: avg_hr = 145  # Fallback

    metrics['efficiency_index'] = avg_speed_kmh / avg_hr
    metrics['display_hr'] = avg_hr

    # 3. CADENCE
    cad = df_active['cadence'].mean() if 'cadence' in df_active.columns else 170
    if cad < 120: cad *= 2
    metrics['cadence_spm'] = cad

    # 4. OTHER
    metrics['elevation_gain'] = df['enhanced_altitude'].diff().clip(
        lower=0).sum() if 'enhanced_altitude' in df.columns else 0
    metrics['total_distance'] = df['distance'].max() / 1000.0 if 'distance' in df.columns else 0
    metrics['stride_length_m'] = (avg_speed_kmh * 1000 / 60) / metrics['cadence_spm']

    return metrics


def score_new_run():
    print(f"Loading model: {MODEL_FILE}...")
    try:
        model = joblib.load(MODEL_FILE)
    except FileNotFoundError:
        print(f"File {MODEL_FILE} not found. Run training script first!")
        return

    # --- GET FILENAME FROM USER ---
    run_file = get_file_from_user()

    metrics = get_run_metrics(run_file)
    if metrics is None:
        print("Failed to extract metrics from FIT file.")
        return

    # Feature list MUST have exactly 6 elements
    feature_order = [
        'pace_min_km',
        'efficiency_index',
        'cadence_spm',
        'stride_length_m',
        'elevation_gain',
        'total_distance'
    ]

    # Building input vector
    input_values = []
    for f in feature_order:
        val = metrics.get(f, 0)
        if pd.isna(val): val = 0
        input_values.append(val)

    X = np.array([input_values])

    # Safety check
    expected_features = len(model['scaler'].mean_)
    if X.shape[1] != expected_features:
        print(f"DIMENSION ERROR: Model expects {expected_features} features.")
        return

    # Transformation
    X_scaled = model['scaler'].transform(X)
    raw_score = model['pca'].transform(X_scaled).flatten()[0]

    # Sign correction
    pace_idx = feature_order.index('pace_min_km')
    pace_weight = model['pca'].components_[0][pace_idx]
    if pace_weight > 0:
        raw_score = -raw_score

    clipped = np.clip(raw_score, model['lower_bound'], model['upper_bound'])
    final_score = model['final_scaler'].transform([[clipped]])[0][0]

    # --- CALCULATE PERCENTAGE CONTRIBUTIONS ---
    weights = model['pca'].components_[0]
    abs_weights = np.abs(weights)
    total_importance = np.sum(abs_weights)

    labels_map = {
        'pace_min_km': 'Pace (Tempo)',
        'efficiency_index': 'Efficiency (Speed/HR)',
        'cadence_spm': 'Cadence',
        'stride_length_m': 'Stride Length',
        'elevation_gain': 'Elevation',
        'total_distance': 'Distance'
    }

    contributions = []
    for i, feature_name in enumerate(feature_order):
        pct = (abs_weights[i] / total_importance) * 100
        label = labels_map.get(feature_name, feature_name)
        contributions.append((label, pct))

    contributions.sort(key=lambda x: x[1], reverse=True)

    # --- PRINT REPORT ---
    print("\n" + "=" * 40)
    print(f"YOUR SCORE: {final_score:.2f} / 10.0")
    print("=" * 40)

    print(f"🚀 Pace:       {format_pace(metrics['pace_min_km'])} /km")
    print(f"❤️ Heart Rate: {metrics['display_hr']:.0f} bpm")
    print(f"⚡ Efficiency: {metrics['efficiency_index']:.2f}")
    print(f"👣 Cadence:    {metrics['cadence_spm']:.0f} spm")
    print(f"📏 Distance:   {metrics['total_distance']:.2f} km")

    print("\n📊 SCORE BREAKDOWN (What matters most?):")
    for label, pct in contributions:
        bar_len = int(pct / 5)
        bar = "█" * bar_len
        print(f"   • {label:<20} {pct:5.1f}%  {bar}")

    print("\n💡 INTERPRETATION:")
    if final_score > 8:
        print("Master Class. Very fast pace with low physiological cost.")
    elif final_score > 6:
        print("Great form. Above population average.")
    elif final_score > 4:
        print("Solid amateur level. Good base.")
    else:
        print("Recovery run or early stage of building form.")


if __name__ == "__main__":
    score_new_run()