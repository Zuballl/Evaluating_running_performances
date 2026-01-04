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
    if pd.isna(decimal_pace) or decimal_pace == 0: return "0:00"
    minutes = int(decimal_pace)
    seconds = int((decimal_pace - minutes) * 60)
    return f"{minutes}:{seconds:02d}"


def get_file_from_user():
    """Asks user for filename or reads from command line args."""
    if len(sys.argv) > 1:
        candidate = sys.argv[1].replace("'", "").replace('"', "").strip()
        if os.path.exists(candidate):
            return candidate
        else:
            print(f"File provided in argument not found: {candidate}")

    while True:
        print("\nDrag & Drop your .FIT file here (or type the name):")
        filename = input(">>> ").strip().replace("'", "").replace('"', "")
        if not filename: continue
        if os.path.exists(filename): return filename
        print(f"File not found: {filename}")


def get_fit_metadata(fit_file_path):
    """
    Extracts User Profile data (Age, Gender, Weight) directly from the FIT file.
    Uses StandardUnitsDataProcessor to ensure correct units.
    """
    metadata = {}
    try:
        # Use StandardUnitsDataProcessor to get kg instead of raw values
        fitfile = fitparse.FitFile(fit_file_path, data_processor=fitparse.StandardUnitsDataProcessor())

        # 1. Search in user_profile (Standard location)
        for record in fitfile.get_messages('user_profile'):
            data = record.get_values()
            if 'age' in data and data['age'] is not None:
                metadata['age'] = float(data['age'])
            if 'weight' in data and data['weight'] is not None:
                metadata['weight'] = float(data['weight'])
            if 'gender' in data and data['gender'] is not None:
                val = str(data['gender']).lower()
                if 'female' in val:
                    metadata['gender'] = 0
                elif 'male' in val:
                    metadata['gender'] = 1

        # 2. Fallback: Search in session/sport messages if still missing
        if 'weight' not in metadata:
            for record in fitfile.get_messages('session'):
                data = record.get_values()
                if 'total_weight' in data:  # Some devices store it here
                    metadata['weight'] = float(data['total_weight'])

    except Exception:
        pass

    return metadata


def get_user_context(fit_meta=None):
    """
    Asks for Age, Weight, and Gender ONLY if not found in file.
    """
    print("\n--- Context Data ---")

    if fit_meta is None:
        fit_meta = {}

    # 1. AGE
    if 'age' in fit_meta:
        age = fit_meta['age']
        print(f"  Age found in file: {int(age)}")
    else:
        try:
            val = input("   Enter your Age [Default: 30]: ").strip()
            age = float(val) if val else 30.0
        except ValueError:
            print("Invalid age. Using default 30.")
            age = 30.0

    # 2. WEIGHT
    if 'weight' in fit_meta:
        weight = fit_meta['weight']
        print(f"  Weight found in file: {weight:.1f} kg")
    else:
        try:
            val = input("   Enter your Weight (kg) [Default: 70]: ").strip()
            weight = float(val) if val else 70.0
        except ValueError:
            print("Invalid weight. Using default 70kg.")
            weight = 70.0

    # 3. GENDER
    if 'gender' in fit_meta:
        is_male = fit_meta['gender']
        gender_str = "Male" if is_male == 1 else "Female"
        print(f"   ✅ Gender found in file: {gender_str}")
    else:
        gender_input = input("   Enter your Gender (M/F) [Default: M]: ").strip().upper()
        is_male = 0 if gender_input.startswith('F') else 1

    return age, weight, is_male


def get_run_metrics(fit_file_path):
    print(f"⚙️  Processing file: {fit_file_path}...")
    try:
        # Standard units are important for speed (m/s) etc
        fitfile = fitparse.FitFile(fit_file_path, data_processor=fitparse.StandardUnitsDataProcessor())
        records = []
        for record in fitfile.get_messages('record'):
            records.append(record.get_values())

        if not records:
            print("Error: FIT file contains no recording data.")
            return None

        df = pd.DataFrame(records)
    except Exception as e:
        print(f"Error opening FIT file: {e}")
        return None

    # Handle Speed Column Name
    speed_col = 'enhanced_speed' if 'enhanced_speed' in df.columns else 'speed'
    hr_col = 'heart_rate'

    if speed_col not in df.columns or hr_col not in df.columns:
        print(f"Error: File is missing Speed ('{speed_col}') or Heart Rate ('{hr_col}').")
        return None

    # Filter Active Running (> 0.5 m/s)
    df_active = df[df[speed_col] > 0.5].copy()
    if df_active.empty:
        print("No active running data found (speed > 0.5 m/s).")
        return None

    metrics = {}

    # --- 1. BASIC METRICS ---
    avg_speed_kmh = df_active[speed_col].mean() * 3.6
    metrics['pace_min_km'] = 60 / avg_speed_kmh
    metrics['average_speed'] = avg_speed_kmh
    metrics['average_hr'] = df_active[hr_col].mean()

    # --- 2. AEROBIC DECOUPLING ---
    if len(df_active) > 60:
        half_idx = len(df_active) // 2
        ef1 = (df_active.iloc[:half_idx][speed_col].mean() * 3.6) / df_active.iloc[:half_idx][hr_col].mean()
        ef2 = (df_active.iloc[half_idx:][speed_col].mean() * 3.6) / df_active.iloc[half_idx:][hr_col].mean()
        # Positive decoupling = Efficiency dropped = Drift
        metrics['aerobic_decoupling'] = ((ef1 - ef2) / ef1) * 100
    else:
        metrics['aerobic_decoupling'] = 0.0

    # --- 3. MECHANICS & CONTEXT ---
    if 'cadence' in df_active.columns:
        cad = df_active['cadence'].mean()
        if cad < 120: cad *= 2  # Fix single leg cadence
        metrics['cadence_spm'] = cad
    else:
        metrics['cadence_spm'] = 170

    metrics['stride_length_m'] = (avg_speed_kmh * 1000 / 60) / metrics['cadence_spm'] if metrics[
                                                                                             'cadence_spm'] > 0 else 0

    alt_col = 'enhanced_altitude' if 'enhanced_altitude' in df.columns else 'altitude'
    metrics['elevation_gain'] = df[alt_col].diff().clip(lower=0).sum() if alt_col in df.columns else 0
    metrics['total_distance'] = df['distance'].max() / 1000.0 if 'distance' in df.columns else 0

    return metrics


def score_new_run():
    print(f"Loading model: {MODEL_FILE}...")
    try:
        model = joblib.load(MODEL_FILE)
    except FileNotFoundError:
        print(f"File {MODEL_FILE} not found. Run training script first!")
        return

    # 1. Get Data
    run_file = get_file_from_user()

    # Try to get metadata from file first to autofill context
    fit_meta = get_fit_metadata(run_file)
    age, weight, is_male = get_user_context(fit_meta)

    metrics = get_run_metrics(run_file)
    if metrics is None: return

    # Add context data to metrics dict
    metrics['is_male'] = is_male
    metrics['age'] = age
    metrics['athlete_weight'] = weight

    # 2. Prepare Feature Vector
    # MUST match train_model.py features list order EXACTLY
    feature_order = [
        'pace_min_km',  # 1
        'average_hr',  # 2
        'aerobic_decoupling',  # 3
        'cadence_spm',  # 4
        'stride_length_m',  # 5
        'athlete_weight',  # 6
        'age',  # 7
        'is_male',  # 8
        'elevation_gain',  # 9
        'total_distance'  # 10
    ]

    input_values = []
    for f in feature_order:
        val = metrics.get(f, 0)
        if pd.isna(val) or np.isinf(val): val = 0
        input_values.append(val)

    X = np.array([input_values])

    # 3. Transform & Score
    if X.shape[1] != len(model['scaler'].mean_):
        print(f"Dimension Error: Model expects {len(model['scaler'].mean_)} features, got {X.shape[1]}.")
        print("   Did you re-run train_model.py after updating clean_data.py?")
        return

    X_scaled = model['scaler'].transform(X)
    raw_score = model['pca'].transform(X_scaled).flatten()[0]

    # Correct Sign (Using Pace logic from training)
    # Check Pace weight (index 0). If Positive -> Flip.
    pace_idx = feature_order.index('pace_min_km')
    pace_weight = model['pca'].components_[0][pace_idx]

    if pace_weight > 0:
        raw_score = -raw_score

    clipped = np.clip(raw_score, model['lower_bound'], model['upper_bound'])
    final_score = model['final_scaler'].transform([[clipped]])[0][0]

    # 4. Feature Importance
    weights = np.abs(model['pca'].components_[0])
    total_importance = np.sum(weights)

    contributions = []
    for i, feature_name in enumerate(feature_order):
        pct = (weights[i] / total_importance) * 100
        contributions.append((feature_name, pct))
    contributions.sort(key=lambda x: x[1], reverse=True)

    # 5. Report
    print("\n" + "=" * 50)
    print(f"🏆 PERFORMANCE SCORE: {final_score:.2f} / 10.0")
    print("=" * 50)

    print(f"📊 Run Statistics:")
    print(f"   • Pace:        {format_pace(metrics['pace_min_km'])} /km")
    print(f"   • Avg HR:      {metrics['average_hr']:.0f} bpm")
    print(f"   • Decoupling:  {metrics['aerobic_decoupling']:.1f}%")
    print(f"   • Cadence:     {metrics['cadence_spm']:.0f} spm")
    print(f"   • Distance:    {metrics['total_distance']:.2f} km")

    print("\n⚖️  AI Weighting (What mattered most?):")
    for label, pct in contributions:
        nice_label = label.replace('_', ' ').title().replace('Spm', '').replace('Min Km', '')
        bar = "█" * int(pct / 4)
        print(f"   {nice_label:<20} {pct:4.1f}% {bar}")

    print("\n💡 AI COACH SAYS:")
    if final_score > 8:
        print("   Elite level! You have a huge engine and great efficiency.")
    elif final_score > 6:
        print("   Strong performance. Above average aerobic base.")
    elif final_score > 4:
        print("   Solid amateur run. Good consistency.")
    else:
        print("   Foundation level. Focus on building aerobic base (Zone 2 training).")


if __name__ == "__main__":
    score_new_run()