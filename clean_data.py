import pandas as pd
import numpy as np
import os
import fetch_data

# --- CONFIGURATION ---
MIN_SPEED_KMH = 4.0
MAX_SPEED_KMH = 35.0
MIN_DIST_KM = 2.0
MAX_DIST_KM = 100.0
MIN_DURATION_MIN = 10.0


def clean_data():
    # 1. Ensure data exists
    fetch_data.ensure_data_exists()

    print("Loading and cleaning data...")

    try:
        df = pd.read_csv('activities.csv', low_memory=False)
    except FileNotFoundError:
        print("Error: activities.csv not found.")
        return

    print(f"Raw rows: {len(df)}")

    # 2. TYPE CONVERSION
    numeric_cols = [
        'total_distance', 'elevation_gain', 'average_speed',
        'average_hr', 'final_cadence', 'workout_time',
        'athlete_weight', 'aerobic_decoupling', 'yob'
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # 3. CALCULATE AGE
    if 'yob' in df.columns:
        df['activity_year'] = df['date'].dt.year
        df['age'] = df['activity_year'] - df['yob']
    else:
        df['age'] = np.nan

    # 4. FILTERING
    # We strictly need HR, Speed, Age, Weight now
    subset_cols = ['total_distance', 'average_speed', 'average_hr', 'age', 'athlete_weight']
    df.dropna(subset=[c for c in subset_cols if c in df.columns], inplace=True)

    mask = (
            (df['average_speed'].between(MIN_SPEED_KMH, MAX_SPEED_KMH)) &
            (df['average_hr'].between(40, 230)) &
            (df['total_distance'].between(MIN_DIST_KM, MAX_DIST_KM)) &
            (df['workout_time'] > (MIN_DURATION_MIN * 60)) &
            (df['age'].between(10, 90)) &
            (df['athlete_weight'].between(30, 150))
    )
    df_clean = df[mask].copy()

    # 5. FEATURE ENGINEERING

    # Pace (min/km)
    df_clean['pace_min_km'] = 60 / df_clean['average_speed']

    # Cadence Fix
    if 'final_cadence' in df_clean.columns:
        df_clean['cadence_spm'] = df_clean['final_cadence'].apply(
            lambda x: x * 2 if (pd.notnull(x) and x > 0 and x < 120) else x
        )
        df_clean = df_clean[df_clean['cadence_spm'].between(120, 260)]
    else:
        df_clean['cadence_spm'] = np.nan

    # Stride Length
    speed_m_min = df_clean['average_speed'] * 1000 / 60
    df_clean['stride_length_m'] = speed_m_min / df_clean['cadence_spm']
    df_clean = df_clean[df_clean['stride_length_m'].between(0.4, 2.6)]

    # Aerobic Decoupling
    if 'aerobic_decoupling' in df_clean.columns:
        median_dec = df_clean['aerobic_decoupling'].median()
        if pd.isna(median_dec): median_dec = 3.0
        df_clean['aerobic_decoupling'] = df_clean['aerobic_decoupling'].fillna(median_dec)
        df_clean['aerobic_decoupling'] = df_clean['aerobic_decoupling'].clip(-10, 20)
    else:
        df_clean['aerobic_decoupling'] = 3.0

    # Gender
    if 'gender' in df_clean.columns:
        df_clean['is_male'] = df_clean['gender'].astype(str).apply(
            lambda x: 1 if x.upper().startswith('M') else 0
        )
    else:
        df_clean['is_male'] = 1

    if 'elevation_gain' not in df_clean.columns: df_clean['elevation_gain'] = 0
    df_clean['elevation_gain'] = df_clean['elevation_gain'].fillna(0)

    # 6. SAVE FINAL DATASET
    # UPDATED: Included average_hr, removed efficiency_index
    final_cols = [
        'pace_min_km',  # Performance
        'average_hr',  # Cost (New!)
        'aerobic_decoupling',  # Endurance
        'cadence_spm',  # Technique
        'stride_length_m',  # Power
        'athlete_weight',  # Context
        'age',  # Context
        'is_male',  # Context
        'elevation_gain',  # Context
        'total_distance'  # Context
    ]

    final_cols = [c for c in final_cols if c in df_clean.columns]
    df_clean.dropna(subset=final_cols, inplace=True)

    df_clean.to_csv('ready_to_train.csv', columns=final_cols, index=False)

    print("\n" + "=" * 40)
    print(f"Saved 'ready_to_train.csv'")
    print(f"Rows: {len(df_clean)}")
    print("=" * 40)

    print(df_clean[['pace_min_km', 'average_hr', 'athlete_weight']].head().round(2))


if __name__ == "__main__":
    clean_data()