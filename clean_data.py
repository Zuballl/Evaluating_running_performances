import pandas as pd
import numpy as np

# --- FILTER CONFIGURATION (SAFETY LIMITS) ---
# Heart Rate limits
MIN_HR = 40
MAX_HR_VALUE = 230

# Speed limits (km/h)
MIN_SPEED_KMH = 4.0  # Below this is walking/standing
MAX_SPEED_KMH = 35.0  # Above this is likely biking or GPS error

# Distance limits (km)
MIN_DIST_KM = 0.5  # Reject very short "test" activities
MAX_DIST_KM = 100.0  # Reject ultras or driving (unless targeting ultra)

# Duration limits (minutes)
MIN_DURATION_MIN = 3.0
MAX_DURATION_MIN = 600.0  # 10 hours limit

# Elevation Gain limits (meters)
MAX_ELEV_GAIN = 4000.0  # Reject barometer errors/spikes

# Cadence limits (steps per minute)
MIN_CADENCE = 120
MAX_CADENCE = 260


def clean_activities():
    print("Loading activities.csv...")

    # 1. COLUMN DEFINITION
    # We only load columns that are relevant for our model
    cols_to_load = [
        'date', 'sport', 'workout_time', 'total_distance',
        'elevation_gain', 'average_speed', 'average_hr', 'average_cad'
    ]

    try:
        # Try to load only specific columns to save memory
        df = pd.read_csv('activities.csv', usecols=lambda c: c in cols_to_load, low_memory=False)
    except ValueError:
        # Fallback: load everything if column names don't match exactly
        print("Warning: Column mismatch. Loading full file...")
        df = pd.read_csv('activities.csv', low_memory=False)

    print(f"Raw rows loaded: {len(df)}")

    # 2. SPORT FILTERING
    # We are only interested in running activities
    running_keywords = ['Run', 'Running', 'Jogging', 'Street Run', 'Trail Run', 'Intervals']
    if 'sport' in df.columns:
        df = df[df['sport'].isin(running_keywords)].copy()
        print(f"Running activities found: {len(df)}")

    # 3. TYPE CONVERSION & CLEANING
    numeric_cols = ['total_distance', 'average_hr', 'average_speed',
                    'average_cad', 'workout_time', 'elevation_gain']

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Remove rows where critical data is missing
    required = ['total_distance', 'average_hr', 'average_speed', 'workout_time']
    df.dropna(subset=[c for c in required if c in df.columns], inplace=True)

    # Handle missing elevation (fill with 0, so we don't lose the run, but filter spikes later)
    # FIX APPLIED HERE: Direct assignment instead of inplace=True
    if 'elevation_gain' not in df.columns:
        df['elevation_gain'] = 0
    else:
        df['elevation_gain'] = df['elevation_gain'].fillna(0)

    # 4. CADENCE FIX (RPM -> SPM)
    if 'average_cad' in df.columns:
        def fix_cadence(cad):
            if pd.isna(cad) or cad == 0: return np.nan
            if cad < 120: return cad * 2  # Assumption: <120 is single-leg count (RPM)
            return cad

        df['cadence_spm'] = df['average_cad'].apply(fix_cadence)
    else:
        df['cadence_spm'] = np.nan

    # Calculate duration in minutes for filtering
    df['duration_min'] = df['workout_time'] / 60

    # 5. MAIN FILTERING (APPLYING LIMITS)
    # Creating boolean masks for readability
    mask_speed = df['average_speed'].between(MIN_SPEED_KMH, MAX_SPEED_KMH)
    mask_hr = df['average_hr'].between(MIN_HR, MAX_HR_VALUE)
    mask_dist = df['total_distance'].between(MIN_DIST_KM, MAX_DIST_KM)
    mask_time = df['duration_min'].between(MIN_DURATION_MIN, MAX_DURATION_MIN)
    mask_elev = df['elevation_gain'] < MAX_ELEV_GAIN

    # Apply all filters
    df_clean = df[mask_speed & mask_hr & mask_dist & mask_time & mask_elev].copy()

    # Cadence Filter: Keep rows where cadence is valid OR is NaN (missing sensor)
    if 'cadence_spm' in df_clean.columns:
        cadence_valid = df_clean['cadence_spm'].isna() | df_clean['cadence_spm'].between(MIN_CADENCE, MAX_CADENCE)
        df_clean = df_clean[cadence_valid].copy()

    # 6. FEATURE ENGINEERING
    # Pace (min/km)
    df_clean['pace_min_km'] = 60 / df_clean['average_speed']

    # Efficiency Index (Speed / HR) - Key Metric
    df_clean['efficiency_index'] = df_clean['average_speed'] / df_clean['average_hr']

    # Stride Length (meters)
    if 'cadence_spm' in df_clean.columns:
        speed_m_min = df_clean['average_speed'] * 1000 / 60
        df_clean['stride_length_m'] = speed_m_min / df_clean['cadence_spm']

        # Filter stride physics (only remove if stride is calculated AND invalid)
        stride_invalid = (df_clean['stride_length_m'].notna()) & \
                         (~df_clean['stride_length_m'].between(0.4, 2.6))

        df_clean = df_clean[~stride_invalid].copy()
    else:
        df_clean['stride_length_m'] = np.nan

    # 7. SAVE TO FILE
    final_cols_candidates = [
        'date', 'duration_min', 'total_distance', 'average_speed',
        'pace_min_km', 'average_hr',
        'cadence_spm', 'stride_length_m', 'efficiency_index', 'elevation_gain'
    ]

    # Save only existing columns
    cols_to_save = [c for c in final_cols_candidates if c in df_clean.columns]

    df_clean.to_csv('ready_to_train.csv', columns=cols_to_save, index=False)

    print("\n" + "=" * 40)
    print(f"SUCCESS! Saved 'ready_to_train.csv'")
    print(f"Clean training samples: {len(df_clean)}")
    print("=" * 40)
    print(df_clean[cols_to_save].head(3).round(2))


if __name__ == "__main__":
    clean_activities()