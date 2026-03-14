from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = ROOT_DIR / "data" / "activities.csv"
PROCESSED_DATA_PATH = ROOT_DIR / "data" / "processed" / "clean_data.csv"


CRUCIAL_COLUMNS = ["average_hr", "average_cad", "average_speed", "total_distance"]
COLUMNS_TO_DROP = [
    "date",
    "yob",
    "gender",
    "athlete_id",
    "sport",
    "workout_time",
    "average_cad",
    "average_run_cad",
    "trimp_points",
]


# 1. Define exactly what you want to keep
COLUMNS_TO_KEEP = [
    "average_hr", 
    "average_speed", 
    "total_distance", 
    "pace_min_km", 
    "age", 
    "is_male", 
    "elevation_gain"
]

COLUMNS_TO_KEEP = [
    "average_hr", 
    "average_speed", 
    "total_distance", 
    "pace_min_km", 
    "is_male", 
    "elevation_gain"
]

def prepare_clean_data(
    raw_data_path: Path = RAW_DATA_PATH,
    processed_data_path: Path = PROCESSED_DATA_PATH,
) -> pd.DataFrame:
    df = pd.read_csv(raw_data_path)
    df.columns = df.columns.str.strip()

    # 1. Basic Cleaning & Calculations
    # Replace 0 with NaN to avoid infinity in pace calculation
    df["pace_min_km"] = 60 / df["average_speed"].replace(0, np.nan)
    df["is_male"] = np.where(df["gender"] == "M", 1, 0)
    df["elevation_gain"] = df["elevation_gain"].fillna(0)

    # 2. SELECT ONLY THE TARGET COLUMNS
    # This automatically drops all the extra columns you didn't want
    existing_keeps = [c for c in COLUMNS_TO_KEEP if c in df.columns]
    df = df[existing_keeps].copy()

    # 3. DROP ROWS WITH MISSING VALUES
    # This ensures the Autoencoder won't crash on NaNs
    initial_len = len(df)
    df = df.dropna()
    
    print(f"Dropped {initial_len - len(df)} rows containing NaNs.")
    print(f"Final dataset shape: {df.shape}")

    # 4. Save
    processed_data_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_data_path, index=False)

    return df

def load_clean_numeric_data(
    clean_data_path: Path = PROCESSED_DATA_PATH,
    sample_size: int | None = 10000,
) -> pd.DataFrame:
    df = pd.read_csv(clean_data_path)
    df_numeric = df.select_dtypes(include=["number"]).copy()
    if sample_size is not None:
        df_numeric = df_numeric.head(sample_size)
    return df_numeric


if __name__ == "__main__":
    cleaned_df = prepare_clean_data()
    print(f"Saved cleaned dataset to {PROCESSED_DATA_PATH}")
    print(cleaned_df.info())
