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


def prepare_clean_data(
    raw_data_path: Path = RAW_DATA_PATH,
    processed_data_path: Path = PROCESSED_DATA_PATH,
) -> pd.DataFrame:
    df = pd.read_csv(raw_data_path)
    df.columns = df.columns.str.strip()

    # 2. Convert to datetime with errors='coerce'
    df["date"] = pd.to_datetime(df["date"], errors='coerce')

    # 3. Remove rows where the date conversion failed (the malformed Thai rows)
    initial_count = len(df)
    df = df.dropna(subset=["date"])
    print(f"Rows dropped due to failed date conversion: {initial_count - len(df)}")

    # 4. Handle Crucial Columns
    df.dropna(subset=CRUCIAL_COLUMNS, inplace=True)

    # 5. Calculations
    # Since 'yob' is missing, we check if 'age' already exists
    if "age" not in df.columns and "yob" in df.columns:
        df["age"] = df["date"].dt.year - df["yob"]
    elif "age" not in df.columns:
        # Fallback if both are missing
        df["age"] = np.nan
        
    df["pace_min_km"] = 60 / df["average_speed"].replace(0, np.nan)
    df["is_male"] = np.where(df["gender"] == "M", 1, 0)

    # 6. Drop columns (only if they exist to avoid more KeyErrors)
    cols_to_remove = [c for c in COLUMNS_TO_DROP if c in df.columns]
    df.drop(columns=cols_to_remove, inplace=True)

    # 7. Fill remaining missing values
    df["elevation_gain"] = df["elevation_gain"].fillna(0)
    if "aerobic_decoupling" in df.columns:
        median_val = df["aerobic_decoupling"].median()
        df["aerobic_decoupling"] = df["aerobic_decoupling"].fillna(median_val)

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
