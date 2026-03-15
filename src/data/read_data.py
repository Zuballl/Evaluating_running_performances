from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = ROOT_DIR / "data" / "activities.csv"
PROCESSED_DATA_PATH = ROOT_DIR / "data" / "processed" / "clean_data.csv"


CRUCIAL_COLUMNS = ["average_hr", "final_cadence", "average_speed", "total_distance"]
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
    "average_speed",
]

def prepare_clean_data(
    raw_data_path: Path = RAW_DATA_PATH,
    processed_data_path: Path = PROCESSED_DATA_PATH,
) -> pd.DataFrame:
    df = pd.read_csv(raw_data_path)
    df.columns = df.columns.str.strip()


    df.dropna(subset=CRUCIAL_COLUMNS, inplace=True)

    df["pace_min_km"] = 60 / df["average_speed"]


    df["date"] = pd.to_datetime(df["date"])
    df["age"] = df["date"].dt.year - df["yob"]

    df["is_male"] = np.where(df["gender"] == "M", 1, 0)

    df.drop(columns=COLUMNS_TO_DROP, inplace=True)

    df["elevation_gain"] = df["elevation_gain"].fillna(0)
    median_aerobic_decoupling = df["aerobic_decoupling"].median()
    df["aerobic_decoupling"] = df["aerobic_decoupling"].fillna(median_aerobic_decoupling)

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
