from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = ROOT_DIR / "data" / "processed" /"clean_data.csv"

df = pd.read_csv(RAW_DATA_PATH)
print(df.count())