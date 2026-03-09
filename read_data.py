import pandas as pd
import numpy as np 


df = pd.read_csv("activities.csv")

# Clean column names (strip whitespace)
df.columns = df.columns.str.strip()


# deleting examples with null values in crucial columns average_hr, final_cadence, average_speed, total_distance
df.dropna(subset=['average_hr', 'final_cadence', 'average_speed', 'total_distance'], inplace=True)


# deleting columns that are not relevant for the analysis
cols_to_drop = ['athlete_id', 'sport', 'workout_time', 'average_cad', 'average_run_cad', 'trimp_points']

df.drop(columns=cols_to_drop, inplace=True)


# change speed into pace
df['pace_min_km'] = 60 / df['average_speed']

df['date'] = pd.to_datetime(df['date'])
df['age'] = df['date'].dt.year - df['yob']



# # Select only numeric columns
# #df_numeric = df.select_dtypes(include=[np.number])

# # Drop rows with any NaN values in the numeric columns
# # df_cleaned = df_numeric.dropna()

# print("Shape before dropping NaNs:", df_numeric.shape)
# print("Shape after dropping NaNs:", df_numeric.shape)
# print("\nFirst 5 rows of cleaned numeric data:")
# print(df_numeric.head())

# # Save the cleaned numeric data to a CSV for the user
# df_numeric.to_csv("cleaned_numeric_data.csv", index=False)