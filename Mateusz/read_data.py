import pandas as pd
import numpy as np 


df_activities = pd.read_csv("activities.csv")
df_athletes = pd.read_csv("athletes.csv")

# Merge the dataframes
df = pd.merge(df_activities, df_athletes, on='id')

# Clean column names (strip whitespace)
df.columns = df.columns.str.strip()

# Select only numeric columns
df_numeric = df.select_dtypes(include=[np.number])

# Drop rows with any NaN values in the numeric columns
# df_cleaned = df_numeric.dropna()

print("Shape before dropping NaNs:", df_numeric.shape)
print("Shape after dropping NaNs:", df_numeric.shape)
print("\nFirst 5 rows of cleaned numeric data:")
print(df_numeric.head())

# Save the cleaned numeric data to a CSV for the user
df_numeric.to_csv("cleaned_numeric_data.csv", index=False)