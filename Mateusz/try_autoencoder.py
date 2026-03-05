import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# 1. Load data
print("Loading data...")
df_activities = pd.read_csv("activities.csv")
df_athletes = pd.read_csv("athletes.csv")

# Merge and clean column names
df = pd.merge(df_activities, df_athletes, on='id')
df.columns = df.columns.str.strip()

# 2. Filter for Numeric and Sample (to speed up processing)
df_numeric = df.select_dtypes(include=[np.number])

# Taking the first 10,000 rows as requested
df_numeric = df_numeric.head(10000).copy() 

# 3. Deep Cleaning (Handle Infinity and NaNs)
# Replace infinity first (the source of your previous error)
df_numeric = df_numeric.replace([np.inf, -np.inf], np.nan)

# Fill NaNs with the mean of each column
df_numeric = df_numeric.fillna(df_numeric.mean())

# Drop columns that are STILL NaN (meaning the whole column was empty or all Infs)
df_numeric = df_numeric.dropna(axis=1)

# Cap extreme values to stay within float64 limits
df_numeric = df_numeric.clip(lower=-1e308, upper=1e308)

print(f"Data ready! Shape: {df_numeric.shape}")

if df_numeric.empty or df_numeric.shape[1] == 0:
    print("Warning: DataFrame is empty or has no numeric columns after cleaning!")
else:
    # 4. Preprocessing
    scaler = MinMaxScaler()
    # Now it is safe to fit_transform because Inf/NaN are gone
    scaled_data = scaler.fit_transform(df_numeric)

    # 5. Define Autoencoder Architecture
    input_dim = df_numeric.shape[1] 
    encoding_dim = 1 

    # Encoder
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(32, activation='relu')(input_layer) # Increased slightly for better learning
    bottleneck = Dense(encoding_dim, activation='linear')(encoded) 

    # Decoder
    decoded = Dense(32, activation='relu')(bottleneck)
    output_layer = Dense(input_dim, activation='sigmoid')(decoded) 

    # Full Model
    autoencoder = Model(input_layer, output_layer)
    autoencoder.compile(optimizer='adam', loss='mse')

    # 6. Train the model
    print(f"Training on {df_numeric.shape[0]} rows and {input_dim} columns...")
    autoencoder.fit(scaled_data, scaled_data, 
                    epochs=50, # 50 is usually enough for a score bottleneck
                    batch_size=32, 
                    shuffle=True, 
                    verbose=1)

    # 7. Extract the Encoder to get the score
    encoder_model = Model(input_layer, bottleneck)
    performance_scores = encoder_model.predict(scaled_data)

    # 8. Add the score back to the dataframe
    df_numeric['performance_score'] = performance_scores
    
    # Sort by score to see your "best" performances (optional)
    df_numeric = df_numeric.sort_values(by='performance_score', ascending=False)

    # Display results
    print("\nTop 5 rows with Calculated Performance Score:")
    print(df_numeric[['performance_score']].head())

    # Optional: Save results
    # df_numeric.to_csv("performance_results.csv")