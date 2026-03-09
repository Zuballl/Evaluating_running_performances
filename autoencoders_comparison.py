import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# --- 1. LOAD & CLEAN DATA ---
df_activities = pd.read_csv("activities.csv")
df_athletes = pd.read_csv("athletes.csv")
df = pd.merge(df_activities, df_athletes, on='id')
df.columns = df.columns.str.strip()

# Select numeric, sample 10k, and clean Inf/NaN
df_numeric = df.select_dtypes(include=[np.number]).head(10000).copy()
df_numeric = df_numeric.replace([np.inf, -np.inf], np.nan)
df_numeric = df_numeric.fillna(df_numeric.mean()).dropna(axis=1)

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df_numeric)
input_dim = df_numeric.shape[1]

# --- 2. MULTI-MODEL TOURNAMENT FUNCTION ---
def run_autoencoder(data, layers, name):
    input_layer = Input(shape=(input_dim,))
    x = input_layer
    for nodes in layers:
        x = Dense(nodes, activation='relu')(x)
    
    bottleneck = Dense(1, activation='linear', name=f"bottleneck_{name}")(x)
    
    x = bottleneck
    for nodes in reversed(layers):
        x = Dense(nodes, activation='relu')(x)
    
    output_layer = Dense(input_dim, activation='sigmoid')(x)
    
    model = Model(input_layer, output_layer)
    model.compile(optimizer='adam', loss='mse')
    model.fit(data, data, epochs=50, verbose=0, batch_size=32)
    
    # Calculate Reconstruction Error (Lower is better)
    reconstructed = model.predict(data)
    mse = mean_squared_error(data, reconstructed)
    
    # Get the 1D Score
    encoder = Model(input_layer, bottleneck)
    score = encoder.predict(data)
    
    return mse, score

# --- 3. EXECUTION ---
print("Running Simple AE (No hidden layers)...")
mse_simple, scores_simple = run_autoencoder(scaled_data, [], "simple")

print("Running Medium AE (Hidden: 16)...")
mse_med, scores_med = run_autoencoder(scaled_data, [16], "medium")

print("Running Deep AE (Hidden: 32, 16, 8)...")
mse_deep, scores_deep = run_autoencoder(scaled_data, [32, 16, 8], "deep")

# --- 4. VISUALIZATION ---
plt.figure(figsize=(10, 5))
plt.scatter(scores_simple, scores_deep, alpha=0.5, c=scores_deep, cmap='viridis')
plt.xlabel("Simple Architecture Scores")
plt.ylabel("Deep Architecture Scores")
plt.title("Model Agreement: Simple vs Deep Autoencoder")
plt.colorbar(label="Deep Score Intensity")
plt.show()

# --- 5. FINAL COMPARISON DATA ---
results = {
    "Model Approach": ["Simple Autoencoder", "Medium Autoencoder", "Deep Autoencoder", "VAE (Previous Task)", "t-SNE (Previous Task)"],
    "Architecture": ["Input -> 1 -> Output", "Input -> 16 -> 1 -> 16 -> Output", "Input -> 32 -> 16 -> 8 -> 1 -> ...", "Probabilistic Latent Space", "Manifold Learning"],
    "MSE / Metric": [f"{mse_simple:.6f}", f"{mse_med:.6f}", f"{mse_deep:.6f}", "N/A (KL Divergence)", "N/A (KL-Loss)"],
    "Best For": ["Linear relationships", "General patterns", "Complex nonlinearities", "Normalizing ranks", "Visualizing clusters"]
}

comparison_table = pd.DataFrame(results)
print("\n--- FINAL MODEL COMPARISON ---")
print(comparison_table.to_markdown(index=False))