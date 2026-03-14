from dataclasses import dataclass
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from src.data.read_data import load_clean_numeric_data, prepare_clean_data

@dataclass
class AutoencoderResult:
    name: str
    architecture: str
    mse: float
    scores: np.ndarray

def run_autoencoder(data: np.ndarray, hidden_layers: list[int], name: str) -> AutoencoderResult:
    # Clear memory from previous model runs
    K.clear_session()
    
    input_dim = data.shape[1]
    input_layer = Input(shape=(input_dim,))

    # Encoder path
    encoded = input_layer
    for units in hidden_layers:
        encoded = Dense(units, activation="relu")(encoded)

    # Bottleneck layer (The 'Score' generator)
    bottleneck = Dense(1, activation="linear", name=f"bottleneck_{name}")(encoded)

    # Decoder path
    decoded = bottleneck
    for units in reversed(hidden_layers):
        decoded = Dense(units, activation="relu")(decoded)

    output_layer = Dense(input_dim, activation="sigmoid")(decoded)

    # Build and compile
    model = Model(input_layer, output_layer)
    model.compile(optimizer="adam", loss="mse")

    print(f"\n>>> Training {name}...")
    # Increased batch_size and enabled verbose progress bar
    model.fit(
        data, 
        data, 
        epochs=20,          # Reduced slightly for speed, 50 is often overkill for AE
        batch_size=2048,    # ESSENTIAL for 1.7M rows
        verbose=1           # Shows progress bar
    )

    print(f">>> Generating predictions and scores for {name}...")
    reconstructed = model.predict(data, batch_size=4096, verbose=1)
    mse = float(mean_squared_error(data, reconstructed))

    # Extract the encoder part to get the performance scores
    encoder = Model(input_layer, bottleneck)
    scores = encoder.predict(data, batch_size=4096, verbose=1).flatten()

    # Clean up model objects to save RAM
    del model
    
    # Generate architecture string for reporting
    architecture = "Input -> 1 -> Output"
    if hidden_layers:
        middle = " -> ".join(str(units) for units in hidden_layers)
        architecture = f"Input -> {middle} -> 1 -> {middle} -> Output"

    return AutoencoderResult(name=name, architecture=architecture, mse=mse, scores=scores)

def run_autoencoder_comparison(df_numeric) -> tuple[list[AutoencoderResult], np.ndarray]:
    print(f"Scaling data with shape {df_numeric.shape}...")
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_numeric)

    configs = [
        ("simple_autoencoder", []),
        ("medium_autoencoder", [4]),
        ("deep_autoencoder", [5, 3]),
    ]

    results = []
    for name, layers in configs:
        res = run_autoencoder(scaled_data, layers, name)
        results.append(res)
        
    return results, scaled_data

if __name__ == "__main__":
    # If running directly, load the data
    # Note: If using the main pipeline, pass the sample size there!
    cleaned_df = load_clean_numeric_data() 
    print(f'Running autoencoder comparison on {len(cleaned_df)} rows...')
    
    results, _ = run_autoencoder_comparison(cleaned_df)
    
    print("\n" + "="*30)
    print("FINAL RESULTS")
    print("="*30)
    for res in results:
        print(f"{res.name:20} | MSE: {res.mse:.6f} | Arch: {res.architecture}")