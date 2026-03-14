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

def run_autoencoder(train_data: np.ndarray, test_data: np.ndarray, all_data: np.ndarray, hidden_layers: list[int], name: str) -> AutoencoderResult:
    # Clear memory from previous model runs
    K.clear_session()
    
    input_dim = train_data.shape[1]
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
    model.fit(
        train_data,
        train_data,
        epochs=20,
        batch_size=2048,
        verbose=1
    )

    print(f">>> Evaluating {name} on test set...")
    reconstructed_test = model.predict(test_data, batch_size=4096, verbose=1)
    mse = float(mean_squared_error(test_data, reconstructed_test))

    # Extract the encoder part to generate performance scores for ALL data
    encoder = Model(input_layer, bottleneck)
    scores = encoder.predict(all_data, batch_size=4096, verbose=1).flatten()

    # Clean up model objects to save RAM
    del model
    
    # Generate architecture string for reporting
    architecture = "Input -> 1 -> Output"
    if hidden_layers:
        middle = " -> ".join(str(units) for units in hidden_layers)
        architecture = f"Input -> {middle} -> 1 -> {middle} -> Output"

    return AutoencoderResult(name=name, architecture=architecture, mse=mse, scores=scores)

def run_autoencoder_comparison(df_train, df_test, df_all) -> tuple[list[AutoencoderResult], np.ndarray]:
    print(f"Scaling data: train={df_train.shape}, test={df_test.shape}, all={df_all.shape}")
    scaler = MinMaxScaler()
    scaled_train = scaler.fit_transform(df_train)
    scaled_test = scaler.transform(df_test)
    scaled_all = scaler.transform(df_all)

    configs = [
        ("simple_autoencoder", []),
        ("medium_autoencoder", [4]),
        ("deep_autoencoder", [5, 3]),
    ]

    results = []
    for name, layers in configs:
        res = run_autoencoder(scaled_train, scaled_test, scaled_all, layers, name)
        results.append(res)
        
    return results, scaled_all

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