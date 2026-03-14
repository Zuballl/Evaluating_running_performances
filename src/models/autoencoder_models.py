from dataclasses import dataclass

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from src.data.read_data import load_clean_numeric_data,prepare_clean_data


@dataclass
class AutoencoderResult:
    name: str
    architecture: str
    mse: float
    scores: np.ndarray


def run_autoencoder(data: np.ndarray, hidden_layers: list[int], name: str) -> AutoencoderResult:
    input_dim = data.shape[1]
    input_layer = Input(shape=(input_dim,))

    encoded = input_layer
    for units in hidden_layers:
        encoded = Dense(units, activation="relu")(encoded)

    bottleneck = Dense(1, activation="linear", name=f"bottleneck_{name}")(encoded)

    decoded = bottleneck
    for units in reversed(hidden_layers):
        decoded = Dense(units, activation="relu")(decoded)

    output_layer = Dense(input_dim, activation="sigmoid")(decoded)

    model = Model(input_layer, output_layer)
    model.compile(optimizer="adam", loss="mse")
    model.fit(data, data, epochs=50, batch_size=32, verbose=0)

    reconstructed = model.predict(data, verbose=0)
    mse = float(mean_squared_error(data, reconstructed))

    encoder = Model(input_layer, bottleneck)
    scores = encoder.predict(data, verbose=0).flatten()

    architecture = "Input -> 1 -> Output"
    if hidden_layers:
        middle = " -> ".join(str(units) for units in hidden_layers)
        architecture = f"Input -> {middle} -> 1 -> {middle} -> Output"

    return AutoencoderResult(name=name, architecture=architecture, mse=mse, scores=scores)


def run_autoencoder_comparison(df_numeric) -> tuple[list[AutoencoderResult], np.ndarray]:
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_numeric)

    configs = [
        ("simple_autoencoder", []),
        ("medium_autoencoder", [4]),
        ("deep_autoencoder", [5,3]),
    ]

    results = [run_autoencoder(scaled_data, layers, name) for name, layers in configs]
    return results, scaled_data


if __name__ == "__main__":
    # prepare_clean_data()
    cleaned_df = load_clean_numeric_data()
    print('running autoencoder comparison...')
    results, scaled_data = run_autoencoder_comparison(cleaned_df)
    for res in results:
        print(f"{res.name}: MSE={res.mse:.6f}, Architecture={res.architecture}")