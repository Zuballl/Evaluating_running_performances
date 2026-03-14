from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.read_data import load_clean_numeric_data

@dataclass
class AutoencoderResult:
    name: str
    architecture: str
    mse: float
    scores: np.ndarray


class ConfigurableAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: list[int]):
        super().__init__()
        encoder_layers: list[nn.Module] = []
        prev_dim = input_dim
        for units in hidden_layers:
            encoder_layers.append(nn.Linear(prev_dim, units))
            encoder_layers.append(nn.ReLU())
            prev_dim = units

        self.encoder_hidden = nn.Sequential(*encoder_layers)
        self.bottleneck = nn.Linear(prev_dim, 1)

        decoder_layers: list[nn.Module] = []
        prev_dim = 1
        for units in reversed(hidden_layers):
            decoder_layers.append(nn.Linear(prev_dim, units))
            decoder_layers.append(nn.ReLU())
            prev_dim = units
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder_hidden(x)
        return self.bottleneck(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decoder(z)


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _to_tensor(data: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.tensor(data, dtype=torch.float32, device=device)


def _predict_reconstruction(model: ConfigurableAutoencoder, data: np.ndarray, device: torch.device, batch_size: int = 4096) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    tensor = _to_tensor(data, device)
    with torch.no_grad():
        for start in range(0, len(tensor), batch_size):
            batch = tensor[start : start + batch_size]
            reconstructed = model(batch)
            outputs.append(reconstructed.cpu().numpy())
    return np.vstack(outputs)


def _predict_scores(model: ConfigurableAutoencoder, data: np.ndarray, device: torch.device, batch_size: int = 4096) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    tensor = _to_tensor(data, device)
    with torch.no_grad():
        for start in range(0, len(tensor), batch_size):
            batch = tensor[start : start + batch_size]
            z = model.encode(batch)
            outputs.append(z.cpu().numpy())
    return np.vstack(outputs).flatten()

def run_autoencoder(
    train_data: np.ndarray,
    test_data: np.ndarray,
    all_data: np.ndarray,
    hidden_layers: list[int],
    name: str,
    epochs: int = 20,
    batch_size: int = 2048,
    patience: int = 3,
    min_delta: float = 1e-5,
) -> AutoencoderResult:
    if epochs <= 0:
        raise ValueError("epochs must be > 0")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if patience <= 0:
        raise ValueError("patience must be > 0")

    device = _get_device()

    input_dim = train_data.shape[1]
    model = ConfigurableAutoencoder(input_dim=input_dim, hidden_layers=hidden_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    train_tensor = _to_tensor(train_data, device)
    train_loader = DataLoader(TensorDataset(train_tensor), batch_size=batch_size, shuffle=True)

    print(f"\n>>> Training {name}...")
    model.train()
    best_loss = float("inf")
    wait = 0
    for epoch in range(epochs):
        epoch_loss = 0.0
        for (batch,) in train_loader:
            optimizer.zero_grad()
            reconstructed = model(batch)
            loss = criterion(reconstructed, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * len(batch)
        avg_loss = epoch_loss / len(train_tensor)
        print(f"Epoch {epoch + 1}/{epochs} - loss: {avg_loss:.6f}")

        if best_loss - avg_loss > min_delta:
            best_loss = avg_loss
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping for {name}: no significant improvement for {patience} epochs.")
                break

    print(f">>> Evaluating {name} on test set...")
    reconstructed_test = _predict_reconstruction(model, test_data, device=device, batch_size=4096)
    mse = float(mean_squared_error(test_data, reconstructed_test))

    # Use bottleneck activation as performance score for all samples.
    scores = _predict_scores(model, all_data, device=device, batch_size=4096)
    
    # Generate architecture string for reporting
    architecture = "Input -> 1 -> Output"
    if hidden_layers:
        middle = " -> ".join(str(units) for units in hidden_layers)
        architecture = f"Input -> {middle} -> 1 -> {middle} -> Output"

    return AutoencoderResult(name=name, architecture=architecture, mse=mse, scores=scores)

def run_autoencoder_comparison(
    df_train,
    df_test,
    df_all,
    ae_epochs: int = 20,
    ae_batch_size: int = 2048,
    ae_patience: int = 3,
) -> tuple[list[AutoencoderResult], np.ndarray]:
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
        res = run_autoencoder(
            scaled_train,
            scaled_test,
            scaled_all,
            layers,
            name,
            epochs=ae_epochs,
            batch_size=ae_batch_size,
            patience=ae_patience,
        )
        results.append(res)
        
    return results, scaled_all

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    # If running directly, load and split data.
    cleaned_df = load_clean_numeric_data()
    print(f'Running autoencoder comparison on {len(cleaned_df)} rows...')

    train_df, test_df = train_test_split(cleaned_df, test_size=0.2, random_state=42)
    results, _ = run_autoencoder_comparison(train_df, test_df, cleaned_df)
    
    print("\n" + "="*30)
    print("FINAL RESULTS")
    print("="*30)
    for res in results:
        print(f"{res.name:20} | MSE: {res.mse:.6f} | Arch: {res.architecture}")