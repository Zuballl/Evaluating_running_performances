from dataclasses import dataclass

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class VAEResult:
    name: str
    architecture: str
    mse: float
    scores: np.ndarray
    kl_loss: float


class VAE(nn.Module):
    def __init__(self, input_dim: int, intermediate_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, intermediate_dim // 2),
            nn.ReLU(),
        )
        self.z_mean = nn.Linear(intermediate_dim // 2, 1)
        self.z_log_var = nn.Linear(intermediate_dim // 2, 1)

        self.decoder = nn.Sequential(
            nn.Linear(1, intermediate_dim // 2),
            nn.ReLU(),
            nn.Linear(intermediate_dim // 2, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, input_dim),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.z_mean(h), self.z_log_var(h)

    def reparameterize(self, z_mean: torch.Tensor, z_log_var: torch.Tensor) -> torch.Tensor:
        epsilon = torch.randn_like(z_mean)
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_mean, z_log_var = self.encode(x)
        z = self.reparameterize(z_mean, z_log_var)
        reconstruction = self.decode(z)
        return reconstruction, z_mean, z_log_var


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def _to_tensor(data: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.tensor(data, dtype=torch.float32, device=device)


def run_vae(
    df_train,
    df_test,
    df_all,
    epochs: int = 50,
    batch_size: int = 32,
    patience: int = 5,
    min_delta: float = 1e-5,
    verbose: bool = False,
) -> VAEResult:
    if epochs <= 0:
        raise ValueError("epochs must be > 0")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if patience <= 0:
        raise ValueError("patience must be > 0")

    scaler = MinMaxScaler()
    scaled_train = scaler.fit_transform(df_train)
    scaled_test = scaler.transform(df_test)
    scaled_all = scaler.transform(df_all)

    device = _get_device()
    input_dim = scaled_train.shape[1]
    vae = VAE(input_dim=input_dim).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

    train_tensor = _to_tensor(scaled_train, device)
    train_loader = DataLoader(TensorDataset(train_tensor), batch_size=batch_size, shuffle=True)

    vae.train()
    best_loss = float("inf")
    wait = 0
    for epoch in range(epochs):
        epoch_loss = 0.0
        for (batch,) in train_loader:
            optimizer.zero_grad()
            reconstruction, z_mean, z_log_var = vae(batch)
            recon_loss = torch.mean((batch - reconstruction) ** 2)
            kl = -0.5 * (1 + z_log_var - z_mean.pow(2) - torch.exp(z_log_var))
            kl_loss = torch.mean(kl)
            total_loss = recon_loss + kl_loss
            total_loss.backward()
            optimizer.step()
            epoch_loss += float(total_loss.item()) * len(batch)

        avg_loss = epoch_loss / len(train_tensor)
        if verbose:
            print(f"VAE epoch {epoch + 1}/{epochs} - loss: {avg_loss:.6f}")

        if best_loss - avg_loss > min_delta:
            best_loss = avg_loss
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print(f"VAE early stopping: no significant improvement for {patience} epochs.")
                break

    # MSE on unseen test data
    vae.eval()
    test_tensor = _to_tensor(scaled_test, device)
    with torch.no_grad():
        z_mean_test, _ = vae.encode(test_tensor)
        reconstructed_test = vae.decode(z_mean_test)

    reconstructed_test_np = reconstructed_test.cpu().numpy()
    mse = float(mean_squared_error(scaled_test, reconstructed_test_np))

    # Scores and KL for all data (for ranking and reporting)
    all_tensor = _to_tensor(scaled_all, device)
    with torch.no_grad():
        z_mean_all, z_log_var_all = vae.encode(all_tensor)

    z_mean_all_np = z_mean_all.cpu().numpy()
    z_log_var_all_np = z_log_var_all.cpu().numpy()
    kl_loss = float(np.mean(-0.5 * (1 + z_log_var_all_np - np.square(z_mean_all_np) - np.exp(z_log_var_all_np))))

    return VAEResult(
        name="vae",
        architecture="Input -> Dense -> z_mean(1) -> Dense -> Output",
        mse=mse,
        scores=z_mean_all_np.flatten(),
        kl_loss=kl_loss,
    )
