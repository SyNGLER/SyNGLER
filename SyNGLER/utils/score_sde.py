from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


class ScoreNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dims=None, time_embed_dim: int = 128):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]

        self.input_dim = input_dim
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        layers = []
        prev_dim = input_dim + time_embed_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.SiLU(), nn.Dropout(0.1)])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, input_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        t_embed = self.time_embed(t)
        h = torch.cat([x, t_embed], dim=-1)
        return self.network(h)


class VPSDE:
    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0):
        self.beta_min = beta_min
        self.beta_max = beta_max

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def mean_coeff(self, t: torch.Tensor) -> torch.Tensor:
        log_mean_coeff = -0.25 * t**2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        return torch.exp(log_mean_coeff)

    def std(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(1 - self.mean_coeff(t) ** 2)

    def marginal_prob(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.mean_coeff(t).view(-1, 1) * x
        std = self.std(t).view(-1, 1)
        return mean, std


class ScoreSDE:
    def __init__(
        self,
        input_dim: int,
        hidden_dims=None,
        beta_min: float = 0.1,
        beta_max: float = 20.0,
        device: str = None,
    ):
        if hidden_dims is None:
            hidden_dims = [256, 256]

        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        self.device = device
        self.input_dim = input_dim
        self.score_net = ScoreNet(input_dim, hidden_dims).to(device)
        self.sde = VPSDE(beta_min, beta_max)
        self.data_mean = None
        self.data_std = None
        self.is_fitted = False

    def normalize(self, data: torch.Tensor) -> torch.Tensor:
        if not self.is_fitted:
            self.data_mean = data.mean(dim=0)
            self.data_std = data.std(dim=0)
            self.is_fitted = True
        return (data - self.data_mean) / (self.data_std + 1e-8)

    def denormalize(self, data: torch.Tensor) -> torch.Tensor:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before denormalization. Call train() first.")
        return data * self.data_std + self.data_mean

    def train_step(self, data: torch.Tensor, optimizer: torch.optim.Optimizer) -> float:
        self.score_net.train()
        optimizer.zero_grad()

        batch_size = data.shape[0]
        t = torch.rand(batch_size, device=self.device) * (1.0 - 1e-4) + 1e-4
        mean, std = self.sde.marginal_prob(data, t)
        z = torch.randn_like(data)
        x_t = mean + std * z

        score = self.score_net(x_t, t)
        loss = torch.mean(torch.sum((score * std + z) ** 2, dim=-1))
        loss.backward()
        optimizer.step()
        return loss.item()

    def train(
        self,
        data: torch.Tensor,
        n_epochs: int = 1000,
        batch_size: int = 256,
        lr: float = 1e-3,
        verbose: bool = True,
    ):
        data = data.to(self.device)
        data = self.normalize(data)
        optimizer = torch.optim.Adam(self.score_net.parameters(), lr=lr)

        n_samples = data.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size
        loss_history = []

        for epoch in range(n_epochs):
            perm = torch.randperm(n_samples)
            data_shuffled = data[perm]
            epoch_loss = 0.0

            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_samples)
                batch_data = data_shuffled[start_idx:end_idx]
                epoch_loss += self.train_step(batch_data, optimizer)

            avg_loss = epoch_loss / n_batches
            loss_history.append(avg_loss)
            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {avg_loss:.6f}")

        return loss_history

    @torch.no_grad()
    def sample(self, n_samples: int, n_steps: int = 1000, denoise: bool = True) -> torch.Tensor:
        self.score_net.eval()
        x = torch.randn(n_samples, self.input_dim, device=self.device)

        time_steps = torch.linspace(1.0, 1e-4, n_steps, device=self.device)
        dt = -1.0 / n_steps
        sqrt_abs_dt = torch.sqrt(torch.tensor(abs(dt), device=self.device))

        for i, t in enumerate(time_steps):
            batch_t = torch.ones(n_samples, device=self.device) * t
            score = self.score_net(x, batch_t)
            beta_t = self.sde.beta(batch_t).view(-1, 1)
            drift = -0.5 * beta_t * x - beta_t * score
            diffusion = torch.sqrt(beta_t)

            if i < n_steps - 1 or not denoise:
                x = x + drift * dt + diffusion * sqrt_abs_dt * torch.randn_like(x)
            else:
                x = x + drift * dt

        return self.denormalize(x)

    def save(self, path: str):
        torch.save(
            {
                "score_net": self.score_net.state_dict(),
                "input_dim": self.input_dim,
                "data_mean": self.data_mean,
                "data_std": self.data_std,
                "is_fitted": self.is_fitted,
            },
            path,
        )

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.score_net.load_state_dict(checkpoint["score_net"])
        self.data_mean = checkpoint.get("data_mean", None)
        self.data_std = checkpoint.get("data_std", None)
        self.is_fitted = checkpoint.get("is_fitted", False)


def train_score_sde(
    data: np.ndarray,
    hidden_dims=None,
    n_epochs: int = 1000,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = None,
    verbose: bool = True,
):
    if hidden_dims is None:
        hidden_dims = [256, 256]
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data).float()

    model = ScoreSDE(
        input_dim=data.shape[1],
        hidden_dims=hidden_dims,
        device=device,
    )
    model.train(data=data, n_epochs=n_epochs, batch_size=batch_size, lr=lr, verbose=verbose)
    return model


def generate_samples(model: ScoreSDE, n_samples: int, n_steps: int = 1000, return_numpy: bool = True):
    samples = model.sample(n_samples, n_steps)
    if return_numpy:
        return samples.cpu().numpy()
    return samples
