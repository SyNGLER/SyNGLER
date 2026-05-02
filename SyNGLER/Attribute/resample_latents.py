import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from SyNGLER.utils.score_sde import ScoreSDE, generate_samples


DEFAULT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_REPO_ROOT = os.path.abspath(os.path.join(DEFAULT_ROOT, "..", ".."))
DEFAULT_DATASET_ROOT = os.path.join(DEFAULT_REPO_ROOT, "datasets", "cora")
DEFAULT_DATA_PATH = os.path.join(DEFAULT_DATASET_ROOT, "generator", "cora.npz")
DEFAULT_SAMPLE_DIR = os.path.join(DEFAULT_DATASET_ROOT, "run", "resamples")


def parse_args():
    parser = argparse.ArgumentParser(description="Resample inferred Cora latent factors.")
    parser.add_argument("--input", default=DEFAULT_DATA_PATH, help="Processed Cora .npz path.")
    parser.add_argument("--output-dir", default=DEFAULT_SAMPLE_DIR, help="Directory for generated samples.")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of resampled graph copies to generate.")
    parser.add_argument("--epochs", type=int, default=5000, help="Training epochs for the score model.")
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Training learning rate.")
    parser.add_argument("--steps", type=int, default=1000, help="Sampling steps per generated draw.")
    return parser.parse_args()


def main():
    args = parse_args()
    data = np.load(args.input, allow_pickle=True)

    Z12 = data["Z12"]
    Z3 = data["Z3"]
    alpha = data["alpha"]

    alpha_reshaped = alpha.reshape(-1, 1)
    if Z3.size > 0:
        X = np.concatenate([Z12, Z3, alpha_reshaped], axis=1)
    else:
        X = np.concatenate([Z12, alpha_reshaped], axis=1)

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    model = ScoreSDE(input_dim=X.shape[1], hidden_dims=[256, 256, 256], device=device)
    X_tensor = torch.from_numpy(X).float()
    loss_history = model.train(
        data=X_tensor,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        verbose=True,
    )
    print(f"Final loss: {loss_history[-1]:.6f}")

    os.makedirs(args.output_dir, exist_ok=True)
    n_nodes = X.shape[0]
    d12 = Z12.shape[1]
    d3 = Z3.shape[1] if Z3.size > 0 else 0

    for i in range(args.num_samples):
        sample = generate_samples(model=model, n_samples=n_nodes, n_steps=args.steps, return_numpy=True)
        output_path = os.path.join(args.output_dir, f"rep{i}.npz")

        Z12_resampled = sample[:, :d12]
        if d3 > 0:
            Z3_resampled = sample[:, d12 : d12 + d3]
            alpha_resampled = sample[:, d12 + d3 :]
        else:
            Z3_resampled = np.array([])
            alpha_resampled = sample[:, d12:]

        np.savez(
            output_path,
            Z12=Z12_resampled,
            Z3=Z3_resampled,
            alpha=alpha_resampled.flatten(),
        )
        print(f"Saved {output_path}")

    model_path = os.path.join(args.output_dir, "score_sde_model.pt")
    model.save(model_path)
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
