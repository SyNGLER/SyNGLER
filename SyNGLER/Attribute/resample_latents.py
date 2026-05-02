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
from resample_io import (
    DEFAULT_DATASET_ROOT,
    default_attr_input_path,
    load_attribute_latents,
    save_resampled_latents,
    split_latents,
    stack_latents,
)


DEFAULT_SAMPLE_DIR = os.path.join(DEFAULT_DATASET_ROOT, "run", "resamples")


def parse_args():
    parser = argparse.ArgumentParser(description="Resample attributed Cora latents with Score-based SDE.")
    parser.add_argument("--r", type=int, default=5, help="Latent dimension r used to build cora_{r}.npz.")
    parser.add_argument("--input", default=None, help="Attribute inference .npz path. Defaults to datasets/cora/generator/cora_{r}.npz.")
    parser.add_argument("--output-dir", default=DEFAULT_SAMPLE_DIR, help="Directory for generated samples.")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of resampled copies to generate.")
    parser.add_argument("--epochs", type=int, default=5000, help="Training epochs for the score model.")
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Training learning rate.")
    parser.add_argument("--steps", type=int, default=1000, help="Sampling steps per generated draw.")
    return parser.parse_args()


def choose_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main():
    args = parse_args()
    input_path = args.input or default_attr_input_path(args.r)

    data, z1, z2, alpha = load_attribute_latents(input_path)
    matrix = stack_latents(z1, z2, alpha)

    print(f"Loaded {input_path}")
    print(f"Z_1 shape: {z1.shape}")
    print(f"Z_2 shape: {z2.shape}")
    print(f"alpha shape: {alpha.shape}")
    print(f"Training matrix shape: {matrix.shape}")

    device = choose_device()
    print(f"Using device: {device}")

    model = ScoreSDE(input_dim=matrix.shape[1], hidden_dims=[256, 256, 256], device=device)
    loss_history = model.train(
        data=torch.from_numpy(matrix).float(),
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        verbose=True,
    )
    print(f"Final loss: {loss_history[-1]:.6f}")

    os.makedirs(args.output_dir, exist_ok=True)
    z1_dim = z1.shape[1]
    z2_dim = z2.shape[1] if z2.ndim == 2 and z2.size > 0 else 0

    for i in range(args.num_samples):
        sample = generate_samples(
            model=model,
            n_samples=matrix.shape[0],
            n_steps=args.steps,
            return_numpy=True,
        )
        z1_resampled, z2_resampled, alpha_resampled = split_latents(sample, z1_dim, z2_dim)
        output_path = os.path.join(args.output_dir, f"resample_{i + 1}.npz")
        save_resampled_latents(output_path, z1_resampled, z2_resampled, alpha_resampled)
        print(f"Saved {output_path}")

    model_path = os.path.join(args.output_dir, "score_sde_model.pt")
    model.save(model_path)
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
