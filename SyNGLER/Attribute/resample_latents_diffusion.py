import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from SyNGLER.utils.diffusion import build_forest_model_from_matrix
from resample_io import (
    DEFAULT_DATASET_ROOT,
    default_attr_input_path,
    load_attribute_latents,
    save_resampled_latents,
    split_latents,
    stack_latents,
)


DEFAULT_SAMPLE_DIR = os.path.join(DEFAULT_DATASET_ROOT, "run", "resamples_diffusion")


def parse_args():
    parser = argparse.ArgumentParser(description="Resample attributed Cora latents with ForestDiffusion.")
    parser.add_argument("--r", type=int, default=5, help="Latent dimension r used to build cora_{r}.npz.")
    parser.add_argument("--input", default=None, help="Attribute inference .npz path. Defaults to datasets/cora/generator/cora_{r}.npz.")
    parser.add_argument("--output-dir", default=DEFAULT_SAMPLE_DIR, help="Directory for generated samples.")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of resampled copies to generate.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--n_t", type=int, default=100, help="Diffusion time steps.")
    parser.add_argument("--duplicate_K", type=int, default=100, help="Forest diffusion duplication factor.")
    parser.add_argument("--max_depth", type=int, default=7, help="Tree depth.")
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of estimators.")
    parser.add_argument("--eta", type=float, default=0.3, help="Learning rate.")
    parser.add_argument("--tree_method", type=str, default="hist", help="XGBoost tree method.")
    parser.add_argument("--reg_lambda", type=float, default=0.0, help="L2 regularization.")
    parser.add_argument("--reg_alpha", type=float, default=0.0, help="L1 regularization.")
    parser.add_argument("--subsample", type=float, default=1.0, help="Subsample ratio.")
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = args.input or default_attr_input_path(args.r)

    _, z1, z2, alpha = load_attribute_latents(input_path)
    matrix = stack_latents(z1, z2, alpha)

    xgb_params = dict(
        max_depth=args.max_depth,
        n_estimators=args.n_estimators,
        eta=args.eta,
        tree_method=args.tree_method,
        reg_lambda=args.reg_lambda,
        reg_alpha=args.reg_alpha,
        subsample=args.subsample,
    )
    model = build_forest_model_from_matrix(
        matrix,
        seed=args.seed,
        n_t=args.n_t,
        duplicate_K=args.duplicate_K,
        xgb_params=xgb_params,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    z1_dim = z1.shape[1]
    z2_dim = z2.shape[1] if z2.ndim == 2 and z2.size > 0 else 0

    for i in range(args.num_samples):
        xy_fake = model.generate(batch_size=matrix.shape[0])
        sample = xy_fake[:, :-1]
        z1_resampled, z2_resampled, alpha_resampled = split_latents(sample, z1_dim, z2_dim)
        output_path = os.path.join(args.output_dir, f"resample_{i + 1}.npz")
        save_resampled_latents(output_path, z1_resampled, z2_resampled, alpha_resampled)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
