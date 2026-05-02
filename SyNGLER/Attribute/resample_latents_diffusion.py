import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from SyNGLER.utils.diffusion import (
    build_forest_model_from_matrix,
    generate_matrix_replicates,
    split_latent_matrix,
    stack_latent_blocks,
)


DEFAULT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_REPO_ROOT = os.path.abspath(os.path.join(DEFAULT_ROOT, "..", ".."))
DEFAULT_DATASET_ROOT = os.path.join(DEFAULT_REPO_ROOT, "datasets", "cora")
DEFAULT_DATA_PATH = os.path.join(DEFAULT_DATASET_ROOT, "generator", "cora.npz")
DEFAULT_SAMPLE_DIR = os.path.join(DEFAULT_DATASET_ROOT, "run", "resamples_diffusion")


def parse_args():
    parser = argparse.ArgumentParser(description="Resample inferred Cora latent factors with forest diffusion.")
    parser.add_argument("--input", default=DEFAULT_DATA_PATH, help="Processed Cora .npz path.")
    parser.add_argument("--output-dir", default=DEFAULT_SAMPLE_DIR, help="Directory for generated samples.")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of resampled graph copies to generate.")
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
    data = np.load(args.input, allow_pickle=True)
    Z12 = data["Z12"]
    Z3 = data["Z3"]
    alpha = data["alpha"]

    blocks = [("Z12", Z12)]
    if Z3.size > 0:
        blocks.append(("Z3", Z3))
    blocks.append(("alpha", alpha))

    X, widths = stack_latent_blocks(blocks)
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
        X,
        seed=args.seed,
        n_t=args.n_t,
        duplicate_K=args.duplicate_K,
        xgb_params=xgb_params,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    samples = generate_matrix_replicates(model, X, args.output_dir, reps=args.num_samples)
    for i, sample in enumerate(samples):
        parts = split_latent_matrix(sample, widths)
        output_path = os.path.join(args.output_dir, f"rep{i}.npz")
        np.savez(
            output_path,
            Z12=parts["Z12"],
            Z3=parts.get("Z3", np.array([])),
            alpha=parts["alpha"],
        )
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
