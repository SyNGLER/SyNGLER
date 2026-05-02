import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from SyNGLER.utils.diffusion import split_latent_matrix, stack_latent_blocks
from SyNGLER.utils.resampling import bootstrap_feature_matrix


DEFAULT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_REPO_ROOT = os.path.abspath(os.path.join(DEFAULT_ROOT, "..", ".."))
DEFAULT_DATASET_ROOT = os.path.join(DEFAULT_REPO_ROOT, "datasets", "cora")
DEFAULT_DATA_PATH = os.path.join(DEFAULT_DATASET_ROOT, "generator", "cora.npz")
DEFAULT_SAMPLE_DIR = os.path.join(DEFAULT_DATASET_ROOT, "run", "resamples_bootstrap")


def parse_args():
    parser = argparse.ArgumentParser(description="Resample inferred Cora latent factors with bootstrap resampling.")
    parser.add_argument("--input", default=DEFAULT_DATA_PATH, help="Processed Cora .npz path.")
    parser.add_argument("--output-dir", default=DEFAULT_SAMPLE_DIR, help="Directory for generated samples.")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of resampled graph copies to generate.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
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

    os.makedirs(args.output_dir, exist_ok=True)
    for i in range(args.num_samples):
        np.random.seed(args.seed + i)
        sample = bootstrap_feature_matrix(X, batch=1).squeeze(0)
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
