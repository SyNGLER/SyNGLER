import argparse
import os

import numpy as np

from resample_io import (
    DEFAULT_DATASET_ROOT,
    default_attr_input_path,
    load_attribute_latents,
    save_resampled_latents,
    split_latents,
    stack_latents,
)


DEFAULT_SAMPLE_DIR = os.path.join(DEFAULT_DATASET_ROOT, "run", "resamples_bootstrap")


def parse_args():
    parser = argparse.ArgumentParser(description="Resample attributed Cora latents with bootstrap resampling.")
    parser.add_argument("--r", type=int, default=5, help="Latent dimension r used to build cora_{r}.npz.")
    parser.add_argument("--input", default=None, help="Attribute inference .npz path. Defaults to datasets/cora/generator/cora_{r}.npz.")
    parser.add_argument("--output-dir", default=DEFAULT_SAMPLE_DIR, help="Directory for generated samples.")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of resampled copies to generate.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = args.input or default_attr_input_path(args.r)

    _, z1, z2, alpha = load_attribute_latents(input_path)
    matrix = stack_latents(z1, z2, alpha)
    z1_dim = z1.shape[1]
    z2_dim = z2.shape[1] if z2.ndim == 2 and z2.size > 0 else 0

    os.makedirs(args.output_dir, exist_ok=True)
    for i in range(args.num_samples):
        rng = np.random.default_rng(args.seed + i)
        indices = rng.choice(matrix.shape[0], size=matrix.shape[0], replace=True)
        sample = matrix[indices]
        z1_resampled, z2_resampled, alpha_resampled = split_latents(sample, z1_dim, z2_dim)
        output_path = os.path.join(args.output_dir, f"resample_{i + 1}.npz")
        save_resampled_latents(output_path, z1_resampled, z2_resampled, alpha_resampled)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
