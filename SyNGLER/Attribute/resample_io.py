import os

import numpy as np


DEFAULT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_REPO_ROOT = os.path.abspath(os.path.join(DEFAULT_ROOT, "..", ".."))
DEFAULT_DATASET_ROOT = os.path.join(DEFAULT_REPO_ROOT, "datasets", "cora")


def default_attr_input_path(r: int) -> str:
    return os.path.join(DEFAULT_DATASET_ROOT, "generator", f"cora_{r}.npz")


def load_attribute_latents(path: str):
    data = np.load(path, allow_pickle=True)
    z1 = np.asarray(data["Z_1"], dtype=np.float32)
    z2 = np.asarray(data["Z_2"], dtype=np.float32)
    alpha = np.asarray(data["alpha"], dtype=np.float32).reshape(-1)
    return data, z1, z2, alpha


def stack_latents(z1: np.ndarray, z2: np.ndarray, alpha: np.ndarray):
    alpha_block = alpha.reshape(-1, 1)
    if z2.size > 0:
        matrix = np.concatenate([z1, z2, alpha_block], axis=1)
    else:
        matrix = np.concatenate([z1, alpha_block], axis=1)
    return matrix


def split_latents(matrix: np.ndarray, z1_dim: int, z2_dim: int):
    z1 = matrix[:, :z1_dim]
    if z2_dim > 0:
        z2 = matrix[:, z1_dim : z1_dim + z2_dim]
        alpha = matrix[:, z1_dim + z2_dim :].reshape(-1)
    else:
        z2 = np.array([], dtype=np.float32)
        alpha = matrix[:, z1_dim:].reshape(-1)
    return z1, z2, alpha


def save_resampled_latents(path: str, z1: np.ndarray, z2: np.ndarray, alpha: np.ndarray):
    np.savez(
        path,
        Z_1=np.asarray(z1, dtype=np.float32),
        Z_2=np.asarray(z2, dtype=np.float32),
        Z1=np.asarray(z1, dtype=np.float32),
        Z2=np.asarray(z2, dtype=np.float32),
        alpha=np.asarray(alpha, dtype=np.float32).reshape(-1),
    )
