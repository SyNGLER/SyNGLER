import argparse
import json
import os
import pickle

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import FactorAnalysis


DEFAULT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_REPO_ROOT = os.path.abspath(os.path.join(DEFAULT_ROOT, "..", ".."))
DEFAULT_DATASET_ROOT = os.path.join(DEFAULT_REPO_ROOT, "datasets", "cora")
DEFAULT_CORA_PATH = os.path.join(DEFAULT_DATASET_ROOT, "generator", "cora.npz")
DEFAULT_LSM_ROOT = os.path.join(DEFAULT_DATASET_ROOT, "lsm")
DEFAULT_OUTPUT_DIR = os.path.join(DEFAULT_DATASET_ROOT, "generator")


def to_repo_relative(path):
    return os.path.relpath(path, DEFAULT_REPO_ROOT)


def load_cora_npz(path):
    data = np.load(path, allow_pickle=True)
    A = sp.csr_matrix(
        (data["adj_data"], data["adj_indices"], data["adj_indptr"]),
        shape=tuple(data["adj_shape"]),
    )
    Y = data["X"].astype(np.float32)
    y = data["y"] if "y" in data.files else None
    return A, Y, y


def load_lsm_result(lsm_root, r):
    run_dir = os.path.join(lsm_root, f"r={r}")
    pkl_path = os.path.join(run_dir, "cora.pkl")
    meta_path = os.path.join(run_dir, "meta.json")

    with open(pkl_path, "rb") as handle:
        result = pickle.load(handle)

    meta = None
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as handle:
            meta = json.load(handle)

    return result, meta, pkl_path, meta_path


def compute_top_eigenvalues(X, k=None):
    n_features, n_samples = X.shape
    if k is None:
        k = min(n_features, n_samples)

    X_centered = X - X.mean(axis=1, keepdims=True)
    if n_features > n_samples:
        cov_small = (X_centered.T @ X_centered) / n_samples
        eigenvals = np.linalg.eigvalsh(cov_small)[::-1]
        return eigenvals[:k]

    eigenvals = np.linalg.eigvalsh(np.cov(X_centered))[::-1]
    return eigenvals[:k]


def test_Z2(eigenvals, k0=0, diff=1, level="ninety_five"):
    quantile_table = {
        "ninety_five": [6.89, 12.41, 18.16, 23.99, 29.41, 35.05, 39.89, 47.35],
        "ninety_nine": [16.56, 28.75, 41.57, 55.07, 67.53, 79.13, 91.90, 106.01],
    }
    max_idx = diff + 1 + k0
    if max_idx >= len(eigenvals):
        return {"output": False, "level": level, "ratio": np.inf}

    ratio = (eigenvals[k0] - eigenvals[diff + k0]) / (
        eigenvals[diff + k0] - eigenvals[diff + 1 + k0]
    )
    return {"output": ratio < quantile_table[level][diff - 1], "level": level, "ratio": ratio}


def factor_analysis_on_residuals(R, n_components):
    fa = FactorAnalysis(n_components=n_components, random_state=0)
    Z = fa.fit_transform(R)
    return {"Z": Z, "Gamma": fa.components_.T, "Sigma": fa.noise_variance_}


def infer_attributes(Z1, Y, max_Z2=10):
    mu = np.mean(Y, axis=0)
    Y_c = Y - mu

    Z1_T_Z1_inv = np.linalg.inv(Z1.T @ Z1)
    Gamma_1_T = Z1_T_Z1_inv @ Z1.T @ Y_c
    Gamma_1 = Gamma_1_T.T

    P_Z1 = Z1 @ Z1_T_Z1_inv @ Z1.T
    R = Y_c - P_Z1 @ Y_c
    Phi = (R.T @ R) / (Y.shape[0] - 1)

    n_nodes, p_attr = R.shape
    eigenvals_CovR = compute_top_eigenvalues(R.T, k=min(max_Z2 + 3, min(n_nodes, p_attr)))
    max_Z2_actual = min(max_Z2, len(eigenvals_CovR) - 2)
    max_Z2_actual = max(0, max_Z2_actual)

    no_Z2_series = [test_Z2(eigenvals_CovR, k0=o, diff=1)["output"] for o in range(max_Z2_actual + 1)]

    num_Z2 = 0
    if len(no_Z2_series) > 1:
        for i in range(len(no_Z2_series) - 1):
            if not no_Z2_series[i] and no_Z2_series[i + 1]:
                num_Z2 = i + 1
                break

    if num_Z2 == 0:
        Z2_hat = np.array([])
        Gamma_2 = np.array([])
    else:
        fac_res = factor_analysis_on_residuals(R, num_Z2)
        Z2_hat = fac_res["Z"]
        Gamma_2 = fac_res["Gamma"]

        eigen_vals, eigen_vecs = np.linalg.eigh(Z2_hat.T @ Z2_hat)
        idx = eigen_vals.argsort()[::-1]
        eigen_vecs = eigen_vecs[:, idx]
        Z2_hat = Z2_hat @ eigen_vecs

    Y_hat = np.zeros_like(Y, dtype=np.float32)
    Y_hat += mu
    Y_hat += Z1 @ Gamma_1.T
    if Z2_hat.size > 0 and Gamma_2.size > 0:
        Y_hat += Z2_hat @ Gamma_2.T

    return {
        "Z_1": Z1,
        "Z_2": Z2_hat,
        "mu": mu,
        "Gamma_1": Gamma_1,
        "Gamma_2": Gamma_2,
        "R": R,
        "Phi": Phi,
        "Y_hat": Y_hat,
        "num_Z2": num_Z2,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Build attributed-network latent outputs from saved Cora LSM fits.")
    parser.add_argument("--cora-path", default=DEFAULT_CORA_PATH, help="Processed Cora .npz path.")
    parser.add_argument("--lsm-root", default=DEFAULT_LSM_ROOT, help="Directory containing lsm/r=*/cora.pkl results.")
    parser.add_argument("--r", type=int, default=5, help="Latent dimension r used by the saved LSM fit.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for generated cora_{r}.npz outputs.")
    parser.add_argument("--max-z2", type=int, default=10, help="Maximum number of attribute-specific factors to test.")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    A, Y, y = load_cora_npz(args.cora_path)
    lsm_result, meta, pkl_path, meta_path = load_lsm_result(args.lsm_root, args.r)

    Z1 = np.asarray(lsm_result["model_Z"], dtype=np.float32)
    alpha = np.asarray(lsm_result["model_alpha"], dtype=np.float32)
    sparsity = float(lsm_result["model_sparsity"])

    attr_res = infer_attributes(Z1, Y, max_Z2=args.max_z2)

    output_path = os.path.join(args.output_dir, f"cora_{args.r}.npz")
    np.savez(
        output_path,
        Z_1=attr_res["Z_1"],
        Z_2=attr_res["Z_2"],
        mu=attr_res["mu"],
        alpha=alpha,
        sparsity=sparsity,
        Gamma_1=attr_res["Gamma_1"],
        Gamma_2=attr_res["Gamma_2"],
        A=A.toarray().astype(np.float32),
        Y=Y.astype(np.float32),
        y=np.array([]) if y is None else y,
        R=attr_res["R"].astype(np.float32),
        Phi=attr_res["Phi"].astype(np.float32),
        Y_hat=attr_res["Y_hat"].astype(np.float32),
        num_Z2=attr_res["num_Z2"],
        lsm_pkl_path=to_repo_relative(pkl_path),
        lsm_meta_path=to_repo_relative(meta_path),
        lsm_meta_json=np.array("" if meta is None else json.dumps(meta)),
    )

    print(f"Saved attributed latent output to {output_path}")
    print(f"Z_1 shape: {attr_res['Z_1'].shape}")
    print(f"Z_2 shape: {attr_res['Z_2'].shape if attr_res['Z_2'].size > 0 else (0,)}")
    print(f"Gamma_1 shape: {attr_res['Gamma_1'].shape}")
    print(f"Gamma_2 shape: {attr_res['Gamma_2'].shape if attr_res['Gamma_2'].size > 0 else (0,)}")


if __name__ == "__main__":
    main()
