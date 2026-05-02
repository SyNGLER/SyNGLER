import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import average_precision_score, roc_auc_score

from lsm_inference import fit_lsm


DEFAULT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_REPO_ROOT = os.path.abspath(os.path.join(DEFAULT_ROOT, "..", ".."))
DEFAULT_DATASET_ROOT = os.path.join(DEFAULT_REPO_ROOT, "datasets", "cora")
DEFAULT_DATA_PATH = os.path.join(DEFAULT_DATASET_ROOT, "generator", "cora.npz")
DEFAULT_OUTPUT_PATH = os.path.join(DEFAULT_DATASET_ROOT, "run", "cora_inference_results.npz")


def load_cora_npz(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Processed file not found: {path}. Run prepare_cora.py first.")

    data = np.load(path, allow_pickle=True)
    A = sp.csr_matrix(
        (data["adj_data"], data["adj_indices"], data["adj_indptr"]),
        shape=tuple(data["adj_shape"]),
    )
    X = data["X"].astype(float)
    y = data["y"] if "y" in data.files else None
    return A, X, y


def plot_and_save_singular_values(A, X, out_path):
    A_dense = A.toarray().astype(float) if sp.issparse(A) else A.astype(float)
    X_dense = X.astype(float)
    sv_A = np.linalg.svd(A_dense, compute_uv=False)
    sv_X = np.linalg.svd(X_dense, compute_uv=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(np.arange(1, len(sv_A) + 1), sv_A, marker="o", markersize=2, linewidth=1)
    axes[0].set_title("Singular values of A")
    axes[0].set_xlabel("Index")
    axes[0].set_ylabel("Singular value")
    axes[0].set_yscale("log")

    axes[1].plot(np.arange(1, len(sv_X) + 1), sv_X, marker="o", markersize=2, linewidth=1)
    axes[1].set_title("Singular values of X")
    axes[1].set_xlabel("Index")
    axes[1].set_ylabel("Singular value")
    axes[1].set_yscale("log")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)


def reconstruct_A_full(Z12, alpha):
    A_hat = alpha[:, None] + alpha[None, :] + Z12 @ Z12.T
    np.fill_diagonal(A_hat, 0.0)
    return A_hat


def evaluate_network_reconstruction_error(A, A_hat):
    A_true = A.toarray().astype(float) if sp.issparse(A) else A.astype(float)
    rel_err_A = np.linalg.norm(A_true - A_hat, "fro") / np.linalg.norm(A_true, "fro")
    return rel_err_A


def split_edges(A, test_ratio=0.1, seed=42):
    rng = np.random.default_rng(seed)
    A_triu = sp.triu(A, k=1)
    row, col = A_triu.nonzero()
    edges = np.stack([row, col], axis=1)

    perm = rng.permutation(edges.shape[0])
    edges = edges[perm]
    num_test = int(edges.shape[0] * test_ratio)

    test_pos = edges[:num_test]
    train_edges = edges[num_test:]

    data_train = np.ones(train_edges.shape[0])
    A_train = sp.csr_matrix((data_train, (train_edges[:, 0], train_edges[:, 1])), shape=A.shape)
    A_train = A_train + A_train.T

    neg_edges = []
    existing_set = set((u, v) for u, v in edges)
    while len(neg_edges) < num_test:
        u = rng.integers(0, A.shape[0])
        v = rng.integers(0, A.shape[0])
        if u < v and (u, v) not in existing_set:
            neg_edges.append([u, v])
            existing_set.add((u, v))

    return A_train, test_pos, np.array(neg_edges)


def run_link_prediction(A, r=5, seed=0, eta_0=0.1, tau=0.0, use_gpu=False, covariate_dim=2, n_iter=500000):
    A_train, test_pos, test_neg = split_edges(A, test_ratio=0.1)
    res = fit_lsm(
        A_train.toarray(),
        r=r,
        seed=seed,
        eta_0=eta_0,
        tau=tau,
        use_gpu=use_gpu,
        covariate_dim=covariate_dim,
        n_iter=n_iter,
    )

    Z12 = res["model_Z"]
    alpha = res["model_alpha"]

    def get_scores(edge_list):
        scores = []
        for u, v in edge_list:
            scores.append(alpha[u] + alpha[v] + np.dot(Z12[u], Z12[v]))
        return np.array(scores)

    pos_scores = get_scores(test_pos)
    neg_scores = get_scores(test_neg)
    y_true = np.hstack([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    y_scores = np.hstack([pos_scores, neg_scores])
    auc = roc_auc_score(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)
    return auc, ap


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the attributed Cora pipeline.")
    parser.add_argument("--input", default=DEFAULT_DATA_PATH, help="Processed Cora .npz path.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Evaluation result .npz path.")
    parser.add_argument(
        "--plot-singular-values",
        action="store_true",
        help="Save a singular-value plot alongside the output file.",
    )
    parser.add_argument(
        "--skip-link-prediction",
        action="store_true",
        help="Skip link prediction and only run reconstruction metrics.",
    )
    parser.add_argument("--r", type=int, default=5, help="Latent dimension for LSM inference.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for LSM inference.")
    parser.add_argument("--eta-0", type=float, default=0.1, help="Base learning-rate scale for LSM PGD.")
    parser.add_argument("--tau", type=float, default=0.0, help="SVD threshold used in LSM initialization.")
    parser.add_argument("--use-gpu", action="store_true", help="Use CuPy GPU acceleration when available.")
    parser.add_argument("--covariate-dim", type=int, default=2, help="Dummy covariate dimension used by LSM backend.")
    parser.add_argument("--n-iter", type=int, default=500000, help="Maximum PGD iterations for LSM inference.")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    A, X, _ = load_cora_npz(args.input)
    n, p = X.shape
    print(f"Cora loaded: n={n}, p={p}, nnz(A)={A.nnz}")

    if args.plot_singular_values:
        plot_path = os.path.join(os.path.dirname(args.output), "cora_singular_values.png")
        plot_and_save_singular_values(A, X, plot_path)
        print(f"Saved singular value plot to {plot_path}")

    data = np.load(args.input, allow_pickle=True)
    Z12_hat = data["Z12"]
    alpha_hat = data["alpha"]
    elapsed = None

    A_hat = reconstruct_A_full(Z12_hat, alpha_hat)
    err_A = evaluate_network_reconstruction_error(A, A_hat)

    auc = None
    ap = None
    if not args.skip_link_prediction:
        start = time.time()
        auc, ap = run_link_prediction(
            A,
            r=args.r,
            seed=args.seed,
            eta_0=args.eta_0,
            tau=args.tau,
            use_gpu=args.use_gpu,
            covariate_dim=args.covariate_dim,
            n_iter=args.n_iter,
        )
        elapsed = time.time() - start
        print(f"Link prediction AUC={auc:.4f}, AP={ap:.4f}")

    np.savez(
        args.output,
        Z12=Z12_hat,
        Z3=np.array([]),
        L=np.zeros((X.shape[1], 0), dtype=np.float32),
        mu=np.mean(X, axis=0).astype(np.float32),
        alpha=alpha_hat,
        metrics={
            "err_A": err_A,
            "err_X": None,
            "auc": auc,
            "ap": ap,
            "elapsed_seconds": elapsed,
        },
    )

    print(f"Network reconstruction error: {err_A:.4f}")
    print("Attribute reconstruction error: unavailable for the LSM-only inference path")
    print(f"Saved evaluation results to {args.output}")


if __name__ == "__main__":
    main()
