import argparse
import os
import tarfile
import urllib.request

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from lsm_inference import fit_lsm


DEFAULT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_REPO_ROOT = os.path.abspath(os.path.join(DEFAULT_ROOT, "..", ".."))
DEFAULT_DATASET_ROOT = os.path.join(DEFAULT_REPO_ROOT, "datasets", "cora")
DEFAULT_RAW_DIR = os.path.join(DEFAULT_DATASET_ROOT, "source")
DEFAULT_PROCESSED_PATH = os.path.join(DEFAULT_DATASET_ROOT, "generator", "cora.npz")


def download_cora(dest_dir):
    url = "https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz"
    os.makedirs(dest_dir, exist_ok=True)
    tgz_path = os.path.join(dest_dir, "cora.tgz")

    if not os.path.exists(tgz_path):
        print(f"Downloading Cora dataset to {tgz_path} ...")
        urllib.request.urlretrieve(url, tgz_path)
    else:
        print(f"Found existing archive at {tgz_path}, skipping download.")

    extract_dir = os.path.join(dest_dir, "cora")
    if not os.path.exists(extract_dir):
        with tarfile.open(tgz_path, "r:gz") as tar:
            tar.extractall(path=dest_dir)
    return extract_dir


def load_cora_raw(path):
    content_path = os.path.join(path, "cora.content")
    cites_path = os.path.join(path, "cora.cites")

    paper_ids = []
    attrs = []
    labels = []

    with open(content_path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            paper_ids.append(parts[0])
            attrs.append([int(x) for x in parts[1:-1]])
            labels.append(parts[-1])

    paper_ids = np.array(paper_ids)
    X = np.array(attrs, dtype=np.float32)

    unique_labels = sorted(set(labels))
    label_to_idx = {label: i for i, label in enumerate(unique_labels)}
    y = np.array([label_to_idx[label] for label in labels], dtype=np.int64)

    id2idx = {pid: i for i, pid in enumerate(paper_ids)}
    edges = []

    with open(cites_path, "r", encoding="utf-8") as handle:
        for line in handle:
            src, dst = line.strip().split()
            if src in id2idx and dst in id2idx:
                i, j = id2idx[src], id2idx[dst]
                if i != j:
                    edges.append((min(i, j), max(i, j)))

    edges = np.unique(np.array(edges, dtype=np.int64), axis=0)
    edges = np.vstack([edges.T, edges[:, ::-1].T])

    n_nodes = X.shape[0]
    A = sp.coo_matrix(
        (np.ones(edges.shape[1], dtype=np.float64), (edges[0], edges[1])),
        shape=(n_nodes, n_nodes),
        dtype=np.float64,
    )
    A.setdiag(0)
    A.eliminate_zeros()
    return A.tocsr(), X, y


def largest_connected_component(A, X, y):
    n_components, labels = connected_components(A, directed=False)
    print(f"Connected components: {n_components}")

    comp_ids, counts = np.unique(labels, return_counts=True)
    lcc_comp = comp_ids[np.argmax(counts)]
    mask = labels == lcc_comp

    A_lcc = A[mask][:, mask].tocsr()
    X_lcc = X[mask]
    y_lcc = y[mask]
    print(f"LCC size: {A_lcc.shape[0]} nodes")
    return A_lcc, X_lcc, y_lcc


def save_processed_data(A, X, y, inference_res, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    A_csr = A.tocsr()

    Z12 = inference_res["model_Z"]
    alpha = inference_res["model_alpha"]
    Z3 = np.array([])
    L = np.zeros((X.shape[1], 0), dtype=np.float32)
    mu = np.mean(X, axis=0).astype(np.float32)

    np.savez(
        out_path,
        adj_data=A_csr.data,
        adj_indices=A_csr.indices,
        adj_indptr=A_csr.indptr,
        adj_shape=A_csr.shape,
        X=X,
        y=y,
        Z12=Z12,
        Z3=Z3,
        L=L,
        mu=mu,
        alpha=alpha,
        num_z1=Z12.shape[1],
        num_z3=0,
        model_sparsity=inference_res["model_sparsity"],
        converged=inference_res["converged"],
        lsm_r=inference_res["r"],
        lsm_seed=inference_res["seed"],
        lsm_eta_0=inference_res["eta_0"],
        lsm_tau=inference_res["tau"],
    )
    print(f"Saved processed data to {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare the Cora attributed-network dataset.")
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR, help="Directory for downloaded raw Cora files.")
    parser.add_argument(
        "--output",
        default=DEFAULT_PROCESSED_PATH,
        help="Path to the processed .npz output.",
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
    data_dir = download_cora(args.raw_dir)
    A, X, y = load_cora_raw(data_dir)
    print(f"Raw Cora: n={A.shape[0]}, p={X.shape[1]}, nnz(A)={A.nnz}")

    A_lcc, X_lcc, y_lcc = largest_connected_component(A, X, y)
    print(f"LCC Cora: n={A_lcc.shape[0]}, p={X_lcc.shape[1]}, nnz(A)={A_lcc.nnz}")

    print("Running LSM inference ...")
    res = fit_lsm(
        A_lcc.toarray(),
        r=args.r,
        seed=args.seed,
        eta_0=args.eta_0,
        tau=args.tau,
        use_gpu=args.use_gpu,
        covariate_dim=args.covariate_dim,
        n_iter=args.n_iter,
    )
    save_processed_data(A_lcc, X_lcc, y_lcc, res, out_path=args.output)


if __name__ == "__main__":
    main()
