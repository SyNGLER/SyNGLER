import argparse
import os
import tarfile
import urllib.request

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from latent_inference import inference_all_in_one


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

    Z12 = inference_res["Z12hat"]
    Z3 = inference_res["Z3hat"]
    L = inference_res["Lhat"]
    mu = inference_res["mu_hat"]
    alpha = inference_res["alpha_hat"]

    if Z3 is None:
        Z3 = np.array([])

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
        num_z1=inference_res["Num_Z1"],
        num_z3=inference_res["Num_Z3"],
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
    parser.add_argument(
        "--variance-target",
        type=float,
        default=0.8,
        help="Variance target used by latent dimension selection.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = download_cora(args.raw_dir)
    A, X, y = load_cora_raw(data_dir)
    print(f"Raw Cora: n={A.shape[0]}, p={X.shape[1]}, nnz(A)={A.nnz}")

    A_lcc, X_lcc, y_lcc = largest_connected_component(A, X, y)
    print(f"LCC Cora: n={A_lcc.shape[0]}, p={X_lcc.shape[1]}, nnz(A)={A_lcc.nnz}")

    dat = {"A": A_lcc, "Y": X_lcc, "n": A_lcc.shape[0], "p": X_lcc.shape[1]}
    print("Running latent inference ...")
    res = inference_all_in_one(dat, variance_target=args.variance_target, use_sparse=True)
    save_processed_data(A_lcc, X_lcc, y_lcc, res, out_path=args.output)


if __name__ == "__main__":
    main()
