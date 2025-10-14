import numpy as np
import torch
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

def bootstrap_alpha_Z(model_alpha: np.ndarray, model_Z: np.ndarray, batch: int = 1):
    model_alpha = model_alpha.flatten()
    n, r = model_Z.shape

    alpha_batch = np.zeros((batch, n))
    Z_batch = np.zeros((batch, n, r))

    for b in range(batch):
        indices = np.random.choice(n, size=n, replace=True)
        alpha_batch[b] = model_alpha[indices]
        Z_batch[b] = model_Z[indices]

    return alpha_batch, Z_batch

def er_resample_gnp(A, B=1, seed=None):

    A = np.asarray(A)
    n = A.shape[0]

    E = np.triu(A, k=1).sum()
    M = n * (n - 1) // 2
    p_hat = float(E / M) if M > 0 else 0.0

    rng = np.random.default_rng(seed)

    if B == 1:
        U = rng.random((n, n)) < p_hat
        U = np.triu(U, 1).astype(np.uint8)
        G = U + U.T
        return G, p_hat
    else:
        samples = np.empty((B, n, n), dtype=np.uint8)
        for b in range(B):
            U = rng.random((n, n)) < p_hat
            U = np.triu(U, 1).astype(np.uint8)
            G = U + U.T
            samples[b] = G
        return samples, p_hat

def lcc_mask_from_dense(A):
    S = sp.csr_matrix(A)
    S.setdiag(0); S.eliminate_zeros()
    _, labels = connected_components(S, directed=False, return_labels=True)
    counts = np.bincount(labels)
    lcc_label = counts.argmax()
    return (labels == lcc_label)

def lcc_density_from_dense(A):
    mask = lcc_mask_from_dense(A)
    n_lcc = int(mask.sum())
    if n_lcc <= 1:
        return n_lcc, 0.0
    S = sp.csr_matrix(A)
    S.setdiag(0); S.eliminate_zeros()
    S_lcc = S[mask][:, mask]
    m_lcc = sp.triu(S_lcc, k=1).nnz
    density = (2.0 * m_lcc) / (n_lcc * (n_lcc - 1))
    return n_lcc, float(density)

def largest_connected_component_dense(A):
    S = sp.csr_matrix(A)
    S.setdiag(0); S.eliminate_zeros()
    _, labels = connected_components(S, directed=False, return_labels=True)
    counts = np.bincount(labels)
    lcc_label = counts.argmax()
    mask = (labels == lcc_label)
    A_lcc = A[np.ix_(mask, mask)]
    return A_lcc, mask