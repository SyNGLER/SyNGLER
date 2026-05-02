import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
from scipy.sparse import csr_matrix, issparse
from scipy.sparse.linalg import eigsh
from scipy.stats import norm
from sklearn.decomposition import FactorAnalysis


def inference_all_in_one(
    dat,
    d=None,
    choosing_maximum=100,
    max_Z3=10,
    use_sparse=True,
    variance_target=None,
):
    n = dat["n"]
    P1 = np.ones((n, n)) / n
    In = np.eye(n)

    A = dat["A"]
    if use_sparse and not issparse(A):
        A_sparse = csr_matrix(A)
    else:
        A_sparse = A

    mu_hat = np.mean(dat["Y"], axis=0)

    if issparse(A_sparse):
        alpha_hat = np.array(A_sparse.mean(axis=1)).flatten()
    else:
        alpha_hat = np.mean(A_sparse, axis=1)

    if d is None:
        search_limit = min(choosing_maximum, 100, dat["p"])
        d = choose_dimension(
            A_sparse,
            n=search_limit,
            plot=False,
            use_sparse=use_sparse,
            variance_target=variance_target,
        )

    Xhat = spec_emb(A_sparse, d=d, use_sparse=use_sparse)
    Xhat = correct_sign(Xhat)
    Z12hat = (In - P1) @ Xhat

    ZTZ = Z12hat.T @ Z12hat
    eigen_vals, eigen_vecs = np.linalg.eigh(ZTZ)
    idx = eigen_vals.argsort()[::-1]
    eigen_vecs = eigen_vecs[:, idx]
    Z12hat = Z12hat @ eigen_vecs

    L12hat = linalg.solve(Z12hat.T @ Z12hat, Z12hat.T @ (In - P1) @ dat["Y"]).T
    Rhat = (In - Z12hat @ linalg.solve(Z12hat.T @ Z12hat, Z12hat.T)) @ (In - P1) @ dat["Y"]

    eigenvals_CovR = compute_top_eigenvalues(Rhat.T, k=min(max_Z3 + 3, min(n, dat["p"])))
    max_Z3_actual = min(max_Z3, len(eigenvals_CovR) - 2)
    max_Z3_actual = max(0, max_Z3_actual)

    no_Z3_series = [
        test_Z3(eigenvals_CovR, k0=o, diff=1)["output"] for o in range(max_Z3_actual + 1)
    ]

    Num_Z3 = 0
    if len(no_Z3_series) > 1:
        for i in range(len(no_Z3_series) - 1):
            if not no_Z3_series[i] and no_Z3_series[i + 1]:
                Num_Z3 = i + 1
                break

    if Num_Z3 == 0:
        Psihat = np.diag(np.diag(np.cov(Rhat.T)))
        Lhat = L12hat
        Z3hat = None
    else:
        fac_res = factor_analysis(Rhat, Num_Z3, method="ml")
        sigmas = fac_res["Sigma"]
        Z3hat = fac_res["Z"]

        eigen_vals, eigen_vecs = np.linalg.eigh(Z3hat.T @ Z3hat)
        idx = eigen_vals.argsort()[::-1]
        eigen_vecs = eigen_vecs[:, idx]
        Z3hat = Z3hat @ eigen_vecs

        Psihat = np.diag(sigmas)
        Lhat = np.hstack([L12hat, fac_res["Gamma"]])

    psi_diag = np.diag(Psihat).astype(float)
    positive = psi_diag[psi_diag > 0]
    eps = 1e-6 * np.median(positive) if positive.size > 0 else 1e-6
    psi_diag_safe = np.maximum(psi_diag, eps)
    inv_psi = 1.0 / psi_diag_safe

    L_weighted = Lhat * inv_psi[:, None]
    ic_mat = Lhat.T @ L_weighted / dat["p"]

    eigen_vals, eigen_vecs = np.linalg.eigh(ic_mat)
    idx = eigen_vals.argsort()[::-1]
    eigen_vecs = eigen_vecs[:, idx]
    Lhat = Lhat @ eigen_vecs

    Num_Z1 = test_Z1(Z12hat, L12hat[:, : L12hat.shape[1]], np.diag(Psihat))["output"]

    return {
        "Num_Z1": Num_Z1,
        "Num_Z3": Num_Z3,
        "Z12hat": Z12hat,
        "Z3hat": Z3hat,
        "Lhat": Lhat,
        "mu_hat": mu_hat,
        "alpha_hat": alpha_hat,
    }


def spec_emb(A, d, use_sparse=True):
    if use_sparse and (issparse(A) or A.shape[0] > 1000):
        try:
            if issparse(A):
                ATA = A.T @ A
            else:
                ATA = csr_matrix(A.T @ A)

            eigenvals, eigenvecs = eigsh(ATA, k=min(d, A.shape[0] - 2), which="LA")
            eigenvals = eigenvals[::-1]
            eigenvecs = eigenvecs[:, ::-1]
        except Exception as exc:
            warnings.warn(
                f"Sparse decomposition failed: {exc}. Falling back to dense computation."
            )
            if issparse(A):
                A = A.toarray()
            eigenvals, eigenvecs = np.linalg.eigh(A.T @ A)
            idx = eigenvals.argsort()[::-1]
            eigenvals = eigenvals[idx][:d]
            eigenvecs = eigenvecs[:, idx][:, :d]
    else:
        if issparse(A):
            A = A.toarray()
        eigenvals, eigenvecs = np.linalg.eigh(A.T @ A)
        idx = eigenvals.argsort()[::-1]
        eigenvals = eigenvals[idx][:d]
        eigenvecs = eigenvecs[:, idx][:, :d]

    eigenvals_positive = np.maximum(eigenvals, 1e-10)
    D = np.diag(np.sqrt(np.sqrt(eigenvals_positive)))
    return eigenvecs @ D


def compute_top_eigenvalues(X, k=None):
    n_features, n_samples = X.shape
    if k is None:
        k = min(n_features, n_samples)

    X_centered = X - X.mean(axis=1, keepdims=True)

    if n_features > n_samples:
        if k < n_samples - 1:
            try:
                eigenvals, _ = eigsh(X_centered.T @ X_centered / n_samples, k=k, which="LA")
                return eigenvals[::-1]
            except Exception:
                pass
        cov_small = (X_centered.T @ X_centered) / n_samples
        eigenvals = np.linalg.eigvalsh(cov_small)[::-1]
        return eigenvals[:k]

    if k < min(n_features, n_samples) - 1:
        try:
            cov_matrix = np.cov(X_centered)
            if n_features > 500:
                cov_sparse = csr_matrix(cov_matrix)
                eigenvals, _ = eigsh(cov_sparse, k=k, which="LA")
                return eigenvals[::-1]
        except Exception:
            pass

    eigenvals = np.linalg.eigvalsh(np.cov(X_centered))[::-1]
    return eigenvals[:k]


def correct_sign(X):
    X_corrected = X.copy()
    for j in range(X.shape[1]):
        if X_corrected[0, j] < 0:
            X_corrected[:, j] = -X_corrected[:, j]
    return X_corrected


def test_Z3(eigenvals, k0=0, diff=1, level="ninety_five"):
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


def test_Z1(Z12hat, L12hat, sigmas, level=0.05):
    k12 = L12hat.shape[1]
    p = L12hat.shape[0]
    V = np.zeros((p, k12))
    Z12_inv = linalg.solve(Z12hat.T @ Z12hat, np.eye(k12))
    for j in range(p):
        sigma_j = sigmas[j]
        for k in range(k12):
            V[j, k] = sigma_j * np.diag(Z12_inv)[k]

    S = np.zeros(k12)
    for k in range(k12):
        S[k] = (np.sum(L12hat[:, k] ** 2) - np.sum(V[:, k])) / np.sqrt(
            2 * np.sum(V[:, k] ** 2)
        )
    critical_value = norm.ppf(1 - level / 2)
    return {"S": S, "output": np.sum(np.abs(S) < critical_value)}


def factor_analysis(R, n_components, method="ml"):
    del method
    fa = FactorAnalysis(n_components=n_components, random_state=0)
    Z = fa.fit_transform(R)
    return {"Z": Z, "Gamma": fa.components_.T, "Sigma": fa.noise_variance_}


def choose_dimension(A, n=100, plot=True, use_sparse=True, variance_target=None):
    matrix_size = A.shape[0]
    n = min(n, 100, matrix_size - 1)

    if issparse(A):
        total_variance = A.power(2).sum()
    else:
        total_variance = np.sum(A**2)

    try:
        if use_sparse and issparse(A):
            ATA = A.T @ A
            eigenvals, _ = eigsh(ATA, k=n, which="LA")
        else:
            if matrix_size > 2000:
                if not issparse(A):
                    A_sparse_tmp = csr_matrix(A)
                    ATA = A_sparse_tmp.T @ A_sparse_tmp
                    eigenvals, _ = eigsh(ATA, k=n, which="LA")
            else:
                ATA = A.T @ A
                eigenvals = np.linalg.eigvalsh(ATA)[-n:]

        eigenvals = eigenvals[::-1]
    except Exception as exc:
        warnings.warn(f"Eigenvalue decomposition failed or timed out: {exc}. Fallback to d=10.")
        return 10

    x = np.maximum(eigenvals, 1e-10)
    n_actual = len(x)
    if n_actual < 2:
        return 1

    profile = np.zeros(n_actual)

    def var_new(vals):
        return 0 if len(vals) <= 1 else np.var(vals, ddof=1)

    for i in range(n_actual - 1):
        x_1 = x[: i + 1]
        x_2 = x[i + 1 : n_actual]
        if len(x_2) == 0:
            continue

        mean_1 = np.mean(x_1)
        mean_2 = np.mean(x_2)
        var_1 = var_new(x_1)
        var_2 = var_new(x_2)

        df = n_actual - 2
        pooled_var = ((len(x_1) - 1) * var_1 + (len(x_2) - 1) * var_2) / df if df > 0 else 1e-10
        sd = max(np.sqrt(pooled_var), 1e-10)

        profile[i] = np.sum(norm.logpdf(x_1, mean_1, sd)) + np.sum(norm.logpdf(x_2, mean_2, sd))

    p_likelihood = np.argmax(profile[: n_actual - 1]) + 1
    final_d = p_likelihood

    d_variance = 0
    if variance_target is not None:
        current_variance = np.cumsum(x)
        ratios = current_variance / total_variance
        idx = np.searchsorted(ratios, variance_target)
        if idx < len(ratios):
            d_variance = idx + 1
        else:
            d_variance = n_actual
            print(
                f"Warning: top {n_actual} dimensions only explain "
                f"{ratios[-1] * 100:.1f}% variance; using d={n_actual}."
            )

        final_d = max(p_likelihood, d_variance)
        print(f"Dim selection: likelihood d={p_likelihood}, variance d={d_variance} -> final d={final_d}")

    if plot:
        plt.figure(figsize=(8, 5))
        plt.plot(x, "o-", label="Top eigenvalues")
        plt.axvline(x=p_likelihood - 1, color="r", linestyle="--", label=f"Likelihood (d={p_likelihood})")
        if variance_target is not None and d_variance != p_likelihood:
            plt.axvline(x=d_variance - 1, color="g", linestyle="--", label=f"Variance (d={d_variance})")
        plt.title(
            f"Scree plot (top {n_actual} explain {np.sum(x) / total_variance * 100:.1f}% total variance)"
        )
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    return final_d
