import numpy as np

import lsm_backend as LSM
from lsm_backend import sigmoid


cp = None
try:
    import cupy as _cp

    if _cp.cuda.runtime.getDeviceCount() > 0:
        cp = _cp
except Exception:
    cp = None


def _xp_logit(p, xp, eps=1e-6):
    p = xp.clip(p, eps, 1 - eps)
    return xp.log(p / (1 - p))


def init_from_A(A_np, r, X=None, tau=0.0, M1=8.0, use_gpu=True):
    using_gpu = bool(use_gpu and (cp is not None))
    xp = cp if using_gpu else np

    n = A_np.shape[0]
    A = xp.asarray(A_np, dtype=xp.float64)
    A = (A + A.T) / 2.0

    U, s, Vt = xp.linalg.svd(A, full_matrices=False)
    keep = s >= tau
    if not xp.any(keep):
        keep = xp.zeros_like(s, dtype=bool)
        keep[0] = True

    P_tilde = (U[:, keep] * s[keep]) @ Vt[keep, :]
    P_hat = xp.clip(P_tilde, 0.5 * xp.exp(-M1), 0.5)
    Theta_hat = _xp_logit((P_hat + P_hat.T) / 2.0, xp)

    one = xp.ones((n, 1), dtype=xp.float64)
    scalar = float((one.T @ Theta_hat @ one).item())
    alpha0 = (Theta_hat @ one / n - (scalar / (2 * n**2)) * one).ravel()

    if X is None or (np.ndim(X) == 0) or np.allclose(X, 0):
        beta0 = np.zeros(0, dtype=float) if X is None else np.zeros(X.shape[-1])
        BX_xp = 0.0
    else:
        iu, ju = np.triu_indices(n, k=1)
        Theta_hat_np = xp.asnumpy(Theta_hat)
        alpha0_np = xp.asnumpy(alpha0)
        y = Theta_hat_np[iu, ju] - alpha0_np[iu] - alpha0_np[ju]
        F = X[iu, ju, :]
        beta0, *_ = np.linalg.lstsq(F, y, rcond=None)
        BX_xp = xp.asarray(np.tensordot(X, beta0, axes=([2], [0])))

    J = xp.eye(n, dtype=xp.float64) - xp.ones((n, n), dtype=xp.float64) / n
    R = Theta_hat - alpha0[:, None] - alpha0[None, :] - (BX_xp if np.ndim(BX_xp) else 0.0)
    R = J @ ((R + R.T) / 2.0) @ J
    w, Ue = xp.linalg.eigh(R)
    w = xp.clip(w, 0, None)

    idx = xp.argsort(w)[::-1][:r]
    Uk = Ue[:, idx]
    Dk = xp.diag(xp.sqrt(w[idx]))
    Z0 = Uk @ Dk

    if using_gpu:
        Z0 = cp.asnumpy(Z0)
        alpha0 = cp.asnumpy(alpha0)
    else:
        Z0 = np.asarray(Z0)
        alpha0 = np.asarray(alpha0)

    return Z0, alpha0, beta0


def fit_lsm(
    A,
    r=5,
    seed=0,
    eta_0=0.1,
    tau=0.0,
    use_gpu=True,
    covariate_dim=2,
    sigma_init=0.1,
    n_iter=500000,
):
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    X = np.zeros((n, n, covariate_dim), dtype=float)

    Z0, alpha0, _ = init_from_A(A, r, X, tau=tau, M1=8.0, use_gpu=use_gpu)
    rho = -np.log(n) * 0.0

    eta_alpha = eta_0 / (2 * n)
    eta_Z = eta_0 / (2 * np.sum(Z0**2) / Z0.shape[1])

    np.random.seed(seed)
    init_Z = Z0 + sigma_init * np.random.randn(*Z0.shape)
    alpha_init = alpha0.reshape(-1) + sigma_init * np.random.randn(n)

    model = LSM.Model(
        A,
        X,
        alpha=alpha_init,
        beta=np.zeros(covariate_dim),
        Z=init_Z,
        alpha_enable=True,
        Z_enable=True,
        Z_standardize=True,
        act=sigmoid,
        sparsity=rho,
        sparsity_estimation=True,
    )

    model.PGD(
        eta_alpha=eta_alpha,
        eta_Z=eta_Z,
        eta_beta=0.0,
        early_stop=True,
        eps=1e-6,
        n_iter=n_iter,
        verbose=True,
    )

    return {
        "model_Z": np.asarray(model.Z),
        "model_alpha": np.asarray(model.alpha),
        "model_sparsity": float(model.sparsity),
        "converged": bool(model.converged),
        "r": int(r),
        "seed": int(seed),
        "eta_0": float(eta_0),
        "tau": float(tau),
        "covariate_dim": int(covariate_dim),
    }
