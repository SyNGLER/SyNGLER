
import os
from tqdm.auto import tqdm

USE_GPU = os.environ.get("USE_GPU", "7") == "1"
if USE_GPU:
    import cupy as xp
    from cupy import linalg as xla
else:
    import numpy as xp
    from numpy import linalg as xla

def as_xp(a, dtype=xp.float32):
    try:
        import cupy as cp
        if USE_GPU:
            if isinstance(a, cp.ndarray):
                return a.astype(dtype, copy=False)
            else:
                return cp.asarray(a, dtype=dtype)
        else:
            import numpy as np
            if isinstance(a, np.ndarray):
                return a.astype(dtype, copy=False)
            else:
                return np.asarray(a, dtype=dtype)
    except Exception:
        import numpy as np
        return np.asarray(a, dtype=dtype)

def diag_delete(A):
    U = xp.triu(A, 1)
    return U + U.T

def sigmoid(x):
    return xp.where(x >= 0, 1.0 / (1.0 + xp.exp(-x)), xp.exp(x) / (1.0 + xp.exp(x)))

def logit(p, eps=1e-8):
    p = xp.clip(p, eps, 1 - eps)
    return xp.log(p / (1 - p))

def sigmoid_prime(x):
    s = sigmoid(x)
    return s * (1 - s)

def sigmoid_prime_prime(x):
    s = sigmoid(x)
    return s * (1 - s) * (1 - 2 * s)

def symmetrization(X):
    if X.ndim != 3 or X.shape[0] != X.shape[1]:
        raise ValueError("X must be 3D and square along the first two dims.")
    Y = xp.zeros_like(X)
    for j in range(X.shape[2]):
        U = xp.triu(X[:, :, j], 1)
        Y[:, :, j] = U + U.T
    return Y

def UniformCovariateSampler(n, p, low=-1, high=1):
    return xp.random.uniform(low, high, (n, p))

def GaussianCovariateSampler(n, mu=None, Sigma=None):
    if mu is None:
        mu = xp.zeros(1, dtype=xp.float32)
    if Sigma is None:
        Sigma = xp.eye(mu.shape[0], dtype=xp.float32)
    return xp.random.multivariate_normal(mu, Sigma, n)

def ClippedGaussianCovariateSampler(n, mu=None, Sigma=None, low=-2, up=2):
    X = GaussianCovariateSampler(n, mu, Sigma)
    return xp.clip(X, low, up)

def GaussianMixtureSampler(n, mu_list, prob_list, Sigma_list):
    p = mu_list[0].shape[0]
    X = xp.zeros((n, p), dtype=xp.float32)
    K = len(mu_list)
    idx = xp.random.choice(K, size=n, p=as_xp(prob_list))
    for i in range(n):
        mu = as_xp(mu_list[idx[i]])
        Sigma = as_xp(Sigma_list[idx[i]])
        X[i] = xp.random.multivariate_normal(mu, Sigma)
    return X

def MatrixBernoilliSampler(P):
    return (xp.random.random(P.shape) < P).astype(P.dtype)

def MatrixGaussianSampler(P, sigma=1.0):
    return P + sigma * xp.random.standard_normal(P.shape, dtype=P.dtype)

def MatrixClippedGaussianSampler(P, sigma=1.0):
    return xp.clip(MatrixGaussianSampler(P, sigma=sigma), -2, 2)

def topk_sqrt_eig_embedding(M, k):
    w, V = xla.eigh(M)
    idx = xp.argsort(w)[-k:]
    vals = xp.clip(w[idx], 0, None)
    return V[:, idx] * xp.sqrt(vals)

def center_and_rotate(Z):
    n, r = Z.shape
    Zc = Z - xp.mean(Z, axis=0, keepdims=True)
    if r == 1:
        return Zc * xp.sign(Zc[0, 0])
    cov = (Zc.T @ Zc) / float(n)
    _, Q = xla.eigh(cov)
    Zr = Zc @ Q
    return Zr * xp.sign(Zr[0, 0])

def matched_error(Z1, Z2):
    S = xp.sign(xp.diag(Z1.T @ Z2))
    Z2f = Z2 * S
    num = xp.linalg.norm(Z1 - Z2f, "fro") ** 2
    return float(num / Z1.shape[0])

def H_functional(model):
    if hasattr(model, 'Z') and model.Z is not None and hasattr(model, 'alpha') and model.alpha is not None:
        return xp.hstack([model.Z, xp.ones((model.n, 1), dtype=model.Z.dtype)])
    elif hasattr(model, 'Z') and model.Z is not None:
        return model.Z
    elif hasattr(model, 'alpha') and model.alpha is not None:
        return xp.ones((model.n, 1), dtype=model.alpha.dtype)
    else:
        raise ValueError("Model must define Z and/or alpha.")

def var_phi_functional_primary(model, loss_prime_prime=sigmoid_prime):
    n = model.n
    H = H_functional(model)
    Lpp = loss_prime_prime(model.Theta)
    out = []
    for i in range(n):
        vs = H.T @ (H * Lpp[i][:, None])
        out.append(xla.inv(vs))
    return xp.stack(out, axis=0)

def var_beta_functional(model, loss_prime_prime=sigmoid_prime, X=None):
    Xuse = model.X if X is None else X
    if Xuse.shape[2] == 0:
        return xp.eye(0, dtype=model.Theta.dtype)
    M = xp.einsum('ij,ijp,ijq->pq', loss_prime_prime(model.Theta), Xuse, Xuse) / 2.0
    return xla.inv(M)

def adjustment_functional(model, eps=1e-6, lr=1e-2, max_iter=50000, loss_prime_prime=sigmoid_prime):
    H = H_functional(model)
    Lpp = diag_delete(loss_prime_prime(model.Theta))
    n, _, p = model.X.shape
    d = H.shape[1]
    mask = xp.triu(xp.ones((n, n), dtype=model.X.dtype), 1)
    mask = mask + mask.T

    def predict_xi(xi):
        t1 = xp.einsum('k i r, j r -> k i j', xi, H)
        t2 = xp.einsum('k j r, i r -> k i j', xi, H)
        pred = (t1 + t2) * mask[None, :, :]
        return pred.transpose(1, 2, 0)

    def grad_step(xi, lr_):
        pred = predict_xi(xi)
        R = model.X - pred
        grad = xp.einsum('ij, ijp, j r -> p i r', Lpp, R, H)
        return xi + lr_ * grad, pred
    if p == 0:
        return xp.zeros_like(model.X)
    xi = xp.tile(H[None, :, :], (p, 1, 1))
    pred = predict_xi(xi)

    diff_old = xp.inf
    for it in range(max_iter):
        xi_new, pred_new = grad_step(xi, lr)
        diff = float(xp.sum((pred_new - pred) ** 2))
        if it >= 2 and diff > diff_old:
            lr = lr / 2.0
            xi_new, pred_new = grad_step(xi, lr)
            diff = float(xp.sum((pred_new - pred) ** 2))
        diff_old = diff
        pred = pred_new
        xi = xi_new
        if it >= 2 and diff < eps:
            break
    return pred

def bias_est_functional(model, var_phi=None, var_beta=None, adjustment=True, lr_adj=None,
                        loss_prime_3=sigmoid_prime_prime, X=None):
    if X is None:
        if adjustment:
            Xuse = model.X - (adjustment_functional(model, lr=(1e-2 if lr_adj is None else lr_adj)))
        else:
            Xuse = model.X
    else:
        Xuse = X

    M = loss_prime_3(model.Theta)[:, :, None] * Xuse
    if var_phi is None:
        var_phi = var_phi_functional_primary(model)
    if var_beta is None:
        var_beta = var_beta_functional(model)
    H = H_functional(model)
    bias_est = var_beta @ xp.einsum('jr,js,ijp,irs->p', H, H, M, var_phi)
    return bias_est / 2.0

class DataGenerator:
    def __init__(self, beta, X, Z_enable=True, alpha_enable=True, act=sigmoid, sparsity=0.0):
        self.X = as_xp(X)
        self.beta = as_xp(beta)
        self.n = self.X.shape[0]
        self.p = self.X.shape[2]
        self.Z_enable = Z_enable
        self.alpha_enable = alpha_enable
        self.act = act
        self.sparsity = float(sparsity)
        self.Z = None
        self.alpha = None
        self.Theta = None
        self.P = None

    def RefreshLatentVar(self, Z_sampler, alpha_sampler, Z_standardize=False, tau=0.0):
        n, p = self.n, self.p
        Theta = xp.zeros((n, n), dtype=xp.float32)

        if self.Z_enable:
            Z = as_xp(Z_sampler(n))
            if Z_standardize:
                Z = center_and_rotate(Z)
            self.Z = Z
            Theta = Theta + Z @ Z.T
        if p > 0 and self.Z_enable:
            for k in range(p):
                self.X[:, :, k] = (1 - tau) * self.X[:, :, k] + tau * xp.outer(self.Z[:, k], self.Z[:, k])

        self.X = symmetrization(self.X)
        if p > 0:
            Theta = Theta + xp.einsum('ijk,k->ij', self.X, self.beta)

        if self.alpha_enable:
            alpha = as_xp(alpha_sampler(n)).reshape(-1)
            alpha = alpha - xp.mean(alpha)
            self.alpha = alpha
            Theta = Theta + xp.outer(alpha, xp.ones(n, dtype=Theta.dtype)) + xp.outer(xp.ones(n, dtype=Theta.dtype), alpha)

        Theta = Theta + (xp.ones((n, n), dtype=Theta.dtype) - xp.eye(n, dtype=Theta.dtype)) * self.sparsity
        Theta = diag_delete(Theta)
        P = self.act(Theta)
        P = diag_delete(P)

        self.Theta, self.P = Theta, P

    def DataInstance(self, noisesampler):
        A = as_xp(noisesampler(self.P))
        return diag_delete(A)

class Model:
    def __init__(self, A, X, alpha=None, beta=None, Z=None,
                 alpha_enable=True, Z_enable=True, Z_standardize=True,
                 act=sigmoid, sparsity=0.0, sparsity_estimation=False):
        self.A = as_xp(A)
        self.X = as_xp(X)
        self.n = self.A.shape[0]
        self.p = self.X.shape[2]
        self.act = act
        self.alpha_enable = alpha_enable
        self.Z_enable = Z_enable
        self.Z_standardize = Z_standardize
        self.sparsity_estimation = sparsity_estimation
        self.sparsity = float(sparsity)

        self.alpha = as_xp(alpha) if alpha is not None else None
        self.beta = as_xp(beta) if beta is not None else xp.zeros((self.p,), dtype=self.A.dtype)
        self.Z = as_xp(Z) if Z is not None else None
        self.r = 0 if self.Z is None else int(self.Z.shape[1])

        self.G = None
        if self.Z_enable:
            if self.Z is None:
                raise ValueError("Z should be provided when Z_enable=True.")
            self.G = self.Z @ self.Z.T

        self.P = None
        self.Theta = None
        self.converged = False
        self.step = 0

    def PGD_single_step(self, eta_alpha=1e-2, eta_Z=1e-2, eta_beta=1e-2, if_init=False):
        n = self.n
        Theta = xp.zeros((n, n), dtype=self.A.dtype)
        if self.p > 0:
            Theta = Theta + xp.einsum('ijk,k->ij', self.X, self.beta)
        if self.Z_enable:
            Theta = Theta + self.G
        if self.alpha_enable:
            Theta = Theta + xp.outer(self.alpha, xp.ones(n, dtype=Theta.dtype)) + xp.outer(xp.ones(n, dtype=Theta.dtype), self.alpha)
        Theta = Theta + (xp.ones((n, n), dtype=Theta.dtype) - xp.eye(n, dtype=Theta.dtype)) * self.sparsity
        Theta = diag_delete(Theta)

        pred = self.act(Theta)
        pred = diag_delete(pred)
        self.P = pred

        diff = self.A - pred

        if self.Z_enable:
            if if_init:
                self.G = self.G + 2.0 * eta_Z * (diff - 1e-3 * self.G - 1e-3 * xp.eye(n, dtype=self.G.dtype))
                J = xp.eye(n, dtype=self.G.dtype) - xp.ones((n, n), dtype=self.G.dtype) / float(n)
                self.G = J @ self.G @ J
            else:
                self.Z = self.Z + 2.0 * eta_Z * (diff @ self.Z)
                if self.Z_standardize:
                    self.Z = center_and_rotate(self.Z)
                self.G = self.Z @ self.Z.T

        if self.alpha_enable:
            self.alpha = self.alpha + 2.0 * eta_alpha * (diff @ xp.ones(n, dtype=diff.dtype))

        if self.p > 0:
            self.beta = self.beta + 2.0 * eta_beta * xp.einsum('ij,ijk->k', diff, self.X)

        loss = float(xla.norm(diff, ord='fro') ** 2)
        return loss, Theta

    def PGD_initialization(self, eta_alpha=1e-2, eta_Z=1e-2, eta_beta=1e-2, n_iter=2000, eps=1e-5):
        self.converged = False
        loss_old = 0.0
        for i in range(n_iter):
            loss_i, _ = self.PGD_single_step(eta_alpha, eta_Z, eta_beta, if_init=True)
            if abs(loss_i - loss_old) < eps:
                self.converged = True
                self.step = i
                break
            loss_old = loss_i
        if self.Z_enable:
            self.Z = topk_sqrt_eig_embedding(self.G, self.r)
            self.Z = center_and_rotate(self.Z)
            self.G = self.Z @ self.Z.T

    def PGD(self, eta_alpha=1e-2, eta_Z=1e-2, eta_beta=1e-2, eps=1e-4, n_iter=2000, early_stop=True, verbose=True):
        beta_traj = [xp.copy(self.beta)]
        losses = []
        self.converged = False

        for i in range(n_iter):
            loss_i, _ = self.PGD_single_step(eta_alpha, eta_Z, eta_beta, if_init=False)
            if verbose:
                z11 = float(self.Z[0, 0]) if self.Z_enable else None
                print(f"Iteration {i}: loss = {loss_i:.2f}, beta_1 {float(self.beta[0]) if self.p>0 else 0:.3f}, z_11 {z11:.3f}")
            losses.append(loss_i)
            beta_traj.append(xp.copy(self.beta))
            if i > 2 and abs(losses[-2] - losses[-1]) < eps and early_stop:
                self.converged = True
                self.step = i
                break

        if self.sparsity_estimation and self.alpha_enable:
            self.sparsity = float(xp.mean(self.alpha) * 2.0)
            self.alpha = self.alpha - xp.mean(self.alpha)

        Theta = xp.zeros((self.n, self.n), dtype=self.A.dtype)
        if self.p > 0:
            Theta = Theta + xp.einsum('ijk,k->ij', self.X, self.beta)
        if self.Z_enable:
            Theta = Theta + (self.Z @ self.Z.T)
        if self.alpha_enable:
            Theta = Theta + xp.outer(self.alpha, xp.ones(self.n, dtype=Theta.dtype)) + xp.outer(xp.ones(self.n, dtype=Theta.dtype), self.alpha)
        Theta = Theta + (xp.ones((self.n, self.n), dtype=Theta.dtype) - xp.eye(self.n, dtype=Theta.dtype)) * self.sparsity
        self.Theta = Theta
        self.P = diag_delete(self.act(Theta))

        return xp.stack(beta_traj), xp.asarray(losses, dtype=xp.float32)
