import numpy as np 

def diag_delete(A):
    return np.triu(A, 1) + np.triu(A, 1).T

def sigmoid(x):
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x))) 

def sigmoid_prime(x):
    s = sigmoid(x)
    return s * (1 - s)

def sigmoid_prime_prime(x):
    s = sigmoid(x)
    return s * (1 - s) * (1 - 2 * s)

def symmetrization(X):
    if X.ndim != 3 or X.shape[0] != X.shape[1]:
        raise ValueError("Input array X must be 3-dimensional and square along the first two dimensions.")
    Y = np.zeros_like(X) 
    for j in range(X.shape[2]):
        upper_triangular = np.triu(X[:, :, j], k=1)  
        Y[:, :, j] = upper_triangular + upper_triangular.T
    return Y

def UniformCovariateSampler(n, p, low = -1, high = 1):
    return np.random.uniform(low, high, (n, p))

def GaussianCovariateSampler(n, mu = np.array([0]), Sigma = np.array([[1]])): 
    return np.random.multivariate_normal(mu, Sigma, n)


def GaussianMixtureSampler(n, mu_list, prob_list, Sigma_list):
    p = mu_list[0].shape[0]
    X = np.zeros((n, p))
    for i in range(n):
        index = np.random.choice(len(mu_list), p = prob_list)
        X[i] = np.random.multivariate_normal(mu_list[index], Sigma_list[index]) 
    return X 

def MatrixBernoilliSampler(P):
    return np.random.binomial(1, P, size = P.shape)

def MatrixGaussianSampler(P, sigma = 1):
    return np.random.normal(P,sigma, P.shape)

def MatrixClippedGaussianSampler(P, sigma = 1):
    return np.clip(np.random.normal(P,sigma, P.shape), -2, 2)

def topk_sqrt_eig_embedding(M, k):
    eigvals, eigvecs = np.linalg.eigh(M)

    topk_vals = eigvals[-k:] 
    topk_vecs = eigvecs[:, -k:]

    sqrt_topk_vals = np.sqrt(topk_vals)
    UD_sqrt = topk_vecs * sqrt_topk_vals  

    return UD_sqrt

class DataGenerator:
    def __init__(self, beta, X, Z_enable=True, alpha_enable=True, act = sigmoid, sparsity = 0.0):

        n = X.shape[0]
        p = X.shape[2]
        self.n, self.p= n, p
        

        self.beta, self.X = beta, X
        self.Z_enable, self.alpha_enable = Z_enable, alpha_enable
        self.act = act
        self.sparsity = sparsity
    
    def RefreshLatentVar(self, Z_sampler, alpha_sampler, Z_standardize = False, tau = 0.0):
        
        if self.Z_enable:
            Z = Z_sampler(self.n)
            if Z_standardize:
                Z = center_and_rotate(center_and_rotate(Z))
            self.Z = Z
            Theta =  Z.__matmul__(Z.T)
        for k in range(self.p):
            self.X[:, :, k] = (1 - tau) * self.X[:, :, k] + tau * np.outer(Z[:,k],Z[:,k])

        self.X = symmetrization(self.X)
        Theta += np.einsum('ijk, k -> ij', self.X, self.beta)
        if self.alpha_enable:
            alpha = alpha_sampler(self.n)
            self.alpha = alpha.reshape(-1)
            self.alpha = self.alpha - self.alpha.mean()
            Theta += np.outer(alpha, np.ones(self.n)) + np.outer(np.ones(self.n), alpha) 

        Theta += np.ones((self.n, self.n)) * self.sparsity
        Theta = np.triu(Theta, 1) + np.triu(Theta, 1).T 
        
        P = self.act(Theta)
        P = np.triu(P, 1) + np.triu(P, 1).T
    
        self.Theta, self.P = Theta, P

    def DataInstance(self, noisesampler):
        A = noisesampler(self.P)
        return np.triu(A, 1) + np.triu(A, 1).T 

class Model:
    def __init__(self, A, X, alpha=None, Z=None, beta=None, Z_enable=True, alpha_enable=True, act = sigmoid, sparsity = None, sparsity_estimation = False, Z_standardize = True):
        self.A, self.X = A, X 
        self.beta = beta
        self.n = A.shape[0]
        if Z_enable:
            if Z is None:
                raise ValueError('Z should be provided')
            self.r = Z.shape[1]
            self.Z = Z
            self.G = Z @ Z.T
        if alpha_enable:
            if alpha is None:
                raise ValueError('alpha should be provided')
            self.alpha = alpha
        self.p = X.shape[2]
        self.Z_enable, self.alpha_enable = Z_enable, alpha_enable
        self.act = act
        if sparsity is None:
            self.sparsity = 0.0
        else: 
            self.sparsity = sparsity

        self.sparsity_estimation = sparsity_estimation
        self.Z_standardize = Z_standardize

    
    def PGD_single_step(self, eta_alpha = 1e-2, eta_Z = 1e-2, eta_beta=1e-2, if_init = False):
        
        Theta = np.einsum('ijk, k -> ij', self.X, self.beta)
        if self.Z_enable:
            Theta += self.G
        
        if self.alpha_enable:
            Theta += np.outer(self.alpha, np.ones(self.n)) + np.outer(np.ones(self.n), self.alpha) 
        
        Theta += (np.ones((self.n, self.n)) - np.eye(self.n)) * self.sparsity
        Theta = np.triu(Theta, 1) + np.triu(Theta, 1).T
        pred = self.act(Theta)
        pred = np.triu(pred, 1) + np.triu(pred, 1).T
        self.P = pred
        diff = (self.A - pred)

        if self.Z_enable:
            if if_init:
                self.G = self.G + 2 * eta_Z * (diff - 1e-3 * self.G - 1e-3 * np.eye(self.n))
                J = np.eye(self.n) - np.ones((self.n, self.n)) / self.n
                self.G = J @ self.G @ J
            else:
                self.Z = self.Z + 2 * eta_Z * np.einsum('ij, jk -> ik', diff, self.Z) 
                if self.Z_standardize:
                    self.Z = center_and_rotate(center_and_rotate(self.Z))
                self.G = self.Z @ self.Z.T
        if self.alpha_enable:
            self.alpha = self.alpha + 2 * eta_alpha * np.einsum('ij, j -> i', diff, np.ones(self.n))
        self.beta = self.beta + 2 * eta_beta * np.einsum('ij, ijk -> k', diff, self.X)
        
        
        return np.linalg.norm(diff, 'fro')**2, Theta
    
    def PGD_initialization(self,eta_alpha=1e-2, eta_Z=1e-2, eta_beta=1e-2, n_iter = 2000, eps =1e-5,no_beta = False):
        
        self.converged = False
        beta_old = self.beta
        loss_step_old = 0
        
        for i in range(n_iter):
            
            loss_step, _ = self.PGD_single_step(eta_alpha, eta_Z, eta_beta, if_init=True)
            print(f"Iteration {i}: loss = {loss_step:.2f}, beta_1 {self.beta[0]:.3f}, z_11 {self.Z[0,0] if self.Z_enable else None:.3f}")
            if np.abs(loss_step - loss_step_old) < eps:
                self.converged = True
                self.step = i
                break
            loss_step_old = loss_step
        self.Z = topk_sqrt_eig_embedding(self.G, self.r) 
        self.Z = center_and_rotate(self.Z)

    def PGD(self, eta_alpha=1e-2, eta_Z=1e-2, eta_beta=1e-2, eps = 1e-4, n_iter = 2000, early_stop = True, verbose = True):

        beta_traj = []
        loss = []

        beta_traj.append(self.beta)
        self.converged = False

        for i in range(n_iter):
            loss_step, _ = self.PGD_single_step(eta_alpha, eta_Z, eta_beta)
            if verbose:
                print(f"Iteration {i}: loss = {loss_step:.2f}, beta_1 {self.beta[0]:.3f}, z_11 {self.Z[0,0] if self.Z_enable else None:.3f}")
            loss.append(loss_step)
            beta_traj.append(self.beta)
            
            if i>2 and np.abs(loss[-2] - loss[-1]) < eps and early_stop:
                self.converged = True
                self.step = i
                break

        if self.sparsity_estimation:
            self.sparsity = self.alpha.mean() * 2
            self.alpha = self.alpha - np.mean(self.alpha)

        self.Theta = np.einsum('ijk, k -> ij', self.X, self.beta)
        if self.Z_enable:
            self.Theta += self.Z.__matmul__(self.Z.T) 
        if self.alpha_enable:
            self.Theta += np.outer(self.alpha, np.ones(self.n)) + np.outer(np.ones(self.n), self.alpha)
        self.Theta += (np.ones((self.n, self.n)) - np.eye(self.n)) * self.sparsity 
        pred = self.act(self.Theta)
        pred = np.triu(pred, 1) + np.triu(pred, 1).T
        self.P = pred

        
        
        return np.array(beta_traj), np.array(loss)

def bias_est_functional(model, var_phi=None, var_beta=None, adjustment=True, lr_adj = None, loss_prime_3 = sigmoid_prime_prime, X = None):
    if X is None:
        if adjustment:
            if lr_adj is None:
                print("adjusting with default learning rate")
                X =  model.X - adjustment_functional(model)
            else:
                print("adjusting with specified learning rate")
                X =  model.X - adjustment_functional(model, lr=lr_adj)
        else:
            X = model.X
    

    M = loss_prime_3(model.Theta)[:,:,None] * X

    if var_phi is None:
        var_phi = var_phi_functional_primary(model) 
    if var_beta is None:
        var_beta = var_beta_functional(model)
    H = H_functional(model)
    bias_est = var_beta @ np.einsum('jr,js,ijp,irs->p',H,H,M,var_phi)

    return bias_est / 2

def var_phi_functional_primary(model, loss_prime_prime = sigmoid_prime):
    var_list = []
    n = model.Theta.shape[0]
    H = H_functional(model) 
    for i in range(n):
        vs = H.T @ (H * loss_prime_prime(model.Theta)[i][:,np.newaxis]) 
        var_list.append(np.linalg.inv(vs))
        
    return np.array(var_list)


def var_beta_functional(model, loss_prime_prime = sigmoid_prime, X = None):
    if X is None:
        return np.linalg.inv(np.einsum('ij,ijp,ijq->pq',loss_prime_prime(model.Theta),model.X,model.X) / 2)
    else:
        return np.linalg.inv(np.einsum('ij,ijp,ijq->pq',loss_prime_prime(model.Theta),X,X) / 2)


def center_and_rotate(Z):

    n, r = Z.shape 
    Z_centered = Z - np.mean(Z, axis=0, keepdims=True)
    if r == 1:
        return Z_centered * np.sign(Z_centered[0,0])
    else:    
        cov = Z_centered.T @ Z_centered / n
        _, eigenvectors = np.linalg.eigh(cov)
        Z_rotated = Z_centered @ eigenvectors
        return Z_rotated * np.sign(Z_rotated[0,0])


def H_functional(model):

    if hasattr(model, 'Z') and hasattr(model, 'alpha'):
        H = np.hstack([model.Z, np.ones((model.X.shape[0], 1))])
    elif hasattr(model, 'Z'):
        H = model.Z
    elif hasattr(model, 'alpha'):
        H = np.ones((model.X.shape[0], 1))
    else:
        raise ValueError("Model must have either Z or alpha defined.") 
    return H


def adjustment_functional(model, eps=1e-6, lr=0.01, max_iter=50000, loss_prime_prime = sigmoid_prime):  

    H = H_functional(model) 
    L_prime_prime = loss_prime_prime(model.Theta)
    L_prime_prime = diag_delete(L_prime_prime)
    mask = np.triu(np.ones((model.X.shape[0], model.X.shape[0])), k=1) + np.triu(np.ones((model.X.shape[0], model.X.shape[0])), k=1).T 
    
    def update_adjustment(xi, lr=lr, standardize=False):
        predicted = (np.einsum('kir,jr->ijk', xi, H) + np.einsum('kir,jr->jik', xi, H)) * mask[:, :, None]
        grad = np.einsum('ij,ijk,jr->kir', L_prime_prime, model.X - predicted, H)
        updated = xi + lr * grad
        if standardize and model.Z_enable:
            updated[:, -1] = center_and_rotate(updated[:, -1]) 
        return updated, predicted

    xi = np.tile(H[np.newaxis, :, :], (model.beta.shape[0], 1, 1))
    predicted = (np.einsum('kir,jr->ijk', xi, H)  + np.einsum('kir,jr->jik', xi, H)) * mask[:, :, None]
    for i in range(max_iter):
        xi_new, predicted_new = update_adjustment(xi, lr=lr)
        diff_norm = np.sum(np.square(predicted_new - predicted))
        if i >=2 and diff_norm > diff_norm_old:
            lr = lr / 2
            xi_new, predicted_new = update_adjustment(xi, lr=lr)
            diff_norm = np.sum(np.square(predicted_new - predicted))
        diff_norm_old = diff_norm
        print(f"Iteration {i}: diff_norm = {diff_norm:.7f}")
        predicted = predicted_new
        xi = xi_new
        if i>2 and diff_norm < eps:
            print("Convergence reached.")
            break
    return predicted

def matched_error(Z1, Z2):
    signs= np.sign(np.diag(Z1.T @ Z2))
    Z2_flipped = Z2 * signs
    return min(np.linalg.norm(Z1 - Z2_flipped, "fro"), np.linalg.norm(Z1 - Z2_flipped, "fro"))**2 / Z1.shape[0]
