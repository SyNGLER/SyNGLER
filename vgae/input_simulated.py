import numpy as np
import pickle
import scipy.sparse as sp
from LSM_source import MatrixBernoilliSampler

def load_simulated_data(filename):

    print(f"Loading {filename}")
    with open(filename, "rb") as f:
        loader = pickle.load(f)
        data = loader["data"]

    np.random.seed(0)

    A = data.DataInstance(MatrixBernoilliSampler)
    A = A.copy()
    if not sp.issparse(A):
        adj = sp.csr_matrix(A)
    else:
        adj = A.copy()
    num_nodes = adj.shape[0]
    features = sp.identity(num_nodes)
    
    return adj, features

def load_real_data(filename):

    A = np.load(filename)
    if not sp.issparse(A):
        adj = sp.csr_matrix(A)
    else:
        adj = A.copy()
    num_nodes = adj.shape[0]
    features = sp.identity(num_nodes)
    return adj, features
