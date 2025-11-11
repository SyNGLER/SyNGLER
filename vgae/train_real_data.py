import torch
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, average_precision_score
import scipy.sparse as sp
import numpy as np
import os
import time
from tqdm import tqdm
from input_data import load_data
from input_simulated import load_real_data
from preprocessing import *
from args import get_args
args = get_args()

import model

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""

# adj, features = load_data(args.dataset)
adj, features = load_real_data(args.data_path)

# Store original adjacency matrix (without diagonal entries) for later
adj_orig = adj
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
adj_orig.eliminate_zeros()

adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
adj = adj_train

# Some preprocessing
adj_norm = preprocess_graph(adj)

num_nodes = adj.shape[0]

features = sparse_to_tuple(features.tocoo())
num_features = features[2][1]
features_nonzero = features[1].shape[0]

pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

adj_label = adj_train + sp.eye(adj_train.shape[0])
adj_label = sparse_to_tuple(adj_label)

adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].T),
                            torch.FloatTensor(adj_norm[1]),
                            torch.Size(adj_norm[2]))
adj_label = torch.sparse.FloatTensor(torch.LongTensor(adj_label[0].T),
                            torch.FloatTensor(adj_label[1]),
                            torch.Size(adj_label[2]))
features = torch.sparse.FloatTensor(torch.LongTensor(features[0].T),
                            torch.FloatTensor(features[1]),
                            torch.Size(features[2]))

weight_mask = adj_label.to_dense().view(-1) == 1
weight_tensor = torch.ones(weight_mask.size(0))
weight_tensor[weight_mask] = pos_weight

def density(adj_sp):
    n = adj_sp.shape[0]
    m = float(sp.triu(adj_sp, k=1).sum())
    return m / (n * (n - 1) / 2)

def _logit(P, eps=1e-6):
    P = torch.clamp(P, eps, 1 - eps)
    return torch.log(P / (1 - P))

def _calibrate(logits_tri_vec, target_rho, lo=-15.0, hi=15.0, steps=40):
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        mean_p = torch.sigmoid(logits_tri_vec - mid).mean().item()
        if mean_p > target_rho:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

@torch.no_grad()
def calibrate(P, rho_target):

    P = (P + P.t()) / 2
    P.fill_diagonal_(0.0)

    n = P.size(0)
    tri_mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=P.device), 1)
    logits = _logit(P)
    c = _calibrate(logits[tri_mask], rho_target)
    P2 = torch.sigmoid(logits - c)
    P2 = (P2 + P2.t()) / 2
    P2.fill_diagonal_(0.0)
    return P2

model = getattr(model, args.model)(adj_norm)
optimizer = Adam(model.parameters(), lr=args.learning_rate)

os.makedirs(args.output_dir, exist_ok=True)

N   = adj.shape[0]
Fin = args.input_dim
h   = args.hidden1_dim
r   = args.hidden2_dim
nnz = int(adj_norm._nnz())
iters_per_epoch = 1
epochs = args.num_epoch


def get_scores(edges_pos, edges_neg, adj_rec):

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]].item()))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]].data))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score

def get_acc(adj_rec, adj_label):
    labels_all = adj_label.to_dense().view(-1).long()
    preds_all = (adj_rec > 0.5).view(-1).long()
    accuracy = (preds_all == labels_all).sum().float() / labels_all.size(0)
    return accuracy

for epoch in range(args.num_epoch):
    t = time.time()

    A_pred = model(features)
    optimizer.zero_grad()
    loss = log_lik = norm * F.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1), weight=weight_tensor)
    if args.model == 'VGAE':
        kl_divergence = 0.5 / A_pred.size(0) * (1 + 2 * model.logstd - model.mean**2 - torch.exp(model.logstd)**2).sum(1).mean()
        loss -= kl_divergence

    loss.backward()
    optimizer.step()

    train_acc = get_acc(A_pred, adj_label)
    val_roc, val_ap = get_scores(val_edges, val_edges_false, A_pred)
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss.item()),
          "train_acc=", "{:.5f}".format(train_acc), "val_roc=", "{:.5f}".format(val_roc),
          "val_ap=", "{:.5f}".format(val_ap),
          "time=", "{:.5f}".format(time.time() - t))

test_roc, test_ap = get_scores(test_edges, test_edges_false, A_pred)
print("End of training!", "test_roc=", "{:.5f}".format(test_roc),
      "test_ap=", "{:.5f}".format(test_ap))

rho_train = density(adj_train)
def generate(model, features, output_dir, num_samples=100, rho_target=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model.eval()
    for i in tqdm(range(num_samples), desc="Generating samples"):
        P = model(features).detach()
        if rho_target is not None:
            P = calibrate(P, rho_target)
        else:
            P = (P + P.t()) / 2
            P.fill_diagonal_(0.0)
        output_file = os.path.join(output_dir, f"rep{i}.npy")
        np.save(output_file, P.cpu().numpy().astype(np.float32))

generate(model, features, args.output_dir, num_samples=0, rho_target=rho_train)
