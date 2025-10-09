import math
import torch
import torch_geometric as pyg
from datasets.data_utils import collate_fn

def _ensure_full_edges(bg):
    if not hasattr(bg, "nodes_per_graph"):
        return bg 

    n_per_graph = bg.nodes_per_graph
    full_edges_per_graph = (n_per_graph * (n_per_graph - 1)) // 2 
    bg.full_edges_per_graph = full_edges_per_graph
    bg.edges_per_graph = full_edges_per_graph

    if hasattr(bg, "log_full_edge_attr") and isinstance(bg.log_full_edge_attr, torch.Tensor):
        E_full = int(full_edges_per_graph.sum().item())
        if bg.log_full_edge_attr.shape[0] != E_full:
            raise RuntimeError(
                f"[EDGE] log_full_edge_attr length mismatch: "
                f"{bg.log_full_edge_attr.shape[0]} vs expected {E_full}. "
            )

    if not hasattr(bg, "degree"):
        if not hasattr(bg, "log_full_edge_attr"):
            raise RuntimeError("[EDGE] Missing degree.")
        deg_chunks = []
        start = 0
        device = bg.log_full_edge_attr.device
        for n in n_per_graph.tolist():
            n = int(n)
            E = n * (n - 1) // 2
            log_edges = bg.log_full_edge_attr[start:start+E]
            e = log_edges.argmax(dim=1)
            ij = torch.triu_indices(n, n, offset=1, device=device)
            d = torch.zeros(n, device=device, dtype=torch.float32)
            d.index_add_(0, ij[0], e.float())
            d.index_add_(0, ij[1], e.float())
            deg_chunks.append(d)
            start += E
        bg.degree = torch.cat(deg_chunks, dim=0)
    return bg

def loglik_nats(model, x):
    """Compute the log-likelihood in nats."""
    x = _ensure_full_edges(x)
    return - model.log_prob(x).mean()


def loglik_bpd(model, x):
    """Compute the log-likelihood in bits per dim."""
    x = _ensure_full_edges(x)
    return -model.log_prob(x).sum() / (math.log(2) * x.num_entries)
    # return - model.log_prob(x).sum() / (math.log(2) * x.shape.numel())


def elbo_nats(model, x):
    """
    Compute the ELBO in nats.
    Same as .loglik_nats(), but may improve readability.
    """
    x = _ensure_full_edges(x)
    return loglik_nats(model, x)


def elbo_bpd(model, x):
    """
    Compute the ELBO in bits per dim.
    Same as .loglik_bpd(), but may improve readability.
    """
    x = _ensure_full_edges(x)
    return loglik_bpd(model, x)


def iwbo(model, x, k):
    x = _ensure_full_edges(x)
    ll = -model.nll(x)
    # ll = torch.stack(torch.chunk(ll_stack, k, dim=0))
    # print(ll)
    return torch.logsumexp(ll, dim=0) - math.log(k)


# def iwbo_batched(model, x, k, kbs):
#     assert k % kbs == 0
#     num_passes = k // kbs
#     ll_batched = []
#     for i in range(num_passes):
#         x_stack = torch.cat([x for _ in range(kbs)], dim=0)
#         ll_stack = model.log_prob(x_stack)
#         ll_batched.append(torch.stack(torch.chunk(ll_stack, kbs, dim=0)))
#     ll = torch.cat(ll_batched, dim=0)
#     return torch.logsumexp(ll, dim=0) - math.log(k)


# def iwbo_nats(model, x, k, kbs=None):
#     """Compute the IWBO in nats."""
#     if kbs: return - iwbo_batched(model, x, k, kbs).mean()
#     else:   return - iwbo(model, x, k).mean()


# def iwbo_bpd(model, x, k, kbs=None):
#     """Compute the IWBO in bits per dim."""
#     if kbs: return - iwbo_batched(model, x, k, kbs).sum() / (x.numel() * math.log(2))
#     else:   return - iwbo(model, x, k).sum() / (x.numel() * math.log(2))


def dataset_elbo_nats(model, data_loader, device, double=False, verbose=True):
    with torch.no_grad():
        nats = 0.0
        count = 0
        for i, x in enumerate(data_loader):
            if double: x = x.double()
            x = x.to(device)
            x = _ensure_full_edges(x)
            nats += elbo_nats(model, x).cpu().item() * len(x)
            count += len(x)
            if verbose: print('{}/{}'.format(i+1, len(data_loader)), nats/count, end='\r')
    return nats / count


def dataset_elbo_bpd(model, data_loader, device, double=False, verbose=True):
    with torch.no_grad():
        bpd = 0.0
        count = 0
        for i, x in enumerate(data_loader):
            if double: x = x.double()
            x = x.to(device)
            x = _ensure_full_edges(x)
            bpd += elbo_bpd(model, x).cpu().item() * len(x)
            count += len(x)
            if verbose: print('{}/{}'.format(i+1, len(data_loader)), bpd/count, end='\r')
    return bpd / count


# def dataset_iwbo_nats(model, data_loader, k, device, double=False, kbs=None, verbose=True):
#     with torch.no_grad():
#         nats = 0.0
#         count = 0
#         for i, x in enumerate(data_loader):
#             if double: x = x.double()
#             x = x.to(device)
#             nats += iwbo_nats(model, x, k=k, kbs=kbs).cpu().item() * len(x)
#             count += len(x)
#             if verbose: print('{}/{}'.format(i+1, len(data_loader)), nats/count, end='\r')
#     return nats / count


# def dataset_iwbo_bpd(model, data_loader, k, device, double=False, kbs=None, verbose=True):
#     with torch.no_grad():
#         bpd = 0.0
#         count = 0
#         for i, x in enumerate(data_loader):
#             if double: x = x.double()
#             x = x.to(device)
#             bpd += iwbo_bpd(model, x, k=k, kbs=kbs).cpu().item() * len(x)
#             count += len(x)
#             if verbose: print('{}/{}'.format(i+1, len(data_loader)), bpd/count, end='\r')
#     return bpd / count
