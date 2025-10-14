###############################################################################
#
# Some code is adapted from https://github.com/JiaxuanYou/graph-generation
#
###############################################################################
import os, glob, re, pickle
import torch
import pickle
import numpy as np
from scipy import sparse as sp
import networkx as nx
import torch.nn.functional as F

__all__ = [
    'save_graph_list', 'load_graph_list', 'graph_load_batch',
    'preprocess_graph_list', 'create_graphs'
]


# save a list of graphs
def save_graph_list(G_list, fname):
  with open(fname, "wb") as f:
    pickle.dump(G_list, f)


def pick_connected_component_new(G):
  # import pdb; pdb.set_trace()

  # adj_list = G.adjacency_list()
  # for id,adj in enumerate(adj_list):
  #     id_min = min(adj)
  #     if id<id_min and id>=1:
  #     # if id<id_min and id>=4:
  #         break
  # node_list = list(range(id)) # only include node prior than node "id"

  adj_dict = nx.to_dict_of_lists(G)
  for node_id in sorted(adj_dict.keys()):
    id_min = min(adj_dict[node_id])
    if node_id < id_min and node_id >= 1:
      # if node_id<id_min and node_id>=4:
      break
  node_list = list(
      range(node_id))  # only include node prior than node "node_id"

  G = G.subgraph(node_list).copy()
  largest_cc = max(nx.connected_components(G), key=len)
  G = G.subgraph(largest_cc).copy()



def load_graph_list(fname, is_real=True):
  with open(fname, "rb") as f:
    graph_list = pickle.load(f)

  # import pdb; pdb.set_trace()
  for i in range(len(graph_list)):
    edges_with_selfloops = list(graph_list[i].selfloop_edges())
    if len(edges_with_selfloops) > 0:
      graph_list[i].remove_edges_from(edges_with_selfloops)
    if is_real:
      largest_cc = max(nx.connected_components(graph_list[i]), key=len)
      graph_list[i] = graph_list[i].subgraph(largest_cc).copy()
      graph_list[i] = nx.convert_node_labels_to_integers(graph_list[i])

    else:
      graph_list[i] = pick_connected_component_new(graph_list[i])
  return graph_list


def preprocess_graph_list(graph_list):
  for i in range(len(graph_list)):
    edges_with_selfloops = list(graph_list[i].selfloop_edges())
    if len(edges_with_selfloops) > 0:
      graph_list[i].remove_edges_from(edges_with_selfloops)
    if is_real:
      graph_list[i] = max(
          nx.connected_component_subgraphs(graph_list[i]), key=len)
      graph_list[i] = nx.convert_node_labels_to_integers(graph_list[i])
    else:
      graph_list[i] = pick_connected_component_new(graph_list[i])
  return graph_list


def graph_load_batch(data_dir,
                     min_num_nodes=20,
                     max_num_nodes=1000,
                     name='ENZYMES',
                     node_attributes=True,
                     graph_labels=True):
  '''
    load many graphs, e.g. enzymes
    :return: a list of graphs
    '''
  print('Loading graph dataset: ' + str(name))
  G = nx.Graph()
  # load data
  path = os.path.join(data_dir, name)
  data_adj = np.loadtxt(
      os.path.join(path, '{}_A.txt'.format(name)), delimiter=',').astype(int)
  if node_attributes:
    data_node_att = np.loadtxt(
        os.path.join(path, '{}_node_attributes.txt'.format(name)),
        delimiter=',')
  data_node_label = np.loadtxt(
      os.path.join(path, '{}_node_labels.txt'.format(name)),
      delimiter=',').astype(int)
  data_graph_indicator = np.loadtxt(
      os.path.join(path, '{}_graph_indicator.txt'.format(name)),
      delimiter=',').astype(int)
  if graph_labels:
    data_graph_labels = np.loadtxt(
        os.path.join(path, '{}_graph_labels.txt'.format(name)),
        delimiter=',').astype(int)

  data_tuple = list(map(tuple, data_adj))
  # print(len(data_tuple))
  # print(data_tuple[0])

  # add edges
  G.add_edges_from(data_tuple)
  # add node attributes
  for i in range(data_node_label.shape[0]):
    if node_attributes:
      G.add_node(i + 1, feature=data_node_att[i])
    G.add_node(i + 1, label=data_node_label[i])
  G.remove_nodes_from(list(nx.isolates(G)))

  # remove self-loop
  G.remove_edges_from(nx.selfloop_edges(G))

  # print(G.number_of_nodes())
  # print(G.number_of_edges())

  # split into graphs
  graph_num = data_graph_indicator.max()
  node_list = np.arange(data_graph_indicator.shape[0]) + 1
  graphs = []
  max_nodes = 0
  for i in range(graph_num):
    # find the nodes for each graph
    nodes = node_list[data_graph_indicator == i + 1]
    G_sub = G.subgraph(nodes)
    if graph_labels:
      G_sub.graph['label'] = data_graph_labels[i]
    # print('nodes', G_sub.number_of_nodes())
    # print('edges', G_sub.number_of_edges())
    # print('label', G_sub.graph)
    if G_sub.number_of_nodes() >= min_num_nodes and G_sub.number_of_nodes(
    ) <= max_num_nodes:
      graphs.append(G_sub)
      if G_sub.number_of_nodes() > max_nodes:
        max_nodes = G_sub.number_of_nodes()
      # print(G_sub.number_of_nodes(), 'i', i)
      # print('Graph dataset name: {}, total graph num: {}'.format(name, len(graphs)))
      # logging.warning('Graphs loaded, total num: {}'.format(len(graphs)))
  print('Loaded')
  return graphs


def create_graphs(graph_type, data_dir='data', noise=10.0, seed=1234, **kwargs):
  print(f"[create_graphs] name={kwargs.get('name')} graph_type={graph_type} source_dir={kwargs.get('source_dir')} glob={kwargs.get('filename_glob')}")

  graphs = []
  gt = str(graph_type).upper()

  if gt in ('CUSTOM_PKL', 'CUSTOM'):
    source_dir     = kwargs.get('source_dir', data_dir)
    filename_glob  = kwargs.get('filename_glob', 'seed=*.pkl')
    pattern = filename_glob if os.path.isabs(filename_glob) else os.path.join(source_dir, filename_glob)
    recursive = ('**' in pattern)
    files = [pattern] if os.path.isfile(pattern) else sorted(glob.glob(pattern, recursive=recursive))

    limit          = kwargs.get('limit', None)
    if limit:
      files = files[:int(limit)]

    sample_from_P  = bool(kwargs.get('sample_from_P', True))
    prob_threshold = float(kwargs.get('prob_threshold', 0.5))
    rng_mode       = kwargs.get('rng_mode', 'fixed')   # from_filename | fixed | none
    rng_fixed_seed = int(kwargs.get('rng_fixed_seed', 0))

    if not files:
      raise FileNotFoundError(f"[create_graphs] 未匹配到任何文件：{pattern}")

    for p in files:
      G = _load_graph_from_custom_pkl(
        p,
        sample_from_P=sample_from_P,
        threshold=prob_threshold,
        rng_mode=rng_mode,
        rng_fixed_seed=rng_fixed_seed,
      )
      graphs.append(G)

  elif gt in ('CUSTOM_NPY', 'NPY', 'ADJ_NUMPY', 'ADJ_NPY'):
    source_dir     = kwargs.get('source_dir', data_dir)
    filename_glob  = kwargs.get('filename_glob', 'seed=*.npy')
    pattern = filename_glob if os.path.isabs(filename_glob) else os.path.join(source_dir, filename_glob)
    recursive = ('**' in pattern)
    files = [pattern] if os.path.isfile(pattern) else sorted(glob.glob(pattern, recursive=recursive))

    limit = kwargs.get('limit', None)
    if limit:
      files = files[:int(limit)]

    npy_threshold         = kwargs.get('npy_threshold', None)     # None or float
    npy_binarize          = bool(kwargs.get('npy_binarize', True))
    npy_force_undirected  = bool(kwargs.get('npy_force_undirected', True))
    npy_remove_self_loops = bool(kwargs.get('npy_remove_self_loops', True))
    npy_take_lcc          = bool(kwargs.get('npy_take_lcc', False))

    if not files:
      raise FileNotFoundError(f"[create_graphs]No .npy found{pattern}")

    for fpath in files:
      G = _load_graph_from_custom_npy(
        fpath,
        threshold=npy_threshold,
        binarize=npy_binarize,
        force_undirected=npy_force_undirected,
        remove_self_loops=npy_remove_self_loops,
        take_lcc=npy_take_lcc,
      )
      graphs.append(G)

  elif gt == 'GRID':
    graphs = []
    for i in range(10, 20):
      for j in range(10, 20):
        graphs.append(nx.grid_2d_graph(i, j))
  elif gt == 'LOBSTER':
    graphs = []
    p1 = 0.7
    p2 = 0.7
    count = 0
    min_node = 10
    max_node = 100
    max_edge = 0
    mean_node = 80
    num_graphs = 100

    seed_tmp = seed
    while count < num_graphs:
      G = nx.random_lobster(mean_node, p1, p2, seed=seed_tmp)
      if len(G.nodes()) >= min_node and len(G.nodes()) <= max_node:
        graphs.append(G)
        if G.number_of_edges() > max_edge:
          max_edge = G.number_of_edges()
        count += 1
      seed_tmp += 1
  elif gt == 'DD':
    graphs = graph_load_batch(
        data_dir,
        min_num_nodes=100,
        max_num_nodes=500,
        name='DD',
        node_attributes=False,
        graph_labels=True)
  elif gt == 'FIRSTMM_DB':
    graphs = graph_load_batch(
        data_dir,
        min_num_nodes=0,
        max_num_nodes=10000,
        name='FIRSTMM_DB',
        node_attributes=False,
        graph_labels=True)
  else:
    raise ValueError(f"[create_graphs] 未知 graph_type: {graph_type}")

  if len(graphs) > 0:
    num_nodes = [gg.number_of_nodes() for gg in graphs]
    num_edges = [gg.number_of_edges() for gg in graphs]
    print('max # nodes = {} || mean # nodes = {}'.format(max(num_nodes), np.mean(num_nodes)))
    print('max # edges = {} || mean # edges = {}'.format(max(num_edges), np.mean(num_edges)))
  else:
    print('[create_graphs] Warning: no graph loaded.')

  return graphs



def _get_attr_or_key(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)

def _load_graph_from_custom_pkl(
    path,
    *,
    make_symmetric=True,
    zero_diag=True,
    sample_from_P=True,
    threshold=0.5,
    rng_mode="from_filename",     # 'from_filename' | 'fixed' | 'none'
    rng_fixed_seed=1234,
):
    with open(path, "rb") as f:
        obj = pickle.load(f)

    data = _get_attr_or_key(obj, "data")
    if data is None:
        raise ValueError(f"{path} no data attribute or key")

    P = _get_attr_or_key(data, "P")
    A = _get_attr_or_key(data, "A")

    if sample_from_P:
        if P is None:
            if A is None:
                raise ValueError(f"{path} no data.P or data.A to sample from")
            A_bin = np.asarray(A, dtype=np.uint8)
        else:
            P = np.asarray(P, dtype=np.float32)
            if rng_mode == "from_filename":
                m = re.search(r"seed=(\d+)", os.path.basename(path))
                seed = int(m.group(1)) if m else rng_fixed_seed
            elif rng_mode == "fixed":
                seed = rng_fixed_seed
            else:
                seed = None
            rng = np.random.RandomState(seed) if seed is not None else np.random

            n = P.shape[0]
            tril_mask = np.tril(np.ones((n, n), dtype=bool), k=-1)
            A_bin = np.zeros_like(P, dtype=np.uint8)
            A_bin[tril_mask] = (rng.rand(*P[tril_mask].shape) < P[tril_mask]).astype(np.uint8)
    else:
        if P is None:
            raise ValueError(f"{path} no data.P.")
        P = np.asarray(P, dtype=np.float32)
        n = P.shape[0]
        tril_mask = np.tril(np.ones((n, n), dtype=bool), k=-1)
        A_bin = np.zeros_like(P, dtype=np.uint8)
        A_bin[tril_mask] = (P[tril_mask] >= float(threshold)).astype(np.uint8)

    if make_symmetric:
        A_bin = np.tril(A_bin, -1)
        A_bin = A_bin + A_bin.T
    if zero_diag:
        np.fill_diagonal(A_bin, 0)

    G = nx.from_numpy_array(A_bin)
    return G


def _adj_to_graph(A,
                  threshold=None,
                  binarize=True,
                  force_undirected=True,
                  remove_self_loops=True):
  A = np.asarray(A)
  if threshold is None:
    A = (A != 0).astype(np.uint8) if binarize else A
  else:
    A = (A > float(threshold)).astype(np.uint8) if binarize else (A > float(threshold)).astype(A.dtype)

  A = np.triu(A, 1)
  A = A + A.T
  if remove_self_loops:
    np.fill_diagonal(A, 0)

  if force_undirected:
    G = nx.from_numpy_array(A)
  else:
    G = nx.from_numpy_array(A, create_using=nx.DiGraph)
  return G


def _load_graph_from_custom_npy(path,
                                *,
                                threshold=None,
                                binarize=True,
                                force_undirected=True,
                                remove_self_loops=True,
                                take_lcc=False):
  A = np.load(path)
  G = _adj_to_graph(
    A,
    threshold=threshold,
    binarize=binarize,
    force_undirected=force_undirected,
    remove_self_loops=remove_self_loops,
  )
  if take_lcc and not G.is_directed():
    if G.number_of_nodes() > 0:
      largest_cc = max(nx.connected_components(G), key=len)
      G = G.subgraph(largest_cc).copy()
      G = nx.convert_node_labels_to_integers(G)
  return G