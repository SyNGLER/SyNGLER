import torch
import pickle
import argparse
from diffusion.utils import add_parent_path
import random
import networkx as nx
import torch_geometric as pyg

# Data
add_parent_path(level=1)
from datasets.data import get_data

# Model
from model import get_model

###########
## Setup ##
###########
parser = argparse.ArgumentParser()
parser.add_argument('--run_name', type=str, default='2023-05-29_18-29-35')
parser.add_argument('--dataset', type=str, default='polblogs')
parser.add_argument('--num_samples', type=int, default=8)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--checkpoint', type=int, default=5500)
parser.add_argument('--data_name', type=str, default='snap')

eval_args = parser.parse_args()

torch.manual_seed(eval_args.seed)

log_dir = f'/data2/network/edge/wandb/{eval_args.dataset}/multinomial_diffusion/multistep/{eval_args.run_name}' 
path_args = '{}/args.pickle'.format(log_dir)
path_check = '{}/check/checkpoint_{}.pt'.format(log_dir, eval_args.checkpoint-1)

with open(path_args, 'rb') as f:
    args = pickle.load(f)

args.device = 'cuda:0'
train_loader, eval_loader, test_loader, num_node_feat, num_node_classes, num_edge_classes, max_degree, augmented_feature_dict, initial_graph_sampler, eval_evaluator, test_evaluator, monitoring_statistics = get_data(args)

model = get_model(args, initial_graph_sampler=initial_graph_sampler)
checkpoint = torch.load(path_check, map_location=args.device)
model.load_state_dict(checkpoint['model'])


seed = 0
random.seed(seed); np.random.seed(seed); os.environ["PYTHONHASHSEED"] = str(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

if torch.cuda.is_available():
    model = model.to(args.device)
model.eval()

# sample 
pyg_datas, generated_nxgraphs = [], []
with torch.no_grad():
    for i in range(eval_args.num_samples):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        batched = model.sample(1)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        gen_sec = time.time() - t0
        per_times.append(gen_sec)

        pyg_data = batched.to_data_list()[0]

        G = pyg.utils.to_networkx(pyg_data, to_undirected=True)
        # if G.number_of_nodes() > 0:
        #     try:
        #         largest_cc = max(nx.connected_components(G), key=len)
        #         G = G.subgraph(largest_cc).copy()
        #     except ValueError:
        #         pass

        pyg_datas.append(pyg_data.cpu())
        generated_nxgraphs.append(G)

        del batched
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

out_dir = f"../synthetic/{eval_args.data_name}/edge-sample/ckpt={eval_args.checkpoint}/seed={eval_args.seed}"
os.makedirs(out_dir, exist_ok=True)


summary = []
for i, G in enumerate(generated_nxgraphs):
    A = nx.to_numpy_array(G, dtype=np.uint8)
    np.save(os.path.join(out_dir, f"rep{i}.npy"), A)
    summary.append({
        "idx": i, 
        "n": int(G.number_of_nodes()), 
        "m": int(G.number_of_edges()),
        "gen_sec": float(per_times[i])
        })
