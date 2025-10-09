import numpy as np
import argparse
from diffusion.utils import add_parent_path, set_seeds


# Data
add_parent_path(level=1)
from datasets.data import get_data, get_data_id, add_data_args

# Exp
from experiment import GraphExperiment, add_exp_args

# Model
from model import get_model, get_model_id, add_model_args

# Optim
from diffusion.optim.multistep import get_optim, get_optim_id, add_optim_args

###########
## Setup ##
###########


parser = argparse.ArgumentParser()
add_data_args(parser)
add_exp_args(parser)
add_model_args(parser)
add_optim_args(parser)
args = parser.parse_args()
set_seeds(args.seed)

##################
## Specify data ##
##################

train_loader, eval_loader, test_loader, num_node_feat, num_node_classes, num_edge_classes, max_degree, augmented_feature_dict, initial_graph_sampler, eval_evaluator, test_evaluator, monitoring_statistics = get_data(args)

args.num_edge_classes = num_edge_classes
args.num_node_classes = num_node_classes

if args.final_prob_node is None:
    args.final_prob_node = [1-1e-12, 1e-12]
    args.num_node_classes = 2
    args.has_node_feature = False

if 0 in args.final_prob_edge:
    args.final_prob_edge[np.argmax(args.final_prob_edge)] = args.final_prob_edge[np.argmax(args.final_prob_edge)]-1e-12
    args.final_prob_edge[np.argmin(args.final_prob_edge)] = 1e-12

args.max_degree = max_degree
args.num_node_feat = num_node_feat
args.augmented_feature_dict = augmented_feature_dict



data_id = get_data_id(args)
###################
## Specify model ##
###################

model = get_model(args, initial_graph_sampler=initial_graph_sampler)
print(model)
model_id = get_model_id(args)
#######################
## Specify optimizer ##
#######################

optimizer, scheduler_iter, scheduler_epoch = get_optim(args, model)
optim_id = get_optim_id(args)

##############
## Training ##
##############
exp = GraphExperiment(args=args,
                 data_id=data_id,
                 model_id=model_id,
                 optim_id=optim_id,
                 train_loader=train_loader,
                 eval_loader=eval_loader,
                 test_loader=test_loader,
                 model=model,
                 optimizer=optimizer,
                 scheduler_iter=scheduler_iter,
                 scheduler_epoch=scheduler_epoch,
                 monitoring_statistics=monitoring_statistics,
                 eval_evaluator=eval_evaluator, 
                 test_evaluator=test_evaluator,
                 n_patient=50)

import time, json, os
import torch
out_dir = os.path.join("/data2/network/edge/wandb/snap/multinomial_diffusion/multistep/",args.name)
os.makedirs(out_dir, exist_ok=True)

# === FLOPs: training forward (one step) ===
from edge_flops_count import FlopsCounter, fmt_flops
import json

# 取一个代表性 batch（不回传梯度）
model.eval()
one_batch = next(iter(train_loader))
if isinstance(one_batch, list) or isinstance(one_batch, tuple):
    pyg_data = one_batch[0]
else:
    pyg_data = one_batch
pyg_data = pyg_data.to(args.device)

counter = FlopsCounter()
with torch.inference_mode(), counter.profile(model_for_hooks=model):
    # elbo_bpd 内部会做前向多步扩散的计算，我们仅统计 forward（不 backward）
    from diffusion.loss import elbo_bpd
    _loss = elbo_bpd(model, pyg_data)

fwd_flops_train_step = int(counter.flops)
alpha = 2.5  # 训练/推理 FLOPs 比例（中档）；也给 low/high 两档
train_step_low  = int(2.0 * fwd_flops_train_step)
train_step_mid  = int(alpha * fwd_flops_train_step)
train_step_high = int(3.0 * fwd_flops_train_step)

# 估计每个 epoch 的迭代数（按 DataLoader）
iters_per_epoch = len(train_loader)
epochs = args.epochs

per_epoch_low  = train_step_low  * iters_per_epoch
per_epoch_mid  = train_step_mid  * iters_per_epoch
per_epoch_high = train_step_high * iters_per_epoch

total_low  = per_epoch_low  * epochs
total_mid  = per_epoch_mid  * epochs
total_high = per_epoch_high * epochs

print("\n[EDGE FLOPs] Training-forward per step:", fmt_flops(fwd_flops_train_step))
print("[EDGE FLOPs] Train step low/mid/high:",
      fmt_flops(train_step_low), "/", fmt_flops(train_step_mid), "/", fmt_flops(train_step_high))
print("[EDGE FLOPs] Per-epoch mid:", fmt_flops(per_epoch_mid), "  | Total mid:", fmt_flops(total_mid))

# 先把训练 FLOPs 写文件（立刻可见）
with open(os.path.join(out_dir, "flops_edge_train.json"), "w") as f:
    json.dump({
        "mode": "training",
        "forward_flops_per_step": fwd_flops_train_step,
        "train_step_low":  train_step_low,
        "train_step_mid":  train_step_mid,
        "train_step_high": train_step_high,
        "iters_per_epoch": iters_per_epoch,
        "epochs": epochs,
        "per_epoch_low":  per_epoch_low,
        "per_epoch_mid":  per_epoch_mid,
        "per_epoch_high": per_epoch_high,
        "total_low":  total_low,
        "total_mid":  total_mid,
        "total_high": total_high,
        "alpha_used": alpha
    }, f, indent=2)

t0 = time.perf_counter()
exp.run()
if torch.cuda.is_available():
    torch.cuda.synchronize()  # 确保GPU任务完成
total_sec = time.perf_counter() - t0
print(f"[TIME] Total training wall time: {total_sec:.6f}s")


with open(os.path.join(out_dir, "train_time.json"), "w") as f:
    json.dump({"total_seconds": total_sec}, f, indent=2)

# === FLOPs: sampling forward (one batch) ===
model.eval()
counter = FlopsCounter()
with torch.inference_mode(), counter.profile(model_for_hooks=model):
    _samples = model.sample(args.num_generation)   # 一次生成 num_generation 张

fwd_flops_sampling_batch = int(counter.flops)
per_graph_fwd = int(fwd_flops_sampling_batch // max(1, args.num_generation))

print("[EDGE FLOPs] Sampling-forward per batch:", fmt_flops(fwd_flops_sampling_batch),
      " | per-sample:", fmt_flops(per_graph_fwd))

with open(os.path.join(out_dir, "flops_edge_sampling.json"), "w") as f:
    json.dump({
        "mode": "sampling",
        "num_generation": int(args.num_generation),
        "forward_flops_total_batch": fwd_flops_sampling_batch,
        "forward_flops_per_graph": per_graph_fwd
    }, f, indent=2)
