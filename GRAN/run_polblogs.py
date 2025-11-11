import os
import sys
import yaml
import pathlib
import argparse
import subprocess
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base-yaml",
        default="config/gran_polblogs.yaml",
        help="Path to the base YAML config."
    )
    ap.add_argument(
        "--cuda",
        type=str,
        required=True,
        help="Comma-separated visible device ids, e.g. '0' or '2,3'."
    )
    ap.add_argument("--n", type=int, default=1500, help="Upper bound for number of nodes.")
    ap.add_argument("--sparse", type=float, default=0.0, help="sparse_level (passthrough, optional).")
    ap.add_argument("--tau", type=float, default=0.0, help="tau (passthrough, optional).")
    ap.add_argument("--seed-start", type=int, default=0, help="Start seed (inclusive).")
    ap.add_argument("--seed-end", type=int, default=199, help="End seed (inclusive).")
    ap.add_argument(
        "--data-root",
        required=True,
        help=f"Directory that directly contains input .npy files (named 'seed={seed}.npy') and 'lcc.csv'."
    )
    ap.add_argument(
        "--out-root",
        required=True,
        help="Output directory for logs/checkpoints/samples/temp YAML. No extra subdirectories are appended."
    )
    ap.add_argument(
        "--just-test",
        action="store_true",
        help="Generate only without training (requires test.model in base YAML to point to existing weights)."
    )
    ap.add_argument("--max-epoch", type=int, default=None, help="Override train.max_epoch (optional).")
    ap.add_argument("--train-batch-size", type=int, default=None, help="Override train.batch_size (optional).")
    ap.add_argument("--gen-num", type=int, default=None, help="Override test.num_test_gen (optional).")
    ap.add_argument(
        "--hidden-dim",
        type=int,
        default=None,
        help="Override model.hidden_dim (and model.embedding_dim if not set separately)."
    )
    ap.add_argument(
        "--embedding-dim",
        type=int,
        default=None,
        help="Override model.embedding_dim (optional; defaults to hidden_dim if not provided)."
    )
    args = ap.parse_args()

    # Load base YAML
    with open(args.base_yaml, "r") as f:
        base_cfg = yaml.safe_load(f)

    # ---- Model overrides ----
    base_cfg.setdefault("model", {})
    base_cfg["model"]["max_num_nodes"] = max(base_cfg["model"].get("max_num_nodes", 0), args.n)
    if args.hidden_dim is not None:
        base_cfg["model"]["hidden_dim"] = int(args.hidden_dim)
        if args.embedding_dim is None:
            base_cfg["model"]["embedding_dim"] = int(args.hidden_dim)
    if args.embedding_dim is not None:
        base_cfg["model"]["embedding_dim"] = int(args.embedding_dim)

    base_cfg.setdefault("train", {})
    base_cfg["train"]["batch_size"] = base_cfg["train"].get("batch_size", 1)
    if args.max_epoch is not None:
        base_cfg["train"]["max_epoch"] = int(args.max_epoch)
    if args.train_batch_size is not None:
        base_cfg["train"]["batch_size"] = int(args.train_batch_size)

    base_cfg.setdefault("test", {})
    base_cfg["test"]["batch_size"] = 1
    base_cfg["test"]["return_prob"] = True
    base_cfg["test"]["only_gen"] = True
    base_cfg["test"]["is_vis"] = False
    base_cfg["test"]["target_num_nodes"] = args.n  # placeholder; will be replaced by n_lcc per seed
    base_cfg["test"]["run_after_train"] = not args.just_test
    if args.gen_num is not None:
        base_cfg["test"]["num_test_gen"] = int(args.gen_num)

    data_dir = args.data_root.rstrip("/")
    if not os.path.isdir(data_dir):
        print(f"[ERROR] Data directory does not exist: {data_dir}")
        sys.exit(1)

    lcc_csv = os.path.join(data_dir, "lcc.csv")
    if not os.path.isfile(lcc_csv):
        print(f"[ERROR] Missing lcc.csv: {lcc_csv}")
        sys.exit(1)
    df_lcc = pd.read_csv(lcc_csv, header=None, names=["seed", "n_lcc"])
    seed2n = {int(row.seed): int(row.n_lcc) for _, row in df_lcc.iterrows()}

    out_root = args.out_root.rstrip("/")
    pathlib.Path(out_root).mkdir(parents=True, exist_ok=True)
    print(out_root)

    for seed in range(args.seed_start, args.seed_end + 1):
        npy_path = os.path.join(data_dir, f"seed={seed}.npy")
        if not os.path.exists(npy_path):
            print(f"[seed {seed}] SKIP: input file not found: {npy_path}")
            continue
        if seed not in seed2n:
            print(f"[seed {seed}] SKIP: n_lcc not found in lcc.csv for this seed.")
            continue

        n_lcc = seed2n[seed]

        cfg = yaml.safe_load(yaml.dump(base_cfg))
        cfg["seed"] = int(seed)

        cfg.setdefault("dataset", {})
        cfg["dataset"]["name"] = cfg["dataset"].get("name", "custom_npy")
        cfg["dataset"]["source_dir"] = data_dir + "/"  # .npy files are directly under data_root
        cfg["dataset"]["filename_glob"] = f"seed={seed}.npy"
        cfg["dataset"]["format"] = "adj_numpy"
        cfg["dataset"]["is_overwrite_precompute"] = True

        cfg["model"]["max_num_nodes"] = max(cfg["model"].get("max_num_nodes", 0), int(n_lcc))
        cfg["test"]["target_num_nodes"] = int(n_lcc)

        out_dir = out_root

        cfg["run_id"] = f"gran_polblogs_seed{seed}"

        cfg["test"]["gen_out_dir"] = out_dir
        cfg["test"]["gen_prefix"] = "rep"

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        visible = [x for x in args.cuda.split(",") if x.strip() != ""]
        num_devs = len(visible)

        cfg.setdefault("use_gpu", True)
        cfg["device"] = "cuda:0"
        cfg["gpus"] = list(range(num_devs))

        base_train_bs = cfg.get("train", {}).get("batch_size", 1)
        base_test_bs = cfg.get("test", {}).get("batch_size", 1)
        cfg["train"]["batch_size"] = max(base_train_bs, num_devs)
        cfg["test"]["batch_size"] = max(base_test_bs, num_devs)

        tmp_yaml = f"{out_dir}/_auto_gran_polblogs_seed_{seed}.yaml"
        with open(tmp_yaml, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

        cmd = ["python", "run_exp.py", "-c", tmp_yaml]
        if args.just_test:
            cmd.append("-t")

        print(f"\n=== [seed {seed} | n_lcc={n_lcc}] start ===")
        ret = subprocess.run(cmd, env=env)
        if ret.returncode != 0:
            print(f"[seed {seed}] FAILED with code {ret.returncode}.")

    print("\nAll done.")


if __name__ == "__main__":
    sys.exit(main())
