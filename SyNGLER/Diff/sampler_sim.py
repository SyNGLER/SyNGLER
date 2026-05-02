import argparse
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from SyNGLER.utils.diffusion import run_simulation_grid

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid_csv", type=str, required=True, help="Path to CSV file containing n,r,seed,sparse_level")
    parser.add_argument("--gen-base", default="../../datasets/run")
    parser.add_argument("--save-base", default="../../synthetic/simulation/Diff-sample")
    parser.add_argument("--n_t", type=int, default=50)
    parser.add_argument("--duplicate_K", type=int, default=100)
    parser.add_argument("--samples_per_data", type=int, default=200)
    parser.add_argument("--max_depth", type=int, default=7)
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--eta", type=float, default=0.3)
    parser.add_argument("--tree_method", type=str, default="hist")
    parser.add_argument("--reg_lambda", type=float, default=0.0)
    parser.add_argument("--reg_alpha", type=float, default=0.0)
    parser.add_argument("--subsample", type=float, default=1.0)
    args = parser.parse_args()
    xgb_params = dict(
        max_depth=args.max_depth,
        n_estimators=args.n_estimators,
        eta=args.eta,
        tree_method=args.tree_method,
        reg_lambda=args.reg_lambda,
        reg_alpha=args.reg_alpha,
        subsample=args.subsample,
        n_jobs=-1,
    )
    run_simulation_grid(
        args.grid_csv,
        args.gen_base,
        args.save_base,
        args.n_t,
        args.duplicate_K,
        args.samples_per_data,
        xgb_params,
    )
