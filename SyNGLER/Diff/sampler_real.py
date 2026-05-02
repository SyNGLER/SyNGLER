import os
import argparse
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from SyNGLER.utils.diffusion import process_real_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["dblp", "youtube", "yelp", "polblogs"])
    parser.add_argument("--data-root", default="../../datasets")
    parser.add_argument("--out-root", default="../../synthetic")
    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_depth", type=int, default=7)
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--eta", type=float, default=0.3)
    parser.add_argument("--tree_method", type=str, default="hist")
    parser.add_argument("--reg_lambda", type=float, default=0.0)
    parser.add_argument("--reg_alpha", type=float, default=0.0)
    parser.add_argument("--subsample", type=float, default=1.0)
    parser.add_argument("--n_t", type=int, default=100)
    parser.add_argument("--duplicate_K", type=int, default=100)
    args = parser.parse_args()

    xgb_params = dict(
        max_depth=args.max_depth,
        n_estimators=args.n_estimators,
        eta=args.eta,
        tree_method=args.tree_method,
        reg_lambda=args.reg_lambda,
        reg_alpha=args.reg_alpha,
        subsample=args.subsample,
    )
    model_cfg = dict(n_t=args.n_t, duplicate_K=args.duplicate_K)

    ok = 0
    total = 0
    for r in [2]:
        total += 1
        ok += bool(
            process_real_dataset(
                args.dataset,
                r,
                args.seed,
                args.data_root,
                args.out_root,
                args.reps,
                xgb_params,
                model_cfg,
            )
        )

    print(f"Complete {ok}/{total} r-values for dataset={args.dataset}.")

if __name__ == "__main__":
    main()
