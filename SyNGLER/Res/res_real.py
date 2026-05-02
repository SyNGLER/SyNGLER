import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from SyNGLER.utils.resampling import run_real_bootstrap

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["dblp", "youtube", "yelp", "polblogs"])
    parser.add_argument("--data-root", default="../../datasets")
    parser.add_argument("--out-root", default="../../synthetic")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--r-values", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    parser.add_argument("--reps", type=int, default=2)
    args = parser.parse_args()

    run_real_bootstrap(
        args.dataset,
        args.data_root,
        args.out_root,
        args.r_values,
        range(args.reps),
        args.seed,
    )
    print("\nBootstrap process completed.")

if __name__ == "__main__":
    main()
