import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from SyNGLER.utils.resampling import run_simulation_bootstrap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="../../datasets/simulation/run")
    parser.add_argument("--out-root", default="../../synthetic/simulation/Res-sample")
    parser.add_argument("--n-values", type=int, nargs="+", default=[500, 1000, 1500])
    parser.add_argument("--r-values", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--sparse-level", type=float, default=0.0)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-end", type=int, default=200)
    parser.add_argument("--reps", type=int, default=200)
    args = parser.parse_args()

    run_simulation_bootstrap(
        args.data_root,
        args.out_root,
        args.n_values,
        args.r_values,
        args.sparse_level,
        range(args.seed_start, args.seed_end),
        range(args.reps),
    )
    print("\nBootstrap process completed.")


if __name__ == "__main__":
    main()
