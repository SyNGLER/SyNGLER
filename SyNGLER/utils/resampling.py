import os
import pickle

import numpy as np

from SyNGLER.utils.SyNG_source import bootstrap_alpha_Z


def bootstrap_feature_matrix(X: np.ndarray, batch: int = 1):
    X = np.asarray(X)
    n, d = X.shape
    out = np.zeros((batch, n, d), dtype=X.dtype)
    for b in range(batch):
        indices = np.random.choice(n, size=n, replace=True)
        out[b] = X[indices]
    return out


def save_bootstrap_replicates(model_alpha, model_Z, out_dir: str, rep_range, seed_offset: int):
    os.makedirs(out_dir, exist_ok=True)
    for rep in rep_range:
        np.random.seed(seed_offset + rep)
        alpha_bootstrap, Z_bootstrap = bootstrap_alpha_Z(model_alpha, model_Z, batch=1)
        Z_processed = Z_bootstrap.squeeze()
        alpha_processed = alpha_bootstrap.squeeze()
        output_file_path = os.path.join(out_dir, f"rep{rep}.npz")
        np.savez(output_file_path, alpha=alpha_processed, Z=Z_processed)


def run_real_bootstrap(dataset: str, data_root: str, out_root: str, r_range, rep_range, seed: int):
    for r in r_range:
        input_base_path = os.path.join(data_root, dataset, "run", f"r={r}")
        output_base_path = os.path.join(out_root, dataset, "Res-sample", f"r={r}")
        input_file_path = os.path.join(input_base_path, f"seed={seed}.pkl")
        seed_output_path = os.path.join(output_base_path, f"seed={seed}")

        try:
            with open(input_file_path, "rb") as handle:
                result = pickle.load(handle)

            model_alpha = np.array(result["model_alpha"]).reshape(-1, 1)
            model_Z = np.array(result["model_Z"])
            print(f"Processing dataset={dataset}, seed={seed}, alpha={model_alpha.shape}, Z={model_Z.shape}")
            save_bootstrap_replicates(model_alpha, model_Z, seed_output_path, rep_range, seed + 10000)
        except FileNotFoundError:
            print(f"Warning: File not found at {input_file_path}. Skipping.")
        except KeyError:
            print(f"Error: Missing 'model_Z' or 'model_alpha' in {input_file_path}. Skipping.")
        except Exception as exc:
            print(f"Unexpected error while processing {input_file_path}: {exc}. Skipping.")


def run_simulation_bootstrap(
    data_root: str,
    out_root: str,
    n_range,
    r_range,
    sparse_level: float,
    seed_range,
    rep_range,
):
    for n in n_range:
        for r in r_range:
            input_base_path = os.path.join(data_root, f"n={n}_r={r}_sparse={sparse_level}")
            output_base_path = os.path.join(out_root, f"n={n}_r={r}_sparse={sparse_level}")

            for seed in seed_range:
                input_file_path = os.path.join(input_base_path, f"seed={seed}.pkl")
                seed_output_path = os.path.join(output_base_path, f"seed={seed}")

                try:
                    with open(input_file_path, "rb") as handle:
                        result = pickle.load(handle)

                    model_alpha = np.array(result["model_alpha"]).reshape(-1, 1)
                    model_Z = np.array(result["model_Z"])
                    print(f"Processing seed={seed}, alpha={model_alpha.shape}, Z={model_Z.shape}")
                    save_bootstrap_replicates(model_alpha, model_Z, seed_output_path, rep_range, seed)
                except FileNotFoundError:
                    print(f"Warning: File not found at {input_file_path}. Skipping this seed.")
                except KeyError:
                    print(f"Error: Missing 'model_Z' or 'model_alpha' in {input_file_path}. Skipping.")
                except Exception as exc:
                    print(f"Unexpected error while processing {input_file_path}: {exc}. Skipping.")
