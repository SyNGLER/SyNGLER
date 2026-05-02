import csv
import json
import os
import pickle

import numpy as np
from tqdm import tqdm

from ForestDiffusion import ForestDiffusionModel as ForestFlowModel


def stack_latent_blocks(blocks):
    arrays = []
    widths = []
    for name, value in blocks:
        array = np.asarray(value)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        arrays.append(array)
        widths.append((name, array.shape[1]))
    return np.hstack(arrays), widths


def split_latent_matrix(matrix: np.ndarray, widths):
    outputs = {}
    start = 0
    for name, width in widths:
        end = start + width
        block = matrix[:, start:end]
        outputs[name] = block if width > 1 else block.reshape(-1)
        start = end
    return outputs


def load_real_result(base_dir: str, r: int, seed: int):
    file_path = os.path.join(base_dir, f"r={r}", f"seed={seed}.pkl")
    with open(file_path, "rb") as handle:
        results = pickle.load(handle)
    Z = np.asarray(results["model_Z"], dtype=np.float32)
    alpha = np.asarray(results["model_alpha"], dtype=np.float32).reshape(-1, 1)
    sparsity = results.get("model_sparsity", 0.0)
    return Z, alpha, sparsity, file_path


def build_forest_model(
    Z: np.ndarray,
    alpha: np.ndarray,
    seed: int,
    n_t: int = 100,
    duplicate_K: int = 100,
    xgb_params=None,
):
    if xgb_params is None:
        xgb_params = {}
    X = np.hstack([Z, alpha])
    y_dummy = np.zeros(X.shape[0], dtype=np.int64)
    model = ForestFlowModel(
        X,
        label_y=y_dummy,
        n_t=n_t,
        duplicate_K=duplicate_K,
        bin_indexes=[],
        cat_indexes=[],
        int_indexes=[],
        diffusion_type="vp",
        n_jobs=-1,
        seed=int(seed),
        **xgb_params,
    )
    return model, X


def build_forest_model_from_matrix(
    X: np.ndarray,
    seed: int,
    n_t: int = 100,
    duplicate_K: int = 100,
    xgb_params=None,
):
    if xgb_params is None:
        xgb_params = {}
    y_dummy = np.zeros(X.shape[0], dtype=np.int64)
    model = ForestFlowModel(
        X,
        label_y=y_dummy,
        n_t=n_t,
        duplicate_K=duplicate_K,
        bin_indexes=[],
        cat_indexes=[],
        int_indexes=[],
        diffusion_type="vp",
        n_jobs=-1,
        seed=int(seed),
        **xgb_params,
    )
    return model


def generate_latent_replicates(model, X: np.ndarray, r: int, out_dir: str, reps: int = 200):
    os.makedirs(out_dir, exist_ok=True)
    n = X.shape[0]
    for rep in range(reps):
        Xy_fake = model.generate(batch_size=n)
        x_fake = Xy_fake[:, :-1]
        Z_fake = x_fake[:, :r]
        alpha_fake = x_fake[:, r : r + 1]
        np.savez(os.path.join(out_dir, f"rep{rep}.npz"), Z=Z_fake, alpha=alpha_fake)


def generate_matrix_replicates(model, X: np.ndarray, out_dir: str, reps: int = 200):
    os.makedirs(out_dir, exist_ok=True)
    n = X.shape[0]
    outputs = []
    for _ in range(reps):
        Xy_fake = model.generate(batch_size=n)
        outputs.append(Xy_fake[:, :-1])
    return outputs


def process_real_dataset(
    dataset: str,
    r: int,
    seed: int,
    data_root: str,
    out_root: str,
    reps: int,
    xgb_params,
    model_cfg,
):
    input_base = os.path.join(data_root, dataset, "run")
    try:
        Z, alpha, sparsity, src_path = load_real_result(input_base, r, seed)
    except FileNotFoundError:
        tqdm.write(f"[WARN] missing input: dataset={dataset}, r={r}, seed={seed}, base_dir={input_base}")
        return False

    model, X = build_forest_model(
        Z,
        alpha,
        seed,
        n_t=model_cfg.get("n_t", 100),
        duplicate_K=model_cfg.get("duplicate_K", 100),
        xgb_params=xgb_params,
    )

    out_dir = os.path.join(out_root, dataset, "Diff-sample", f"r={r}", f"seed={seed}")
    os.makedirs(out_dir, exist_ok=True)

    meta = {
        "dataset": dataset,
        "r": r,
        "seed": seed,
        "n": int(X.shape[0]),
        "reps": reps,
        "src_path": src_path,
        "sparsity": sparsity,
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, ensure_ascii=False)

    generate_latent_replicates(model, X, r, out_dir, reps=reps)
    tqdm.write(f"[OK] dataset={dataset}, r={r} -> {out_dir} (reps={reps})")
    return True


def load_simulation_matrix(gen_base: str, n: int, r: int, sparse_level: float, seed: int):
    run_dir = os.path.join(gen_base, f"n={n}_r={r}_sparse={sparse_level}")
    file_path = os.path.join(run_dir, f"seed={seed}.pkl")
    with open(file_path, "rb") as handle:
        results = pickle.load(handle)
    Z = np.asarray(results["model_Z"])
    alpha = np.asarray(results["model_alpha"]).reshape(-1, 1)
    return np.hstack([Z, alpha])


def run_simulation_grid(
    grid_csv: str,
    gen_base: str,
    save_base: str,
    n_t: int,
    duplicate_K: int,
    samples_per_data: int,
    xgb_params,
):
    jobs = []
    with open(grid_csv, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            jobs.append(
                (
                    int(row["n"]),
                    int(row["r"]),
                    int(row["seed"]),
                    float(row.get("sparse_level", 0.0)),
                )
            )

    for n, r, seed, sparse_level in tqdm(jobs, desc="Datasets", unit="job"):
        X = load_simulation_matrix(gen_base, n, r, sparse_level, seed)
        y_dummy = np.zeros(X.shape[0])
        forest_model = ForestFlowModel(
            X,
            label_y=y_dummy,
            n_t=n_t,
            duplicate_K=duplicate_K,
            bin_indexes=[],
            cat_indexes=[],
            int_indexes=[],
            diffusion_type="vp",
            seed=seed,
            **xgb_params,
        )

        out_dir = os.path.join(save_base, f"n={n}_r={r}_sparse={sparse_level}", f"seed={seed}")
        os.makedirs(out_dir, exist_ok=True)

        Xy_fake_all = forest_model.generate(batch_size=X.shape[0] * samples_per_data)
        for rep in range(samples_per_data):
            start = rep * X.shape[0]
            end = (rep + 1) * X.shape[0]
            Xy_fake = Xy_fake_all[start:end, :]
            x_fake = Xy_fake[:, :-1]
            Z_fake = x_fake[:, :r]
            alpha_fake = x_fake[:, r : r + 1]
            np.savez(os.path.join(out_dir, f"rep{rep}.npz"), Z=Z_fake, alpha=alpha_fake)

        tqdm.write(f"[n={n}, r={r}, seed={seed}] saved {samples_per_data} reps to: {out_dir}")
