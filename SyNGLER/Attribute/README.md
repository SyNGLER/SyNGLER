# Attribute Network Pipeline

This directory contains the attributed-network workflow for the Cora dataset.
It covers:

- downloading and preprocessing Cora
- LSM-based network latent inference with PGD optimization
- attribute-side latent inference from saved LSM fits
- reconstruction and link-prediction evaluation
- score-based latent resampling

## Files

- `prepare_cora.py`: download Cora, extract the largest connected component, run latent inference, and save a processed `.npz`
- `lsm_backend.py`: LSM backend used by the attributed-network inference path
- `lsm_inference.py`: Cora-facing wrapper around the LSM/PGD inference workflow
- `attribute_inference.py`: build `cora_{r}.npz` files by combining saved LSM fits with attribute-side factor inference
- `run_cora.py`: evaluate reconstruction quality and optional link prediction on the processed Cora data
- `resample_latents_bootstrap.py`: bootstrap resampling for inferred attributed-network latents
- `resample_latents_diffusion.py`: forest-diffusion resampling for inferred attributed-network latents
- `resample_latents.py`: train the score model on inferred latent factors and generate resampled latent draws

## Quick Start

From the repository root:

```bash
cd SyNGLER/Attribute
python prepare_cora.py
python attribute_inference.py --r 5
python run_cora.py
python resample_latents_bootstrap.py --r 5
python resample_latents_diffusion.py --r 5
python resample_latents.py --r 5
```

## Default Paths

- Raw Cora source: `datasets/cora/source/`
- Processed data: `datasets/cora/generator/cora.npz`
- Saved LSM runs: `datasets/cora/lsm/r=*/cora.pkl`
- Attribute inference outputs: `datasets/cora/generator/cora_{r}.npz`
- Evaluation results: `datasets/cora/run/cora_inference_results.npz`
- Bootstrap resamples: `datasets/cora/run/resamples_bootstrap/`
- Diffusion resamples: `datasets/cora/run/resamples_diffusion/`
- Resampled latents: `datasets/cora/run/resamples/`

## Notes

- `prepare_cora.py` writes the shared graph/features bundle to `datasets/cora/generator/cora.npz`.
- `attribute_inference.py` reads `datasets/cora/lsm/r=*/cora.pkl` and writes `datasets/cora/generator/cora_{r}.npz`.
- The three resampling scripts read `datasets/cora/generator/cora_{r}.npz` by default and write outputs under `datasets/cora/run/`.
- The current inference path is LSM/PGD-based and does not use the older spectral `latent_inference.py` pipeline.
