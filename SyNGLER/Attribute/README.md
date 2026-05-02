# Attribute Network Pipeline

This directory contains the attributed-network workflow for the Cora dataset.
It covers:

- downloading and preprocessing Cora
- latent inference for the joint network/attribute model
- reconstruction and link-prediction evaluation
- score-based latent resampling

## Files

- `prepare_cora.py`: download Cora, extract the largest connected component, run latent inference, and save a processed `.npz`
- `latent_inference.py`: core inference utilities for the attributed latent model
- `run_cora.py`: evaluate reconstruction quality and optional link prediction on the processed Cora data
- `resample_latents_bootstrap.py`: bootstrap resampling for inferred attributed-network latents
- `resample_latents_diffusion.py`: forest-diffusion resampling for inferred attributed-network latents
- `resample_latents.py`: train the score model on inferred latent factors and generate resampled latent draws

## Quick Start

From the repository root:

```bash
cd SyNGLER/Attribute
python prepare_cora.py
python run_cora.py
python resample_latents_bootstrap.py
python resample_latents_diffusion.py
python resample_latents.py
```

## Default Paths

- Raw Cora source: `datasets/cora/source/`
- Processed data: `datasets/cora/generator/cora.npz`
- Evaluation results: `datasets/cora/run/cora_inference_results.npz`
- Bootstrap resamples: `datasets/cora/run/resamples_bootstrap/`
- Diffusion resamples: `datasets/cora/run/resamples_diffusion/`
- Resampled latents: `datasets/cora/run/resamples/`

## Notes

- The processed `.npz` stores the sparse adjacency matrix, Cora features and labels, and inferred latent quantities needed for reconstruction.
