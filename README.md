# SyNGLER: Efficient Synthetic Network Generation via Latent Embedding Reconstruction

This repository contains the implementation of **SyNGLER** (Efficient Synthetic Network Generation via Latent Embedding Reconstruction) along with baseline methods including GRAN, VGAE, and EDGE for graph generation tasks. It also includes an attributed-network pipeline for the Cora dataset with LSM-based network inference, attribute-side latent inference, evaluation, and multiple resampling backends.

## Overview

SyNGLER is a novel approach for generating synthetic networks using latent embedding reconstruction combined with advanced sampling techniques. The repository provides a comprehensive framework for:

- **SyNGLER Method**: Our proposed approach using latent embedding reconstruction with diffusion and resampling techniques
- **Attributed Network Pipeline**: Cora preprocessing, LSM network inference, attribute-side inference, reconstruction, link prediction, and attributed latent resampling
- **Baseline Methods**: GRAN, VGAE, and EDGE implementations for comparison
- **Multiple Datasets**: Support for DBLP, YouTube, Yelp, PolBlogs, and Cora

## Installation

### Dependencies

The repository requires Python 3.8+ and the following main dependencies:

**Core Requirements:**
- PyTorch (1.2.0+)
- NumPy
- SciPy
- NetworkX
- scikit-learn
- tqdm

**Method-specific Requirements:**

For GRAN:
```bash
pip install -r GRAN/requirements.txt
```

For EDGE:
```bash
pip install dgl prettytable scikit-learn tensorboard tensorflow torch-geometric tqdm wandb
```

For VGAE:
```bash
pip install torch networkx scikit-learn scipy
```

For SyNGLER:
```bash
pip install ForestDiffusion  # For diffusion-based sampling
```

For the attributed-network pipeline:
```bash
pip install matplotlib
```

## Usage

### 1. SyNGLER Method

#### Diffusion-based Sampling (Real Datasets)
```bash
cd SyNGLER/Diff
python sampler_real.py --dataset dblp --data-root ../../datasets --out-root ../../synthetic --reps 200
```

#### Resampling Method (Real Datasets)
```bash
cd SyNGLER/Res
python res_real.py --dataset dblp
```

### 2. Attributed Network Pipeline (Cora)

#### Prepare Cora
```bash
cd SyNGLER/Attribute
python prepare_cora.py
```

#### Build Attribute Inference Outputs
```bash
cd SyNGLER/Attribute
python attribute_inference.py --r 5
```

#### Evaluate Inference
```bash
cd SyNGLER/Attribute
python run_cora.py
```

#### Attributed Latent Resampling
```bash
cd SyNGLER/Attribute
python resample_latents_bootstrap.py --r 5
python resample_latents_diffusion.py --r 5
python resample_latents.py --r 5
```

### 3. Baseline Methods

#### GRAN
```bash
cd GRAN
python run_dblp.py --data-root ../datasets/dblp/generator --out-root ../synthetic/dblp/gran --cuda 0
```

#### VGAE
```bash
cd vgae
python real_data_train.py --dataset dblp --data_path ../datasets/dblp/generator/seed=0.npy --output_dir ../synthetic/dblp/vgae
```

#### EDGE
Please refer to the original [EDGE repository](https://github.com/ehoogeboom/multinomial_diffusion) for detailed usage instructions and examples.

### 4. Latent Space Model Training

```bash
cd Latent-Space-Model/simulated_data
python run.py --config ../config/default.json
```

### 5. Evaluation Framework

To evaluate and compare different methods:

```bash
cd synthetic
jupyter notebook evaluation_demo.ipynb
```

The evaluation framework provides comprehensive analysis using multiple network metrics and statistical distances. See `synthetic/EVALUATION_SETUP.md` for detailed usage instructions.

## Repository Structure

```
SyNGLER/
├── SyNGLER/                    # Main SyNGLER implementation
│   ├── Attribute/              # Attributed-network pipeline for Cora
│   │   ├── prepare_cora.py     # Cora download, preprocessing, and latent inference
│   │   ├── attribute_inference.py # Build cora_{r}.npz from saved LSM runs
│   │   ├── run_cora.py         # Reconstruction and link-prediction evaluation
│   │   ├── lsm_backend.py      # LSM backend for Cora inference
│   │   ├── lsm_inference.py    # LSM/PGD inference wrapper for Cora
│   │   ├── resample_latents_bootstrap.py # Bootstrap attributed resampling
│   │   ├── resample_latents_diffusion.py # Forest-diffusion attributed resampling
│   │   └── resample_latents.py # Score-based latent resampling
│   ├── Diff/                   # Diffusion-based sampling
│   │   ├── sampler_real.py     # Real dataset sampling
│   │   └── sampler_sim.py      # Simulated dataset sampling
│   ├── Res/                    # Resampling methods
│   │   ├── res_real.py         # Real dataset resampling
│   │   └── res_sim.py          # Simulated dataset resampling
│   └── utils/                  # Utility functions
│       ├── SyNG_source.py      # Core SyNGLER utilities
│       ├── diffusion.py        # Shared diffusion helpers
│       ├── resampling.py       # Shared bootstrap/resampling helpers
│       └── score_sde.py        # Shared score-based latent generator
├── GRAN/                       # GRAN baseline implementation
│   ├── config/                 # Configuration files
│   ├── model/                  # Model definitions
│   ├── runner/                 # Training runners
│   ├── utils/                  # Utility functions
│   └── run_*.py               # Dataset-specific runners
├── graph-generation-EDGE/      # EDGE baseline implementation
│   ├── diffusion/              # Diffusion model components
│   ├── datasets/               # Data handling
│   ├── eval_utils/             # Evaluation utilities
│   └── train.py               # Training script
├── vgae/                       # VGAE baseline implementation
│   ├── model.py               # VGAE model
│   ├── train*.py              # Training scripts
│   └── input_*.py             # Data input handlers
├── Latent-Space-Model/         # Core latent space model
│   ├── config/                 # Configuration files
│   └── simulated_data/         # Data generation
├── datasets/                   # Dataset storage
│   ├── cora/                   # Cora attributed-network dataset
│   │   ├── source/             # Raw downloaded Cora files
│   │   ├── generator/          # Processed Cora latent/inference data
│   │   ├── lsm/                # Saved network-side LSM fits
│   │   └── run/                # Evaluation and attributed resampling outputs
│   ├── dblp/                   # DBLP dataset
│   ├── youtube/                # YouTube dataset
│   ├── yelp/                   # Yelp dataset
│   └── polblogs/               # PolBlogs dataset
└── synthetic/                  # Generated synthetic data and evaluation
    ├── evaluation/             # Evaluation framework
    │   └── utils.py           # Core evaluation functions
    ├── evaluation_demo.ipynb  # Evaluation demo notebook
    ├── EVALUATION_SETUP.md    # Evaluation setup guide
    └── README.md              # Evaluation documentation
```

## Datasets

The repository supports five real-world datasets:

- **DBLP**: Academic collaboration network
- **YouTube**: Social network data
- **Yelp**: Business review network
- **PolBlogs**: Political blog network
- **Cora**: Citation network with node attributes

Each dataset is stored in `datasets/{dataset_name}/` with:
- `generator/`: Original data files
- `run/`: Processed results for different latent dimensions

For Cora, the repository also stores:
- `source/`: Raw downloaded dataset files
- `generator/cora.npz`: Processed sparse graph, attributes, labels, and network-side LSM outputs
- `generator/cora_{r}.npz`: Attribute inference outputs built from saved `datasets/cora/lsm/r=*/cora.pkl` runs
- `lsm/r=*/cora.pkl`: Saved LSM fits used by the attribute-side inference step
- `run/resamples_bootstrap/`, `run/resamples_diffusion/`, `run/resamples/`: Attributed latent resampling outputs



## Results

Generated synthetic graphs are saved in the `synthetic/` directory with the following structure:
```
synthetic/
├── {dataset}/
│   ├── Diff-sample/           # SyNGLER diffusion samples
│   ├── Res-sample/            # SyNGLER resampling samples
│   ├── gran-sample/           # GRAN samples
│   ├── vgae-sample/           # VGAE samples
│   └── edge-sample/           # EDGE samples
```

## Evaluation Framework

We provide a comprehensive evaluation framework for systematic comparison of different network generation methods. The evaluation framework supports multiple baselines and datasets with detailed analysis through various network metrics and statistical distances.

### Quick Start

```bash
cd synthetic
jupyter notebook evaluation_demo.ipynb
```

### Supported Baselines

- **SyNGLER-Diff**: Diffusion-based generation
- **SyNGLER-Res**: Residual-based generation  
- **GRAN**: Graph Recurrent Attention Networks
- **EDGE**: Edge-based generation
- **VGAE**: Variational Graph Auto-Encoders
- **ER**: Erdos-Renyi random graphs

### Evaluation Metrics

1. **Triangle Density** - Measures clustering in the network
2. **Global Clustering Coefficient** - Measures overall transitivity
3. **Degree Centrality Energy Distance** - Measures degree distribution preservation
4. **Eigenvalues Energy Distance** - Measures spectral properties preservation

### File Structure

```
synthetic/
├── evaluation/
│   └── utils.py                    # Core evaluation functions
├── evaluation_demo.ipynb          # Full evaluation demo
├── EVALUATION_SETUP.md           # Setup and usage guide
└── README.md                     # This file
```

For detailed evaluation documentation, see `synthetic/EVALUATION_SETUP.md`.


## Acknowledgments

This repository builds upon several existing implementations:
- [GRAN](https://github.com/lrjconan/GRAN) for graph recurrent attention networks
- [EDGE](https://github.com/ehoogeboom/multinomial_diffusion) for discrete diffusion modeling
- [VGAE](https://github.com/tkipf/gae) for variational graph auto-encoders
- [ForestDiffusion](https://github.com/forest-diffusion/ForestDiffusion) for diffusion-based sampling

## Contact

For questions and support, please open an issue in this repository.
