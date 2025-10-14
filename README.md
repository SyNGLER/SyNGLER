# SyNGLER: Efficient Synthetic Network Generation via Latent Embedding Reconstruction

This repository contains the implementation of **SyNGLER** (Efficient Synthetic Network Generation via Latent Embedding Reconstruction) along with baseline methods including GRAN, VGAE, and EDGE for graph generation tasks.

## Overview

SyNGLER is a novel approach for generating synthetic networks using latent embedding reconstruction combined with advanced sampling techniques. The repository provides a comprehensive framework for:

- **SyNGLER Method**: Our proposed approach using latent embedding reconstruction with diffusion and resampling techniques
- **Baseline Methods**: GRAN, VGAE, and EDGE implementations for comparison
- **Multiple Datasets**: Support for DBLP, YouTube, Yelp, and PolBlogs datasets

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

### 2. Baseline Methods

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

### 3. Latent Space Model Training

```bash
cd Latent-Space-Model/simulated_data
python run.py --config ../config/default.json
```

## Repository Structure

```
SyNGLER/
├── SyNGLER/                    # Main SyNGLER implementation
│   ├── Diff/                   # Diffusion-based sampling
│   │   ├── sampler_real.py     # Real dataset sampling
│   │   └── sampler_sim.py      # Simulated dataset sampling
│   ├── Res/                    # Resampling methods
│   │   ├── res_real.py         # Real dataset resampling
│   │   └── res_sim.py          # Simulated dataset resampling
│   └── utils/                  # Utility functions
│       └── SyNG_source.py      # Core SyNGLER utilities
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
│   ├── dblp/                   # DBLP dataset
│   ├── youtube/                # YouTube dataset
│   ├── yelp/                   # Yelp dataset
│   └── polblogs/               # PolBlogs dataset
└── synthetic/                  # Generated synthetic data
    └── evaluation/             # Evaluation results
```

## Datasets

The repository supports four real-world datasets:

- **DBLP**: Academic collaboration network
- **YouTube**: Social network data
- **Yelp**: Business review network
- **PolBlogs**: Political blog network

Each dataset is stored in `datasets/{dataset_name}/` with:
- `generator/`: Original data files
- `run/`: Processed results for different latent dimensions



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

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{syngler2024,
  title={Efficient Synthetic Network Generation via Latent Embedding Reconstruction},
  author={[Authors]},
  booktitle={OpenReview},
  year={2024},
  url={https://openreview.net/forum?id=JtL7kCe32S}
}
```

**Paper Link**: https://openreview.net/forum?id=JtL7kCe32S

## Acknowledgments

This repository builds upon several existing implementations:
- [GRAN](https://github.com/lrjconan/GRAN) for graph recurrent attention networks
- [EDGE](https://github.com/ehoogeboom/multinomial_diffusion) for discrete diffusion modeling
- [VGAE](https://github.com/tkipf/gae) for variational graph auto-encoders
- [ForestDiffusion](https://github.com/forest-diffusion/ForestDiffusion) for diffusion-based sampling

## License

This project is licensed under the MIT License - see the individual method directories for specific license information.

## Contact

For questions and support, please open an issue in this repository.