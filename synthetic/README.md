# SyNGLER Evaluation Framework

We provide a comprehensive evaluation framework for network generation baselines. Our framework enables systematic comparison of different approaches using multiple network metrics and statistical distances.

## 🚀 Quick Start

### Full Demo (All baselines)
```bash
cd synthetic
jupyter notebook evaluation_demo.ipynb
```

## 📊 Evaluation Metrics

We provide comprehensive evaluation using the following metrics:

1. **Triangle Density** - Measures clustering in the network
2. **Global Clustering Coefficient** - Measures overall transitivity
3. **Degree Centrality Energy Distance** - Measures degree distribution preservation
4. **Eigenvalues Energy Distance** - Measures spectral properties preservation

## 🎯 Supported Baselines

- **SyNGLER-Diff**: Diffusion-based generation
- **SyNGLER-Res**: Residual-based generation
- **GRAN**: Graph Recurrent Attention Networks
- **EDGE**: Edge-based generation
- **VGAE**: Variational Graph Auto-Encoders
- **ER**: Erdos-Renyi random graphs

## 📈 Supported Datasets

- **DBLP**: Academic collaboration network
- **PolBlogs**: Political blog network
- **Yelp**: Business review network
- **YouTube**: Social network

## 📁 File Structure

```
synthetic/
├── evaluation/
│   └── utils.py                    # Core evaluation functions
├── evaluation_demo.ipynb          # Full evaluation demo
├── EVALUATION_SETUP.md           # Setup and usage guide
└── README.md                     # This file
```


## 📚 Documentation

- **Setup Guide**: `EVALUATION_SETUP.md`

## 🔧 Key Features

- **Comprehensive Evaluation**: Multiple network metrics and statistical distances
- **Flexible Data Loading**: Support for different baseline formats
- **Reproducible Results**: Fixed random seeds and CSV export
- **Easy Extension**: Modular design for adding new baselines
- **Visual Analysis**: Error plots and performance comparisons

## 📦 Dependencies

- Python 3.7+
- NumPy
- PyTorch
- Pandas
- Matplotlib
- Seaborn
- SciPy

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for details on how to submit pull requests, report issues, or suggest enhancements.
