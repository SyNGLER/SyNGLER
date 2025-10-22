# SyNGLER Evaluation Framework Setup

We provide a comprehensive evaluation framework for network generation baselines. This document outlines the setup, usage, and key features of our evaluation system.

## 📁 Files Created

We have created the following files for our evaluation framework:

### Core Files
- `synthetic/evaluation/utils.py` - Core evaluation functions and utilities

### Demo Notebooks
- `synthetic/evaluation_demo.ipynb` - Full evaluation demo (all baselines)

### Documentation
- `synthetic/EVALUATION_SETUP.md` - This setup guide

## Quick Start

### Full Demo (All baselines)
```bash
cd synthetic
jupyter notebook evaluation_demo.ipynb
```

## 📊 Evaluation Metrics

We provide comprehensive evaluation using the following metrics:

### 1. Triangle Density
- Measures clustering in the network
- Formula: `(number of triangles) / (maximum possible triangles)`

### 2. Global Clustering Coefficient
- Measures overall transitivity
- Formula: `3 * (number of triangles) / (number of connected triples)`

### 3. Degree Centrality Energy Distance
- Measures preservation of degree distribution
- Uses energy distance between degree sequences

### 4. Eigenvalues Energy Distance
- Measures preservation of spectral properties
- Uses energy distance between eigenvalue sequences

### Additional Metrics (Available in utils)
- Local Clustering Coefficients
- MMD Distance
- Fiedler Value

## 🎯 Supported Baselines

We support the following baselines:

- **SyNGLER-Diff**: Diffusion-based generation
- **SyNGLER-Res**: Residual-based generation
- **GRAN**: Graph Recurrent Attention Networks
- **EDGE**: Edge-based generation
- **VGAE**: Variational Graph Auto-Encoders
- **ER**: Erdos-Renyi random graphs (generated on-the-fly)

## 📈 Supported Datasets

Our framework works with:

- **DBLP**: Academic collaboration network
- **PolBlogs**: Political blog network
- **Yelp**: Business review network
- **YouTube**: Social network

## 🔧 Core Functions

### Data Loading
- `load_real_data(dataset_name, data_root)` - Load real datasets
- `load_synthetic_data(baseline_name, dataset_name, data_root, num_samples)` - Load synthetic data

### Metrics Computation
- `compute_metrics(real_data, synthetic_data_list, device)` - Main evaluation function
- `triangle_density(A_batch, device, return_numpy)` - Triangle density
- `global_clustering_coefficient(A_batch, device, return_numpy)` - Global clustering
- `degree_centrality(A_batch, device)` - Degree centrality
- `eigenvalues(A_batch, device, return_eigen, return_fiedler)` - Eigenvalues and Fiedler value
- `energy_distance(x, y, device, dtype)` - Energy distance between distributions

### Statistical Analysis
- `compute_mmd(x, y, subsample, seed)` - Maximum Mean Discrepancy
- `local_clustering_coefficients(A_batch, device)` - Local clustering analysis

## 📋 Data Format Requirements

### Real Data
- **Path**: `../../datasets/{dataset}/generator/seed=0.npy`
- **Format**: Adjacency matrix (numpy array)

### Synthetic Data
- **Adjacency matrices**: `{baseline}-sample/{dataset}/rep{i}.npy`
- **Latent representations**: `{baseline}-sample/{dataset}/rep{i}.npz` with 'Z' and 'alpha' keys


## 📊 Output

Our framework generates:

1. **Summary Tables**: Real values, synthetic means, energy distances
2. **Comparison Tables**: Synthetic vs real with error analysis
3. **Ranking Analysis**: Performance ranking across datasets
4. **Visualizations**: Error plots, performance comparisons
5. **CSV Exports**: All results saved for further analysis

## 🔄 Adding New Baselines

To add a new baseline to our framework:

1. **Generate synthetic data** using your method
2. **Save in appropriate format**:
   - Adjacency matrices: `.npy` files
   - Latent representations: `.npz` files with 'Z' and 'alpha' keys
3. **Update `load_synthetic_data()`** in `utils.py`
4. **Add to baselines list** in evaluation notebooks

Example:
```python
elif baseline_name == "your_baseline":
    base_path = os.path.join(data_root, dataset_name, "your-baseline-sample")
    for i in range(num_samples):
        file_path = os.path.join(base_path, f"sample_{i}.npy")
        if os.path.exists(file_path):
            synthetic_data.append(np.load(file_path))
```

## 📦 Dependencies

We require the following Python packages:

- Python 3.7+
- NumPy
- PyTorch
- Pandas
- Matplotlib
- Seaborn
- SciPy

## 🎯 Key Features

### 1. Comprehensive Evaluation
- Multiple network metrics
- Statistical distance measures
- Energy distance for distribution comparison

### 2. Flexible Data Loading
- Support for different baseline formats
- Automatic format detection
- Error handling for missing data

### 3. Reproducible Results
- Fixed random seeds
- CSV export functionality
- Detailed logging

### 4. Easy Extension
- Modular design
- Clear function interfaces
- Comprehensive documentation

### 5. Visual Analysis
- Error bar charts
- Performance comparisons
- Ranking visualizations

## 🚨 Troubleshooting

### Common Issues

1. **CUDA not available**: Framework automatically falls back to CPU
2. **Missing synthetic data**: ER baseline is generated on-the-fly in the demo
3. **Path errors**: Ensure relative paths are correct from notebook location

### Getting Help

If you encounter issues:

1. Verify data paths and formats
2. Check device availability (CUDA/CPU)
3. Review error messages in notebook output

## 📚 Next Steps

1. **Run the evaluation demo** to familiarize yourself with the framework
2. **Generate synthetic data** for your baselines
3. **Run the full evaluation** with all baselines
4. **Analyze results** using provided visualizations
5. **Extend the framework** for additional metrics or baselines

## 🎉 Conclusion

Our evaluation framework provides a robust, comprehensive, and extensible system for evaluating network generation baselines. We have designed it to be easy to use while providing detailed analysis capabilities for research purposes.

The framework is ready to use and can be easily extended to support additional baselines, metrics, or datasets as needed.