
# GRAN

This is the PyTorch implementation for GRAN adapted from [Efficient Graph Generation with Graph Recurrent Attention Networks](https://arxiv.org/abs/1910.00760). The training and generation processes are as described in the following NeurIPS 2019 paper:

```
@inproceedings{liao2019gran,
  title={Efficient Graph Generation with Graph Recurrent Attention Networks}, 
  author={Liao, Renjie and Li, Yujia and Song, Yang and Wang, Shenlong and Nash, Charlie and Hamilton, William L. and Duvenaud, David and Urtasun, Raquel and Zemel, Richard}, 
  booktitle={NeurIPS},
  year={2019}
}
```

## Dependencies
Python 3, PyTorch(1.2.0)

Other dependencies can be installed via 

  ```pip install -r requirements.txt```

## Quick Start

We've adapted this GRAN implementation to work with three real-world datasets: **DBLP**, **PolBlogs**, and **YouTube**. The Yelp dataset was excluded due to memory constraints (OOM issues). For each of these datasets, you can simply run the corresponding script to train the model and generate samples.

### Supported Datasets

- **DBLP**: Academic collaboration network
- **PolBlogs**: Political blog network  
- **YouTube**: Social network data

### Running Experiments

For each dataset, use the corresponding run script:

**DBLP Dataset:**
```bash
python run_dblp.py --cuda 0 --data-root /path/to/dblp/data --out-root /path/to/output
```

**PolBlogs Dataset:**
```bash
python run_polblogs.py --cuda 0 --data-root /path/to/polblogs/data --out-root /path/to/output
```

**YouTube Dataset:**
```bash
python run_youtube.py --cuda 0 --data-root /path/to/youtube/data --out-root /path/to/output
```

### Required Arguments

- `--cuda`: Comma-separated visible device IDs (e.g., '0' or '2,3')
- `--data-root`: Path to the prepared dataset files (should contain `seed={seed}.npy` files and `lcc.csv`)
- `--out-root`: Directory to save generated models and samples

### Optional Arguments

- `--n`: Upper bound for number of nodes (default: 1500 for DBLP/PolBlogs, 2000 for YouTube)
- `--seed-start`: Start seed (inclusive, default: 0)
- `--seed-end`: End seed (inclusive, default: 0 for DBLP, 199 for PolBlogs/YouTube)
- `--max-epoch`: Override training epochs
- `--train-batch-size`: Override training batch size
- `--gen-num`: Override number of generated samples
- `--hidden-dim`: Override model hidden dimension
- `--embedding-dim`: Override model embedding dimension
- `--just-test`: Only generate samples without training (requires pre-trained model)

### Example Usage

```bash
# Train and generate samples for DBLP dataset
python run_dblp.py --cuda 0 --data-root datasets/dblp/generator --out-root results/dblp

# Generate samples only (after training) for PolBlogs
python run_polblogs.py --cuda 0 --data-root datasets/polblogs/generator --out-root results/polblogs --just-test

# Train with custom parameters for YouTube
python run_youtube.py --cuda 0 --data-root datasets/youtube/generator --out-root results/youtube --max-epoch 1000 --hidden-dim 256
```

## Original Implementation

### Train
* Original training:
  * To run the training of experiment ```X``` where ```X``` is one of {```gran_grid```, ```gran_DD```, ```gran_DB```, ```gran_lobster```}:

    ```python run_exp.py -c config/X.yaml```

**Note**:

* Please check the folder ```config``` for a full list of configuration yaml files.
* Most hyperparameters in the configuration yaml file are self-explanatory.

### Test

* After training, you can specify the ```test_model``` field of the configuration yaml file with the path of your best model snapshot, e.g.,

  ```test_model: exp/gran_grid/xxx/model_snapshot_best.pth```	

* To run the test of experiments ```X```:

  ```python run_exp.py -c config/X.yaml -t```

**Note**:

* Please check the [evaluation](https://github.com/JiaxuanYou/graph-generation) to set up.


