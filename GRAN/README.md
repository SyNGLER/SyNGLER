
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


## Run

### Train
* Original training:
  * To run the training of experiment ```X``` where ```X``` is one of {```gran_grid```, ```gran_DD```, ```gran_DB```, ```gran_lobster```}:

    ```python run_exp.py -c config/X.yaml```

* Adapted training:
  * To run the experiments for YouTube, DBLP, and PolBlogs, please refer to the corresponding `run_{dataset}.py` file and run it. Make sure to provide both the **input data path** and the **output directory** for generated results.
  * Required arguments
    - `--data-root`: path to the prepared dataset files.
    - `--out-root`: directory to save generated models/samples.
  * The generation process is included in `run_{dataset}.py`.

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


