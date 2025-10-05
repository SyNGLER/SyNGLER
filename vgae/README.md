# Variational Graph Auto-encoder
This repository implements variational graph auto-encoder by Thomas Kipf. For details of the model, refer to his original [tensorflow implementation](https://github.com/tkipf/gae) and [his paper](https://arxiv.org/abs/1611.07308). We also use the [pytorch implement](https://github.com/DaehanKim/vgae_pytorch).  

# Requirements

* Pytorch 
* python 3.x
* networkx
* scikit-learn
* scipy

# How to run
* Simulated datasets: 
  * First ensure that the simulated datasets are generated in `../datasets/simulation`.
  * run `python batch_train.py`.
* Real-world datasets:
  * The real world datasets are already provided in `../datasets/`.
  * Run template:`python real_data_train.py --datasets youtube`. Please specify the dataset for training and sampling. Choices include `youtube`, `yelp`,`dblp`, `polblogs`.

