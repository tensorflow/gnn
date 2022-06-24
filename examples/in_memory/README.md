# In-Memory TF-GNN Example
While TF-GNN is developed for learning at scale, we developed this directory for
researchers that are interested on performing graph representation learning, on
datasets that fit in memory. For instance, smaller OGB [[1]](#1) datasets.

## How to run
To perform learning on ogbn-arxiv dataset, you can use one of these two
binaries:

1. Using raw tensorflow features (i.e., training-loop exposed):
  
  ```sh
  python3 tf_trainer.py
  ```

1. Using Keras Model.fit():
  
  ```sh
  python3 keras_trainer.py
  ```

Both of these binaries accept the following flags:

* `--dataset=<dataset_name>`. We currently support `ogbn-*` datasets, such as,
  `--dataset=ogbn-arxiv`. Datasets are implemented in file `datasets.py`.
* `--model=<model_name>`. For instance, `--model=JKNet` or `--model=GCN`,
  respectively, for models of [[2]](#2) or [[3]](#3). Models are implemented in
  file `models.py`.
* Other flags, such as learning-rate, regularization, number of data epochs, are
  available. You may run:
  
  ```sh
  python3 tf_trainer.py --help
  ```

  for an explanation of all the flags.

## References
<a id="1">[1]</a>
Hu, Fey, Zitnik, Dong,  Ren, Liu, Catasta, Leskovec (2020).
Open Graph Benchmark: Datasets for Machine Learning on Graphs.
arXiv:2005.00687

<a id="2">[2]</a>
Xu, Li, Tian, Sonobe, Kawarabayashi, Jegelka (2018).
*Representation Learning on Graphs with Jumping Knowledge Networks*.
International Conference on Machine Learning

<a id="3">[3]</a>
Kipf, Welling (2017).
*Semi-Supervised Classification with Graph Convolutional Networks*.
International Conference on Learning Representations.
