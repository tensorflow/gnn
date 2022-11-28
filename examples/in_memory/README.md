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

Both binaries accept identical flags, including:

* `--dataset=<dataset_name>`. We currently support `ogbn-*` and planetoid
  datasets, such as, `--dataset=ogbn-arxiv` or `--dataset=pubmed`. Datasets are
  implemented in file `datasets.py`.
* `--model=<model_name>`. For instance, `--model=JKNet` or `--model=GCN`,
  respectively, for models of [[2]](#2) or [[3]](#3). Models are implemented in
  file `models.py`.
* `--model_kwargs_json=JSON_ENCODING` (optional), where `JSON_ENCODING` depends
  on the `--model` parameter. See *Experiments on Planetoid Datasets*.
* Other flags, such as learning-rate, regularization, number of data epochs, are
  available. You may run:
  
  ```sh
  python3 tf_trainer.py --help
  ```

  for an explanation of all the flags.

### Experiments on OGBN Datasets

You can run experiments on OGBN datasets, e.g.,

```
python3 tf_trainer.py --dataset=ogbn-arxiv
```


### Experiments on Planetoid Datasets

To replicate the GCN paper [[3]](#3), you can run as:

```sh
python3 tf_trainer.py --dataset=cora --model=GCN --model_kwargs_json='{"dropout": 0.8, "hidden_units": 32, "depth": 2, "use_bias": false}' --l2_regularization=1e-2 --steps=500
```

You can replace the `--dataset=cora` value with `pubmed` or `citeseer`.

**Note**: these Planetoid datasets are sensitive to initial conditions and
hyper-parameters.


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
