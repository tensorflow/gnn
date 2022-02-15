# GraphSAGE Network implementation and respective convolution layers.

## Overview

This code implements GraphSAGE Networks, originally published by
  * William L. Hamilton, Rex Ying, Jure Leskovec:
    ["Inductive Representation Learning
    on Large Graphs"] (https://arxiv.org/abs/1706.02216), 2017.

TensorFlow programs can import it as
```python
from tensorflow_gnn.models import graph_sage
```
to reuse the following components:
  * `GraphSAGEPoolingConv` for use with the node updates of a
     a `tfgnn.keras.layers.GraphUpdate`.
  * `GraphSAGENextState` for one round of GraphSAGE on a `GraphTensor` with one
    node set and edge set.

## Maintenance and stability

This code is experimental for now, with no promises of maintenance or stability,
and no assigned maintainer. Use at your own risk.

