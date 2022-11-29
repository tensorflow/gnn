# Module: multi_head_attention

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/models/multi_head_attention/__init__.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Transformer-style multi-head attention.

Users of TF-GNN can use this model by importing it next to the core library as

```python
import tensorflow_gnn as tfgnn
from tensorflow_gnn.models import multi_head_attention
```

## Classes

[`class MultiHeadAttentionConv`](./multi_head_attention/MultiHeadAttentionConv.md):
Transformer-style (dot-product) multi-head attention on GNNs.

## Functions

[`MultiHeadAttentionEdgePool(...)`](./multi_head_attention/MultiHeadAttentionEdgePool.md):
Returns a layer for pooling edges with Transformer-style Multi-Head Attention.

[`MultiHeadAttentionHomGraphUpdate(...)`](./multi_head_attention/MultiHeadAttentionHomGraphUpdate.md):
Returns a GraphUpdate layer with a transformer-style multihead attention.

[`MultiHeadAttentionMPNNGraphUpdate(...)`](./multi_head_attention/MultiHeadAttentionMPNNGraphUpdate.md):
Returns a GraphUpdate layer for message passing with MultiHeadAttention pooling.
