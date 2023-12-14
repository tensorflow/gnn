# The TF-GNN In Memory Sampler

## Overview

In many applications graphs are small enough to fit on a single worker machine
memory (e.g.
[OGB Node Property Prediction](https://ogb.stanford.edu/docs/nodeprop)). The
good representation for those small (heterogeneous) graphs is
`tfgnn.GraphTensor` itself. The TF-GNN library provides set of tools that allow
graph sampling from those small graphs using graph schema and sampling spec, as
covered in [the data prep guide](./data_prep.md).

Below we describe in-memory sampling using a toy example graph. For complete
end-to-end story see
[Solving OGBN-MAG end-to-end](https://colab.research.google.com/github/tensorflow/gnn/blob/master/examples/notebooks/ogbn_mag_e2e.ipynb).

## Sampling from GraphTensor A,B,C

For the sake of example, let's consider toy homogeneous graph with nodes `A, B,
C` and edges `A->{B, C}, B->{A}, C->{B}`:

```python
import tensorflow as tf
import tensorflow_gnn as tfgnn

full_graph = tfgnn.homogeneous(
    tf.constant([0, 0, 1, 2]),
    tf.constant([1, 2, 0, 1]),
    node_features={'name': tf.constant(['A', 'B', 'C'])},
)
```

NOTE: for real applications, the full graph could be stored as `tf.Example`,
e.g. using `tfgnn.write_example`, and loaded using
`tfgnn.parse_single_example`.

Now, consider uniform sampling for two hops with sampling sizes `[4, 2]`. The
library provides `tfgnn.sampler.SamplingSpecBuilder()` class that simplifies
sampling spec construction:

```python
graph_schema = tfgnn.create_schema_pb_from_graph_spec(full_graph)

sampling_spec_builder = tfgnn.sampler.SamplingSpecBuilder(
    graph_schema,
    default_strategy=tfgnn.sampler.SamplingStrategy.RANDOM_UNIFORM)

seed = sampling_spec_builder.seed("nodes")
hop1 = seed.sample(4, "edges")
hop2 = hop1.sample(2, "edges")

sampling_spec = sampling_spec_builder.build()
```

The sampling could be encapsulated as a "sampling model", instance of
`tf.keras.Model`, which takes as an input collection of seed node ids and
returns sampled subgaphs as `tfgnn.GraphTensor`.

The sampler library provides helper function that constructs sampling models
using graph schema, sampling spec and two factory methods: one for edges and one
for node features (see
[Sampling API](https://github.com/tensorflow/gnn/blob/main/tensorflow_gnn/experimental/sampler)
for mode details).

```python
from tensorflow_gnn.experimental import sampler

def edge_sampler(sampling_op: tfgnn.sampler.SamplingOp):
  edge_set_name = sampling_op.edge_set_name
  sample_size = sampling_op.sample_size
  return sampler.InMemUniformEdgesSampler.from_graph_tensor(
      full_graph, edge_set_name, sample_size=sample_size, name=sampling_op.op_name
  )

def get_features(node_set_name: tfgnn.NodeSetName):
  return sampler.InMemIndexToFeaturesAccessor.from_graph_tensor(
      full_graph, node_set_name
  )

sampling_model = sampler.create_sampling_model_from_spec(
    graph_schema, sampling_spec, edge_sampler, get_features,
    seed_node_dtype=tf.int32)
```

Here the `sampling_model` takes *indices* of seed nodes in the `full_graph` as
its input and returns sampled `tfgnn.GraphTensor` as result. The seeds are
ragged rank 1 tensors with a shape `[batch_size, (num_seeds)]`. The 1st
dimension is a batch dimension. It is used to create batches of sampled
subgrahs. The 2nd ragged dimension allows to specify potentially multiple
seed nodes to use a a starting points for sampling for each subgraph, as:

`[[subgraph1/seed1, subgraph1/seed2,...], [subgraph2/seed1, ...], ...]`

The result `tfgnn.GraphTensor` has shape `[batch_size]` with one graph component
per graph.

The sampling model can be used with `tf.data.Dataset` to construct input [data
pipeline](https://www.tensorflow.org/guide/data):

```python
seeds = tf.data.Dataset.from_tensor_slices([0, 1, 2, 0])
# Create batches of up to two seeds
seeds = seeds.batch(2)
# [seed1, seed2] -> [[seed1], [seed2]]
seeds = seeds.map(
    lambda s: tf.RaggedTensor.from_row_lengths(s, tf.ones_like(s))
)
graphs = seeds.map(sampling_model)
for graph in graphs:
  print(graph)
```
