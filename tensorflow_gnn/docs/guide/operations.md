# Graph Tensor Broadcasting and Pooling Operations

The `GraphTensor` class has associated operators to broadcast and pool features
between node sets, edge sets and graph context. These form the low-level API for
building GNN models.

A `GraphTensor` contains features indexed by nodes of a node set, edges of an
edge set, or by graph components (for context features). Basic operations
propagating information on a graph involves transforming feature values between
nodes and incident edges as well as between nodes/edges in a component and their
graph context. The two main categories of operations are:

* **Broadcasting**, that is, repeating a feature value for each receiver.
  For example, broadcasting from nodes to outgoing or incoming edges
  repeats the entries of a node-indexed tensor into an edge-indexed tensor.
* **Pooling**, that is, collecting feature values from multiple origins for
  each receiver and aggregating them with a pooling function such as summation.
  For example, sum-pooling from edges to incident nodes adds up the entries of
  an edge-indexed tensor into a node-indexed tensor.

Each operation receives a GraphTensor with the structural information (node
adjacencies, components). The input feature can be referenced from the
GraphTensor by name or passed in as an actual value. The latter is convenient
when there is a need to create intermediate context, node or edge values which
are not intended to be materialized as graph features (e.g., computing edge
messages in a convolution function before pooling them to graph nodes). The
output is returned to the caller for further use. The input GraphTensor remains
unchanged.

Ragged and dense values are supported by all methods, but XLA support is
currently planned only for dense values as XLA compiler requires that tensors
have static (compile-time) shapes. See https://www.tensorflow.org/xla.

NOTE: For efficiency reasons all operations are implemented only for scalar
graph tensors (rank=0). Use `GraphTensor.merge_batch_to_components()` to
reshape any graph tensor to a scalar graph tensor.

Example1. Graph classification.

```python
graph_embedding = pool_nodes_to_context(graph_tensor, 'node', 'sum',
                                        feature_name='h')
logit_layer = tf.keras.layers.Dense(N_CLASSES)
logit = logit_layer(graph_embedding)
label = graph_tensor.context['label']
```

Example2. Weighted message passing.

```python
# For each `a->b` edge gather source (`a`) and target (`b`) node states.
a_h = broadcast_node_to_edges(graph_tensor, 'a->b', gt.SOURCE,
                              feature_name='h')
b_h = broadcast_node_to_edges(graph_tensor, 'a->b', gt.TARGET,
                              feature_name='h')
# Extract `a->b` edge weight.
weight = graph_tensor.edge_sets['a->b']['weight']
weight = tf.expand_dims(weight, axis=-1)

# Compute edge message by applying linear layer on top of concatenated source
# and target node states.
message_layer = tf.keras.layers.Dense(MESSAGE_SIZE)
message = message_layer(tf.concat([a_h, b_h], axis=-1))

# Compute weighted sum of messages.
num = pool_edges_to_node(graph_tensor, 'a->b', gt.TARGET, 'sum',
                         feature_value=weight * message)
den = pool_edges_to_node(graph_tensor, 'a->b', gt.TARGET, 'sum',
                         feature_value=weight)
pooled_message = num / (den + kTolerance)
```
