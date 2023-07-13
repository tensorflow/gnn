# Copyright 2021 The TensorFlow GNN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""The MapFeatures layer and related definitions."""

import re
from typing import Mapping, Optional, Union

import tensorflow as tf

from tensorflow_gnn.graph import dict_utils as du
from tensorflow_gnn.graph import graph_constants as const
from tensorflow_gnn.graph import graph_tensor as gt


@tf.keras.utils.register_keras_serializable(package="GNN")
class MapFeatures(tf.keras.layers.Layer):
  """Transforms features on a GraphTensor by user-defined callbacks.

  This layer transforms the feature maps of graph pieces (that is, EdgeSets,
  NodeSets, or the Context) by applying Keras Models to them. Those Models
  are built by user-supplied callbacks that receive a KerasTensor for the
  graph piece as input and return a dict of output features computed with
  the Keras functional API, see https://tensorflow.org/guide/keras/functional.

  Auxiliary graph pieces (e.g., for `tfgnn.keras.layers.StructuredReadout`)
  are skipped, unless explicitly requested via `allowed_aux_node_sets_pattern`
  or `allowed_aux_edge_sets_pattern`.

  Examples:

  ```python
  # Hashes edge features called "id", leaves others unchanged:
  def edge_sets_fn(edge_set, *, edge_set_name):
    features = edge_set.get_features_dict()
    ids = features.pop("id")
    num_bins = 100_000 if edge_set_name == "views" else 20_000
    hashed_ids = tf.keras.layers.Hashing(num_bins=num_bins)(ids)
    features["hashed_id"] = hashed_ids
    return features
  graph = tfgnn.keras.layers.MapFeatures(edge_sets_fn=edge_sets_fn)(graph)
  ```

  ```python
  # A simplistic way to map node features to an initial state.
  def node_sets_fn(node_set, *, node_set_name):
    state_dims_by_node_set = {"author": 32, "paper": 64}  # ...and so on.
    state_dim = state_dims_by_node_set[node_set_name]
    features = node_set.features  # Immutable view.
    if features: # Concatenate and project all inputs (assumes they are floats).
      return tf.keras.layers.Dense(state_dim)(
          tf.keras.layers.Concatenate([v for _, v in sorted(features.items())]))
    else:  # There are no inputs, create an empty state.
      return tfgnn.keras.layers.MakeEmptyFeature()(node_set)
  graph = tfgnn.keras.layers.MapFeatures(node_sets_fn=node_sets_fn)(graph)
  ```

  ```python
  # Doubles all feature values, with one callback used for all graph pieces,
  # including auxiliary ones.
  def fn(inputs, **unused_kwargs):
    return {k: tf.add(v, v) for k, v in inputs.features.items()}
  graph = tfgnn.keras.layers.MapFeatures(
      context_fn=fn, node_sets_fn=fn, edge_sets_fn=fn,
      allowed_aux_node_sets_pattern=r".*", allowed_aux_edge_sets_pattern=r".*"
  )(graph)
  ```

  When this layer is called on a GraphTensor, it transforms the feature map
  of each graph piece with the model built by the respective callbacks.
  The very first call to this layer triggers building the models. Subsequent
  calls to this layer do not use the callbacks again, but check that their
  input does not have more graph pieces or features than seen by the callbacks:

    * It is an error to call with a node set or edge set that was not present
      in the first call. (After the first call, it is too late to initialize
      another model for it and find out what the callback would have done.)
      An exception is made for auxiliary node sets and edge sets: If they would
      have been ignored in the first call anyways, they may be present in later
      calls and get ignored there.
    * It is an error to call with a set of feature names of some graph piece
      that has changed since the first call, except for those graph pieces for
      which the callback was `None` or returned `None` to request passthrough.
      (Without this check, the model for the graph piece would silently drop
      new features, even though the callback might have handled them.)

  More details on the callbacks:

  The model-building callbacks are passed as arguments when initializing this
  layer (see "Init args" below). Each callback is invoked as
  `fn(graph_piece, **kwargs)` where

    * `graph_piece` is a KerasTensor for the EdgeSet, NodeSet or Context
      that is being transformed. It provides access to the input features.
    * the keyword argument (if any) is
        * `edge_set_name=...` when transforming the features of that EdgeSet,
        * `node_set_name=...` when transforming the features of that NodeSet,
        * absent when transforming the features of the Context.

  The output of the callbacks can take one of the following forms:

    * A returned dict of feature values is used as the new feature map of
      the respective graph piece in this layer's output. Returning the
      empty dict `{}` is allowed and results in an empty feature map.
    * A returned feature `value` not wrapped in a dict is a shorthand for
      `{tfgnn.HIDDEN_STATE: value}`, to simplify the set-up of initial
      states.
    * Returning `None` as the callback's result indicates to leave this graph
      piece alone and not even validate that subsequent inputs have the same
      features.

  The output values are required to

    * have the correct shape for a feature on the respective piece of the
      GraphTensor;
    * depend on the input, so that the Keras functional API can use them
      as Model outputs.

  This happens naturally for outputs of transformed input features.
  Outputs created from scratch still need to depend on the input for its size.
  The helper `tfgnn.keras.layers.MakeEmptyFeature()(graph_piece)` does this
  for the common case of creating an empty hidden state for a latent node;
  see its documentation for details on how to use it with TPUs.
  If TPUs and shape inference are no concern, the callback can simply use
  `graph_piece.sizes` or (esp. for rank 0) graph_piece.total_size` to construct
  outputs of the right shape, but not `graph_piece.spec.total_size`, which
  breaks the dependency chain of KerasTensors.

  Weight sharing between the transformation of different graph pieces is
  possible by sharing the Keras objects between the respective callback
  invocations.

  WARNING: Weight sharing fails in `tf.keras.models.load_model()`
  with an error message on weights missing from the checkpoint.
  (Most users don't need to re-load their models this way.)

  TODO(b/285243815): Remove warning when fixed.

  Init args:
    context_fn: A callback to build a Keras model for transforming context
      features. It will be called as `output = context_fn(g.context)`.
      Leaving this at the default `None` is equivalent to returning `None`.
    node_sets_fn: A callback to build a Keras model for transforming node set
      features. It will be called for every node sets as
      `node_sets_fn(g.node_sets[node_set_name], node_set_name=node_set_name)`.
      Leaving this at the default `None` is equivalent to returning `None`
      for every node set.
    edge_sets_fn: A callback to build a Keras model for transforming edge set
      features. It will be called for every edge sets as
      `edge_sets_fn(g.edge_sets[edge_set_name], edge_set_name=edge_set_name)`.
      Leaving this at the default `None` is equivalent to returning `None`
      for every edge set.
    allowed_aux_node_sets_pattern: If set, `node_sets_fn` is also invoked for
      those auxiliary node sets that match this pattern, according to Python's
      `re.fullmatch(pattern, node_set_name)`.
    allowed_aux_edge_sets_pattern: If set, `edge_sets_fn` is also invoked for
      those auxiliary edge sets that match this pattern, according to Python's
      `re.fullmatch(pattern, edge_set_name)`.

  Call args:
    graph: A GraphTensor. The very first call triggers the building of
      the models that map the various feature maps, with tensor specs
      taken from the GraphTensorSpec of the first input.

  Call returns:
    A GraphTensor with the same nodes and edges as the input, but with
    transformed feature maps.
  """

  def __init__(self,
               context_fn=None,
               node_sets_fn=None,
               edge_sets_fn=None,
               *,
               allowed_aux_node_sets_pattern: Optional[str] = None,
               allowed_aux_edge_sets_pattern: Optional[str] = None,
               **kwargs):
    from_config = kwargs.pop("_from_config", False)
    if from_config:
      context_model = kwargs.pop("context_model")
      node_set_models = kwargs.pop("node_set_models")
      edge_set_models = kwargs.pop("edge_set_models")
    super().__init__(**kwargs)

    if not from_config:
      # Usually, an object of this class is partially initialized by remembering
      # the user-supplied model builder functions that are called later to
      # initialize the actual mapper models, once the GraphTensorSpec is known.
      self._context_fn = context_fn
      self._node_sets_fn = node_sets_fn
      self._edge_sets_fn = edge_sets_fn
      self._context_model = None
      self._node_set_models = None
      self._edge_set_models = None
      self._is_initialized = False
    else:
      # In the special case of restoring a saved model from a config,
      # the mapper models are restored directly.
      self._context_fn = None
      self._node_sets_fn = None
      self._edge_sets_fn = None
      self._context_model = context_model
      self._node_set_models = node_set_models
      self._edge_set_models = edge_set_models
      self._is_initialized = True
    self._allowed_aux_node_sets_pattern = allowed_aux_node_sets_pattern
    self._allowed_aux_edge_sets_pattern = allowed_aux_edge_sets_pattern

  def get_config(self):
    if not self._is_initialized:
      raise ValueError("Cannot get a config for saving a MapFeatures layer "
                       "before it has been built (during the first call).")
    return dict(
        context_model=self._context_model,
        # Sublayers need to be top-level objects in the config (b/209560043).
        **du.with_key_prefix(self._node_set_models, "node_set_models/"),
        **du.with_key_prefix(self._edge_set_models, "edge_set_models/"),
        allowed_aux_node_sets_pattern=self._allowed_aux_node_sets_pattern,
        allowed_aux_edge_sets_pattern=self._allowed_aux_edge_sets_pattern,
        **super().get_config())

  @classmethod
  def from_config(cls, config):
    config["node_set_models"] = du.pop_by_prefix(config, "node_set_models/")
    config["edge_set_models"] = du.pop_by_prefix(config, "edge_set_models/")
    return cls(**config, _from_config=True)

  def _init_from_spec(self, spec: gt.GraphTensorSpec):
    self._context_model = _make_model_or_none(
        self._context_fn, spec.context_spec)

    # All node sets seen at initialization time. Value `None` means ignore.
    self._node_set_models = {}
    for node_set_name, node_set_spec in spec.node_sets_spec.items():
      if self._ignore_node_set(node_set_name):
        continue
      self._node_set_models[node_set_name] = _make_model_or_none(
          self._node_sets_fn, node_set_spec, node_set_name=node_set_name)

    # All edge sets seen at initialization time. Value `None` means ignore.
    self._edge_set_models = {}
    for edge_set_name, edge_set_spec in spec.edge_sets_spec.items():
      if self._ignore_edge_set(edge_set_name):
        continue
      self._edge_set_models[edge_set_name] = _make_model_or_none(
          self._edge_sets_fn, edge_set_spec, edge_set_name=edge_set_name)

    self._is_initialized = True

  def call(self, graph: gt.GraphTensor) -> gt.GraphTensor:
    if not self._is_initialized:
      with tf.init_scope():
        self._init_from_spec(graph.spec)
        self._context_fn = self._node_sets_fn = self._edge_sets_fn = None
    assert self._is_initialized

    context_features = None
    if self._context_model is not None:
      context_features = _call_model(self._context_model, graph.context,
                                     logging_name="context")

    node_set_features = {}
    for node_set_name, node_set in graph.node_sets.items():
      try:
        model = self._node_set_models[node_set_name]
        if model is None: continue  # Was explicitly ignored in initialization.
      except KeyError as e:
        if self._ignore_node_set(node_set_name):
          continue  # Would have been ignored in initialization.
        raise KeyError(f"Unexpected node set '{node_set_name}' "
                       "not seen in first call") from e
      node_set_features[node_set_name] = _call_model(
          model, node_set, logging_name=f"node_set '{node_set_name}'")

    edge_set_features = {}
    for edge_set_name, edge_set in graph.edge_sets.items():
      try:
        model = self._edge_set_models[edge_set_name]
        if model is None: continue  # Was explicitly ignored in initialization.
      except KeyError as e:
        if self._ignore_edge_set(edge_set_name):
          continue  # Would have been ignored in initialization.
        raise KeyError(f"Unexpected edge set '{edge_set_name}' "
                       "not seen in first call") from e
      edge_set_features[edge_set_name] = _call_model(
          model, edge_set, logging_name=f"edge_set '{edge_set_name}'")

    result = graph.replace_features(context=context_features,
                                    node_sets=node_set_features,
                                    edge_sets=edge_set_features)
    return result

  def _ignore_node_set(self, node_set_name):
    if not gt.get_aux_type_prefix(node_set_name):
      return False
    if self._allowed_aux_node_sets_pattern is None:
      return True
    return not re.fullmatch(self._allowed_aux_node_sets_pattern, node_set_name)

  def _ignore_edge_set(self, edge_set_name):
    if not gt.get_aux_type_prefix(edge_set_name):
      return False
    if self._allowed_aux_edge_sets_pattern is None:
      return True
    return not re.fullmatch(self._allowed_aux_edge_sets_pattern, edge_set_name)


def _make_model_or_none(model_fn, graph_piece_spec, **kwargs):
  """Returns a Model to map this graph piece, or None to leave it alone."""
  if model_fn is None:
    return None  # This graph piece is to be left alone.

  graph_piece_input = tf.keras.layers.Input(type_spec=graph_piece_spec)
  raw_outputs = model_fn(graph_piece_input, **kwargs)
  if raw_outputs is None:
    return None  # As if model_fn were None to begin with.
  if isinstance(raw_outputs, Mapping):
    outputs = dict(raw_outputs)
  else:
    outputs = {const.HIDDEN_STATE: raw_outputs}

  non_keras_outputs = {k: v for k, v in outputs.items()
                       if not tf.keras.backend.is_keras_tensor(v)}
  if non_keras_outputs:
    raise ValueError(
        "`MapFeatures(...=fn)` requires the callback `fn(inputs, ...)` to only "
        "return KerasTensors that depend on the `inputs` graph piece. "
        "For values created from scratch, use a tensor depdendency on "
        "`inputs.total_size` (possibly static, useful for scalar GraphTensors) "
        "or `inputs.sizes`.\n"
        f"The callback for {kwargs or 'context'} "
        f"returned the following non-KerasTensor outputs: {non_keras_outputs}")

  return tf.keras.Model(graph_piece_input, outputs)


def _call_model(model, graph_piece, *, logging_name):
  """Returns results of model, after checking the unchanged set of features."""
  actual_features = set(graph_piece.features.keys())
  expected_features = set(model.input.features.keys())
  if expected_features != actual_features:
    raise ValueError(
        f"The feature set of {logging_name} has changed since this layer "
        f"was built. Expected {sorted(expected_features)}, "
        f"got {sorted(actual_features)}.")

  return model(graph_piece)


# TODO(b/217538005): When fixed, update the paragraph on TPU compatibility
# and the the matching explanations in gnn_modeling.md.
@tf.keras.utils.register_keras_serializable(package="GNN")
class MakeEmptyFeature(tf.keras.layers.Layer):
  """Returns an empty feature with a shape that fits the input graph piece.

  Init args:
    dtype: the tf.DType to use for the result, defaults to tf.float32.
    **kwargs: Other arguments for the tf.keras.layers.Layer base class.

  Call args:
    graph_piece: a Context, NodeSet or EdgeSet from a GraphTensor.

  Call returns:
    A potentially ragged tensor of shape [*graph_shape, (num_items), 0] where
    graph_shape is the shape of the graph_piece and its containing GraphTensor,
    (num_items) is the number of edges, nodes or components contained in the
    graph piece, and 0 is the feature dimension that makes this an empty tensor.
    In particular, if graph_shape == [], meaning graph_piece is from a scalar
    GraphTensor, the result is a Tensor of shape [graph_piece.total_size, 0].

  TPU compatibility:
    If graph_shape == [], the shape of the result is static (as required)
    if graph_piece.spec.total_size is not None. That, however, requires the
    presence of other features on the same graph piece from which its static
    total_size can be inferred. Therefore, to create an empty hidden state for
    a latent graph piece (one without input features), this layer must be used
    already in dataset preprocessing, before padding inputs to fixed sizes.
  """

  def call(self, graph_piece: Union[gt.EdgeSet, gt.NodeSet, gt.Context]):
    def _make_empty_state(size):
      return tf.zeros([size, 0], dtype=self.dtype)

    # graph_rank = 0 occurs inside a model after .merge_batch_to_components(),
    # and .total_size attempts to to provide a constant shape for TPU like so:
    graph_rank = graph_piece.rank
    if graph_rank == 0:
      return _make_empty_state(graph_piece.total_size)

    # graph_rank = 1 occurs for a GraphTensor of shape [batch_size] before
    # .merge_batch_to_components(), so we don't need TPU compatibility.
    # We need to build RaggedTensor of shape [batch_size, (num_items), 0]
    # from the rank-1 tensor of num_items values.
    # TODO(b/228126030): Can num_items be a non-ragged dimension?
    if graph_rank == 1:
      num_items = tf.reduce_sum(graph_piece.sizes, axis=-1)  # Sum components.
      assert num_items.shape.rank == 1
      result = tf.RaggedTensor.from_row_lengths(
          values=_make_empty_state(tf.reduce_sum(num_items)),
          row_lengths=num_items)
      assert result.shape[0:graph_rank].is_compatible_with(num_items.shape)
      assert result.shape[graph_rank:].as_list() == [None, 0]  # [num_items, 0]
      return result

    # TODO(b/228126030): Implement and test the case graph_rank > 1 .
    # (Maybe tf.ragged.map_flat_values(..., num_items) can help.)
    raise NotImplementedError(
        "Cannot yet MakeEmptyFeature for GraphTensor.rank > 1 (b/228126030).")
