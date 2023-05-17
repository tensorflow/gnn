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
"""Task for integrated gradients attribution method.

This task will wrap any `tfgnn.GraphTensor` model with another `tf.keras.Model`
model that provides an additional `tf.function` for computing integrated
gradients by Riemann sum approximation.

The task implements the method as described in:
https://papers.nips.cc/paper/2020/hash/417fbbf2e9d5a28a855a11894b2e795a-Abstract.html.
"""
import operator
import os
from typing import Callable, Optional, Sequence, Union

import numpy as np
import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn.runner import interfaces


def reduce_graph_sequence(
    graphs: Sequence[tfgnn.GraphTensor],
    initializer: Optional[tfgnn.GraphTensor] = None,
    *,
    context_fn: Callable[[Sequence[tfgnn.Field]], tfgnn.Field],
    node_set_fn: Callable[[Sequence[tfgnn.Field]], tfgnn.Field],
    edge_set_fn: Callable[[Sequence[tfgnn.Field]], tfgnn.Field]
) -> tfgnn.GraphTensor:
  """Reduces a sequence of graph tensors to a single graph tensor.

  Args:
    graphs: A sequence of `tfgnn.GraphTensor.`
    initializer: A `tfgnn.GraphTensor` with the structure of the returned
      `tfgnn.GraphTensor,` if unset, `graphs[0]` is used.
    context_fn: A function reducing a sequence of `tfgnn.Field` to a single
      `tfgnn.Field.`
    node_set_fn: A function reducing a sequence of `tfgnn.Field` to a single
      `tfgnn.Field.`
    edge_set_fn: A function reducing a sequence of `tfgnn.Field` to a single
      `tfgnn.Field.`

  Returns:
    A single `tfgnn.GraphTensor` with the structure of `graph` and reduce
    feature values of `graphs.`
  """
  if not graphs and not initializer:
    raise TypeError(
        "reduce_graph_sequence() of empty sequence with no initial value")
  elif graphs:
    initializer = graphs[0]
  elif initializer:
    graphs = (initializer,)

  if not all(initializer.spec == g.spec for g in graphs):
    raise ValueError("reduce_graph_sequence() with graphs of different spec")

  context, edge_sets, node_sets = {}, {}, {}

  for k in initializer.context.features.keys():
    context[k] = context_fn([g.context.features[k] for g in graphs])

  for k, v in initializer.edge_sets.items():
    features = {}
    for kk in v.features.keys():
      features[kk] = edge_set_fn([g.edge_sets[k].features[kk] for g in graphs])
    edge_sets[k] = features

  for k, v in initializer.node_sets.items():
    features = {}
    for kk in v.features.keys():
      features[kk] = node_set_fn([g.node_sets[k].features[kk] for g in graphs])
    node_sets[k] = features

  return initializer.replace_features(
      context=context,
      edge_sets=edge_sets,
      node_sets=node_sets)


def counterfactual(graph: tfgnn.GraphTensor,
                   *,
                   random: bool = True,
                   seed: Optional[int] = None) -> tfgnn.GraphTensor:
  """Return a `tfgnn.GraphTensor` counterfactual.

  Random uniform or zero'd counterfactuals are produced. For a random uniform
  counterfactual, the min and max values of the target features of `graph` are
  used: this produces a counterfactual with the same range as the target
  features of `graph` but with maximum entropy.

  Args:
    graph: A `tfgnn.GraphTensor.`
    random: Whether to produce a random uniform counterfactual.
    seed: An optional random seed.

  Returns:
    A counterfactual `tfgnn.GraphTensor.`
  """
  if random:
    def fn(inputs):
      dtype = inputs[0].dtype
      minval = tf.math.reduce_min(*inputs)
      maxval = tf.math.reduce_max(*inputs)
      # The `maxval` boundary is exclusive for `tf.random.uniform`:
      # For integers, add one to obtain a closed range.
      # For floats, the half-open range is ok. (The special case
      # minval = maxval is accepted by TF and uses that single value.)
      if dtype.is_integer:
        maxval = maxval + 1
      return tf.random.uniform(tf.shape(*inputs), minval, maxval, dtype, seed)
  else:
    fn = lambda inputs: tf.zeros_like(*inputs)
  return reduce_graph_sequence(
      (graph,),
      context_fn=fn,
      edge_set_fn=fn,
      node_set_fn=fn)


def subtract_graph_features(graph_a: tfgnn.GraphTensor,
                            graph_b: tfgnn.GraphTensor) -> tfgnn.GraphTensor:
  """Return the element-wise delta between two graph tensors.

  The returned graph tensor contains the deltas between all pairs of edge set,
  feature name and all pairs of node set, feature name.

  `graph_a` and `graph_b` must share a `tfgnn.GraphTensorSpec.`

  Args:
    graph_a: A `tfgnn.GraphTensor.`
    graph_b: A `tfgnn.GraphTensor.`

  Returns:
    A delta `tfgnn.GraphTensor.`
  """
  if graph_a.spec != graph_b.spec:
    raise ValueError("subtract_graph_features() with graphs of different spec")

  fn = lambda inputs: tf.math.subtract(*inputs)

  return reduce_graph_sequence(
      (graph_a, graph_b),
      context_fn=fn,
      edge_set_fn=fn,
      node_set_fn=fn)


def interpolate_graph_features(
    graph: tfgnn.GraphTensor,
    baseline: tfgnn.GraphTensor,
    *,
    steps: int) -> Sequence[tfgnn.GraphTensor]:
  """Return interpolations between a graph tensor and a baseline.

  `graph` and `baseline` must share a `tfgnn.GraphTensorSpec.`

  Args:
    graph: A `tfgnn.GraphTensor.`
    baseline: A `tfgnn.GraphTensor.`
    steps: The number of interpolations.

  Returns:
    A sequence of interpolation `tfgnn.GraphTensor.`
  """
  if graph.spec != baseline.spec:
    raise ValueError("subtract_graph_features() with graphs of different spec")

  alphas = np.linspace(start=0., stop=1., num=steps, endpoint=True)
  delta = subtract_graph_features(graph, baseline)

  def interpolate_(alpha):
    fn = lambda inputs: tf.math.multiply(*inputs, alpha)
    return sum_graph_features((
        baseline,
        reduce_graph_sequence(
            (delta,),
            context_fn=fn,
            edge_set_fn=fn,
            node_set_fn=fn),
    ))

  return tuple(interpolate_(a) for a in alphas)


def sum_graph_features(
    graphs: Sequence[tfgnn.GraphTensor]) -> tfgnn.GraphTensor:
  """Return a summation of a sequence of graph tesnors.

  The returned graph tensor contains for each `tfgnn.GraphTensor` in `graphs`:
  the sum between all pairs of edge set, feature name and all pairs of node set,
  feature name.

  All elements of `graphs` must share a `tfgnn.GraphTensorSpec.`

  Args:
    graphs: A sequence of `tfgnn.GraphTensor.`

  Returns:
    A `tfgnn.GraphTensor.`
  """
  return reduce_graph_sequence(
      graphs,
      context_fn=tf.math.add_n,
      edge_set_fn=tf.math.add_n,
      node_set_fn=tf.math.add_n)


TypeSpec = Union[tfgnn.GraphTensorSpec, tf.TensorSpec, tf.RaggedTensorSpec]


def _input_signature(model: tf.keras.Model) -> Sequence[TypeSpec]:
  if tf.nest.is_nested(model.input):
    return tf.nest.map_structure(operator.attrgetter("type_spec"), model.input)
  return (model.input.type_spec,)


def integrated_gradients(
    preprocess_model: tf.keras.Model,
    model: tf.keras.Model,
    *,
    output_name: Optional[str] = None,
    random_counterfactual: bool,
    steps: int,
    seed: Optional[int] = None) -> tf.types.experimental.ConcreteFunction:
  """Integrated gradients.

  This `tf.function` computes integrated gradients over a `tfgnn.GraphTensor.`
  The `tf.function` will be persisted in the ultimate saved model for
  subsequent attribution.

  Args:
    preprocess_model: A `tf.keras.Model` for preprocessing. This model is
      expected to return a tuple (`GraphTensor`, `Tensor`) where the
      `GraphTensor` is used to invoke the below `model` and the tensor is used
      used for any loss computation. (Via `model.compiled_loss`.)
    model: A `tf.keras.Model` for integrated gradients.
    output_name: The output `Tensor` name. If unset, the tensor will be named
      by Keras defaults.
    random_counterfactual: Whether to use a random uniform counterfactual.
    steps: The number of interpolations of the Riemann sum approximation.
    seed: An option random seed.

  Returns:
    A `tf.function` with the integrated gradients as output.
  """
  @tf.function(input_signature=_input_signature(preprocess_model))
  def fn(inputs):
    try:
      graph, labels = preprocess_model(inputs)
      if isinstance(graph, Sequence): graph, *_ = graph
    except ValueError as error:
      msg = "Integrated gradients requires both examples and labels"
      raise ValueError(msg) from error
    else:
      tfgnn.check_scalar_graph_tensor(graph, name="integrated_gradients")

    baseline = counterfactual(graph, random=random_counterfactual, seed=seed)
    interpolations = interpolate_graph_features(graph, baseline, steps=steps)
    gradients = []

    for interpolation in interpolations:
      with tf.GradientTape(persistent=True) as tape:
        tape.watch(interpolation)
        logits = model(interpolation)
        loss = model.compiled_loss(
            labels,
            logits,
            regularization_losses=model.losses)

      def gradient_fn(inputs):
        return tape.gradient(  # pylint: disable=cell-var-from-loop
            loss,  # pylint: disable=cell-var-from-loop
            *inputs,
            unconnected_gradients=tf.UnconnectedGradients.ZERO)

      gradients += [
          reduce_graph_sequence(
              (interpolation,),
              context_fn=gradient_fn,
              edge_set_fn=gradient_fn,
              node_set_fn=gradient_fn)
      ]

    gradients = sum_graph_features(gradients)
    return {output_name: gradients} if output_name is not None else gradients

  return fn


class IntegratedGradientsExporter(interfaces.ModelExporter):
  """Exports a Keras model with an additional integrated gradients signature."""

  # TODO(b/196880966): Support specifying IG and serving default output names.
  def __init__(self,
               integrated_gradients_output_name: Optional[str] = None,
               subdirectory: Optional[str] = None,
               random_counterfactual: bool = True,
               steps: int = 32,
               seed: Optional[int] = None,
               options: Optional[tf.saved_model.SaveOptions] = None):
    """Captures the args shared across `save(...)` calls.

    Random counterfactuals (see `random_counterfactual` below) sample from, per
    graph piece and feature, a uniform distribution bounded by the minimum
    (inclusive) and maximum (exclusive) of that graph piece instance's features.
    For integer features, the maximum is adjust to maximum += 1. Uses of
    integrated gradients on inputs* with many uniform (e.g.: a default value)
    or near uniform (e.g.: a categorical value with few states) features should
    consider i) not use a random counterfactual or ii) omitting transformations
    like one-hot encoding in any preprocessing.

    *) Where inputs are: the inputs of the `trained_model` (which correspond to
       the outputs of the `preprocess_model`).

    Args:
      integrated_gradients_output_name: The name for the integrated gradients
        output tensor. If unset, the tensor will be named by Keras defaults.
      subdirectory: An optional subdirectory, if set: models are exported to
        `os.path.join(export_dir, subdirectory).`
      random_counterfactual: Whether to use a random uniform counterfactual.
      steps: The number of interpolations of the Riemann sum approximation.
      seed: An optional random seed.
      options: Options for saving to SavedModel.
    """
    self._integrated_gradients_output_name = integrated_gradients_output_name
    self._subdirectory = subdirectory
    self._random_counterfactual = random_counterfactual
    self._steps = steps
    self._seed = seed
    self._options = options

  def save(self, run_result: interfaces.RunResult, export_dir: str):
    """Exports a Keras model with an additional integrated gradients signature.

    Importantly: the `run_result.preprocess_model`, if provided, and
    `run_result.trained_model` are stacked before any export. Stacking involves
    the chaining of the first output of `run_result.preprocess_model` to the
    only input of `run_result.trained_model.` The result is a model with the
    input of `run_result.preprocess_model` and the output of
    `run_result.trained_model.`

    Two serving signatures are exported:

    'serving_default') The default serving signature (i.e., the
      `preprocess_model` input signature),
    'integrated_gradients') The integrated gradients signature (i.e., the
      `preprocess_model` input signature).

    Args:
      run_result: A `RunResult` from training.
      export_dir: A destination directory.
    """
    preprocess_model = run_result.preprocess_model
    model = run_result.trained_model

    if preprocess_model is None:
      raise ValueError("Integrated gradients requires a `preprocess_model.`")
    elif not preprocess_model.built:
      raise ValueError("`preprocess_model` is expected to have been built")
    elif not model.built:
      raise ValueError("`model` is expected to have been built")

    xs, *_ = preprocess_model.output
    model_for_export = tf.keras.Model(preprocess_model.input, model(xs))

    ig = integrated_gradients(
        preprocess_model,
        model,
        output_name=self._integrated_gradients_output_name,
        random_counterfactual=self._random_counterfactual,
        steps=self._steps,
        seed=self._seed)
    serving_default = tf.function(
        model_for_export,
        input_signature=_input_signature(model_for_export))

    signatures = {
        "integrated_gradients": ig,
        "serving_default": serving_default,
    }

    if self._subdirectory:
      export_dir = os.path.join(export_dir, self._subdirectory)

    tf.keras.models.save_model(
        model_for_export,
        export_dir,
        signatures=signatures,
        options=self._options)
