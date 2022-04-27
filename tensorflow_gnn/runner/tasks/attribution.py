"""Task for integrated gradients attribution method.

This task will wrap any `tfgnn.GraphTensor` model with another `tf.keras.Model`
model that provides an additional `tf.function` for computing integrated
gradients by Riemann sum approximation.

The task implements the method as described in:
https://papers.nips.cc/paper/2020/hash/417fbbf2e9d5a28a855a11894b2e795a-Abstract.html.
"""
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_gnn as tfgnn


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
    fn = lambda inputs: tf.random.uniform(  # pylint: disable=g-long-lambda
        tf.shape(*inputs),
        minval=tf.math.reduce_min(*inputs),
        maxval=tf.math.reduce_max(*inputs),
        dtype=inputs[0].dtype,
        seed=seed)
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


# TODO(b/196880966): Link a colab to visualize any integrated gradients.
class ModelWithIntegratedGradients(tf.keras.Model):
  """`tf.keras.Model` wrapper that provides integrated gradients."""

  def __init__(self, model: tf.keras.Model):
    super(ModelWithIntegratedGradients, self).__init__(
        inputs=model.inputs,
        outputs=model.outputs)

  @tf.function
  def integrated_gradients(
      self,
      inputs: Tuple[tfgnn.GraphTensor, tfgnn.Field],
      *,
      random_counterfactual: bool = True,
      steps: int = 32,
      seed: Optional[int] = None) -> tfgnn.GraphTensor:
    """Integrated gradients.

    This `tf.function` computes integrated gradients over a `tfgnn.GraphTensor.`
    The `tf.function` will be persisted in the ultimate saved model for
    subsequent attribution.

    Args:
      inputs: Model inputs: a `tfgnn.GraphTensor` and labels.
      random_counterfactual: Whether to use a random uniform counterfactual.
      steps: The number of interpolations of the Riemann sum approximation.
      seed: An option random seed.

    Returns:
      A `tfgnn.GraphTensor` with a the integreated gradients.
    """
    graph, labels = inputs

    if graph.rank != 0:
      raise ValueError(
          f"Expected a scalar (rank=0) GraphTensor, received rank={graph.rank}")

    baseline = counterfactual(graph, random=random_counterfactual, seed=seed)
    interpolations = interpolate_graph_features(graph, baseline, steps=steps)
    gradients = []

    for interpolation in interpolations:
      with tf.GradientTape(persistent=True) as tape:
        tape.watch(interpolation)
        logits = self(interpolation)
        loss = self.compiled_loss(
            logits,
            labels,
            regularization_losses=self.losses)

      def fn(inputs):
        return tape.gradient(  # pylint: disable=cell-var-from-loop
            loss,  # pylint: disable=cell-var-from-loop
            *inputs,
            unconnected_gradients=tf.UnconnectedGradients.ZERO)

      gradients += [
          reduce_graph_sequence(
              (interpolation,),
              context_fn=fn,
              edge_set_fn=fn,
              node_set_fn=fn)
      ]

    return sum_graph_features(gradients)


class IntegratedGradients():
  """Integrated gradients task.

  This task wraps any `tfgnn.GraphTensor` model in another `tf.keras.Model` that
  provides a `tf.function` for computing integrated gradients.
  """

  def adapt(self, model: tf.keras.Model) -> tf.keras.Model:
    return ModelWithIntegratedGradients(model)

  def preprocessors(self) -> Sequence[Callable[..., tf.data.Dataset]]:
    return ()

  def losses(self) -> Sequence[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]:
    return ()

  def metrics(self) -> Sequence[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]:
    return ()
