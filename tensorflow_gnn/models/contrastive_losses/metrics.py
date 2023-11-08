# Copyright 2023 The TensorFlow GNN Authors. All Rights Reserved.
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
"""Metrics for unsupervised embedding evaluation."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Optional, Protocol

import tensorflow as tf


@tf.function
def self_clustering(
    representations: tf.Tensor, *, subtract_mean: bool = False, **_
) -> tf.Tensor:
  """Self-clustering metric implementation.

  Computes a metric that measures how well distributed representations are, if
  projected on the unit sphere. If `subtract_mean` is True, we additionally
  remove the mean from representations. The metric has a range of
  (-0.5, 1]. It achieves its maximum of 1 if representations collapse to a
  single point, and it is approximately 0 if representations are distributed
  randomly on the sphere. In theory, it can achieve negative values if the
  points are maximally equiangular, although this is very rare in practice.
  Refer to https://arxiv.org/abs/2305.16562 for more details.

  Args:
    representations: Input representations.
    subtract_mean: Whether to subtract the mean from representations.

  Returns:
    Metric value as scalar `tf.Tensor`.
  """
  if representations.shape.rank != 2:
    raise ValueError(f"Expected 2D tensor (got shape {representations.shape})")
  if subtract_mean:
    representations -= tf.reduce_mean(representations, axis=0)
  representations, _ = tf.linalg.normalize(representations, ord=2, axis=1)
  batch_size, feature_dim = tf.unstack(
      tf.cast(tf.shape(representations), tf.float32)
  )
  if tf.math.reduce_all(tf.math.is_nan(representations)):
    # If representations are completely collapsed, return 1.
    return tf.constant(1.0)
  expected = batch_size + batch_size * (batch_size - 1) / feature_dim
  actual = tf.reduce_sum(
      tf.square(tf.matmul(representations, representations, transpose_b=True))
  )
  return (actual - expected) / (batch_size * batch_size - expected)


@tf.function
def pseudo_condition_number(
    representations: tf.Tensor,
    *,
    sigma: Optional[tf.Tensor] = None,
    u: Optional[tf.Tensor] = None,
) -> tf.Tensor:
  """Pseudo-condition number metric implementation.

  Computes a metric that measures the decay rate of the singular values.
  NOTE: Can be unstable in practice, when using small batch sizes, leading
  to numerical instabilities.

  Args:
    representations: Input representations. We expect rank 2 input.
    sigma: An optional tensor with singular values of representations. If not
      present, computes SVD (singular values only) of representations.
    u: Unused.

  Returns:
    Metric value as scalar `tf.Tensor`.
  """
  del u
  if representations.shape.rank != 2:
    raise ValueError(f"Expected 2D tensor (got shape {representations.shape})")
  if sigma is None:
    sigma = tf.linalg.svd(representations, compute_uv=False)
  return tf.math.divide_no_nan(sigma[0], sigma[-1])


@tf.function
def numerical_rank(
    representations: tf.Tensor,
    *,
    sigma: Optional[tf.Tensor] = None,
    u: Optional[tf.Tensor] = None,
) -> tf.Tensor:
  """Numerical rank implementation.

  Computes a metric that estimates the numerical column rank of a matrix.
  Rank is estimated as a matrix trace divided by the largest eigenvalue. When
  our matrix is a covariance matrix, we can compute both the trace and the
  largest eigenvalue efficiently via SVD as the largest singular value squared.

  Args:
    representations: Input representations. We expect rank 2 input.
    sigma: An optional tensor with singular values of representations. If not
      present, computes SVD (singular values only) of representations.
    u: Unused.

  Returns:
    Metric value as scalar `tf.Tensor`.
  """
  del u
  if representations.shape.rank != 2:
    raise ValueError(f"Expected 2D tensor (got shape {representations.shape})")
  if sigma is None:
    sigma = tf.linalg.svd(representations, compute_uv=False)
  trace = tf.reduce_sum(tf.math.square(representations))
  denominator = tf.math.square(sigma[0])
  return tf.math.divide_no_nan(trace, denominator)


@tf.function
def rankme(
    representations: tf.Tensor,
    *,
    sigma: Optional[tf.Tensor] = None,
    u: Optional[tf.Tensor] = None,
    epsilon: float = 1e-12,
    **_,
) -> tf.Tensor:
  """RankMe metric implementation.

  Computes a metric that measures the decay rate of the singular values.
  For the paper, see https://arxiv.org/abs/2210.02885.

  Args:
    representations: Input representations as rank-2 tensor.
    sigma: An optional tensor with singular values of representations. If not
      present, computes SVD (singular values only) of representations.
    u: Unused.
    epsilon: Epsilon for numerican stability.

  Returns:
    Metric value as scalar `tf.Tensor`.
  """
  del u
  if representations.shape.rank != 2:
    raise ValueError(f"Expected 2D tensor (got shape {representations.shape})")
  if sigma is None:
    sigma = tf.linalg.svd(representations, compute_uv=False)
  return rankme_from_singular_values(sigma, epsilon)


def rankme_from_singular_values(
    sigma: tf.Tensor, epsilon: float = 1e-12
) -> tf.Tensor:
  """RankMe metric implementation.

  Computes a metric that measures the decay rate of the singular values.
  For the paper, see https://arxiv.org/abs/2210.02885.

  Args:
    sigma: Singular values of the input representations as rank-1 tensor.
    epsilon: Epsilon for numerican stability.

  Returns:
    Metric value as scalar `tf.Tensor`.
  """
  if sigma.shape.rank != 1:
    raise ValueError(f"Expected 1D tensor (got shape {sigma.shape})")
  tf.debugging.assert_non_negative(
      sigma,
      message="All singular values must be non-negative.",
  )
  p_ks = tf.math.divide_no_nan(sigma, tf.math.reduce_sum(sigma)) + epsilon
  return tf.math.exp(-tf.math.reduce_sum(p_ks * tf.math.log(p_ks)))


@tf.function
def coherence(
    representations: tf.Tensor,
    *,
    sigma: Optional[tf.Tensor] = None,
    u: Optional[tf.Tensor] = None,
) -> tf.Tensor:
  """Coherence metric implementation.

  Coherence measures how easy it is to construct a linear classifier on top of
  data without knowing downstream labels.
  Refer to https://arxiv.org/abs/2305.16562 for more details.

  Args:
    representations: Input representations, a rank-2 tensor.
    sigma: Unused.
    u: An optional tensor with left singular vectors of representations. If not
      present, computes a SVD of representations.

  Returns:
    Metric value as scalar `tf.Tensor`.
  """
  del sigma
  if representations.shape.rank != 2:
    raise ValueError(f"Expected 2D tensor (got shape {representations.shape})")
  if u is None:
    _, u, _ = tf.linalg.svd(
        representations, compute_uv=True, full_matrices=False
    )
  return coherence_from_singular_vectors(u)


@tf.function
def coherence_from_singular_vectors(u: tf.Tensor) -> tf.Tensor:
  """Coherence metric implementation.

  Refer to https://arxiv.org/abs/2305.16562 for more details.

  Args:
    u: Left singular vectors of representations, a rank-2 tensor.

  Returns:
    Metric value as scalar `tf.Tensor`.
  """
  if u.shape.rank != 2:
    raise ValueError(f"Expected 2D tensor (got shape {u.shape})")
  n_examples, dimensions = tf.unstack(tf.shape(u))
  maximum_norm = tf.math.reduce_max(tf.norm(u, axis=1))
  return tf.square(maximum_norm) * tf.cast(
      tf.divide(n_examples, dimensions), tf.float32
  )


@tf.keras.utils.register_keras_serializable()
class TripletLossMetrics(tf.keras.metrics.Metric):
  """Triplet loss metrics."""

  def __init__(self, name="triplet_loss_metrics", **kwargs):
    super().__init__(name=name, **kwargs)
    self.positive_distance_metric = tf.keras.metrics.Mean(
        name="positive_distance_metric"
    )
    self.negative_distance_metric = tf.keras.metrics.Mean(
        name="negative_distance_metric"
    )
    self.triplet_distance_metric = tf.keras.metrics.Mean(
        name="triplet_distance_metric"
    )

  def update_state(self, y_true, y_pred, sample_weight=None):
    del sample_weight
    y_pred = tf.ensure_shape(y_pred, (None, 2))
    positive_distance, negative_distance = tf.unstack(y_pred, axis=1)
    self.positive_distance_metric.update_state(positive_distance)
    self.negative_distance_metric.update_state(negative_distance)
    self.triplet_distance_metric.update_state(
        positive_distance - negative_distance
    )

  def result(self):
    return {
        "positive_distance": self.positive_distance_metric.result(),
        "negative_distance": self.negative_distance_metric.result(),
        "triplet_distance": self.triplet_distance_metric.result(),
    }

  def reset_state(self):
    self.positive_distance_metric.reset_states()
    self.negative_distance_metric.reset_states()
    self.triplet_distance_metric.reset_states()


class _SvdProtocol(Protocol):

  def __call__(
      self,
      representations: tf.Tensor,
      *,
      sigma: Optional[tf.Tensor] = None,
      u: Optional[tf.Tensor] = None,
  ) -> tf.Tensor:
    ...


class _SvdMetrics(tf.keras.metrics.Metric):
  """Computes multiple metrics for representations using one SVD call.

  Refer to https://arxiv.org/abs/2305.16562 for more details.
  """

  def __init__(
      self,
      fns: Mapping[str, _SvdProtocol],
      y_pred_transform_fn: Optional[
          Callable[[tf.Tensor], tf.Tensor]
      ] = None,
      name: str = "svd_metrics",
  ):
    """Constructs the `tf.keras.metrics.Metric` that reuses SVD decomposition.

    Args:
      fns: a mapping from a metric name to a `Callable` that accepts
        representations as well as the result of their SVD decomposition.
        Currently only singular values are passed.
      y_pred_transform_fn: a function to extract clean representations
        from model predictions. By default, no transformation is applied.
      name: Name for the metric class, used for Keras bookkeeping.
    """
    super().__init__(name=name)
    self._fns = fns
    self._metric_container = {
        k: tf.keras.metrics.Mean(name=k) for k in fns.keys()
    }
    if not y_pred_transform_fn:
      y_pred_transform_fn = lambda x: x
    self._y_pred_transform_fn = y_pred_transform_fn

  def reset_state(self) -> None:
    for v in self._metric_container.values():
      v.reset_state()

  def update_state(self, _, y_pred: tf.Tensor, sample_weight=None) -> None:
    representations = self._y_pred_transform_fn(y_pred)
    sigma, u, _ = tf.linalg.svd(
        representations, compute_uv=True, full_matrices=False
    )
    for k, v in self._metric_container.items():
      v.update_state(self._fns[k](representations, sigma=sigma, u=u))

  def result(self) -> Mapping[str, tf.Tensor]:
    return {k: v.result() for k, v in self._metric_container.items()}


class AllSvdMetrics(_SvdMetrics):

  def __init__(self, *args, **kwargs):
    super().__init__(
        *args,
        fns={
            "numerical_rank": numerical_rank,
            "rankme": rankme,
            "coherence": coherence,
        },
        **kwargs,
    )
