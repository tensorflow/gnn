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

import tensorflow as tf


@tf.function
def self_clustering(
    representations: tf.Tensor, *, subtract_mean: bool = False
) -> tf.Tensor:
  """Self-clustering metric implementation.

  Computes a metric that measures how well distributed representations are, if
  projected on the unit sphere. If `subtract_mean` is True, we additionally
  remove the mean from representations. The metric has a range of
  (-0.5, 1]. It achieves its maximum of 1 if representations collapse to a
  single point, and it is approximately 0 if representations are distributed
  randomly on the sphere. In theory, it can achieve negative values if the
  points are maximally equiangular, although this is very rare in practice.

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
    representations: tf.Tensor, *, subtract_mean: bool = True
) -> tf.Tensor:
  """Pseudo-condition number metric implementation.

  Computes a metric that measures the decay rate of the singular values.

  Args:
    representations: Input representations. We expect rank 2 input.
    subtract_mean: Whether to subtract the mean from representations.

  Returns:
    Metric value as scalar `tf.Tensor`.
  """
  if representations.shape.rank != 2:
    raise ValueError(f"Expected 2D tensor (got shape {representations.shape})")
  if subtract_mean:
    representations -= tf.reduce_mean(representations, axis=0)
  sigma = tf.linalg.svd(representations, compute_uv=False)
  return tf.math.divide_no_nan(sigma[0], sigma[-1])


@tf.function
def numerical_rank(representations: tf.Tensor) -> tf.Tensor:
  """Numerical rank implementation.

  Computes a metric that estimates the numerical column rank of a matrix.
  Rank is estimated as a matrix trace divided by the largest eigenvalue. When
  our matrix is a covariance matrix, we can compute both the trace and the
  largest eigenvalue efficiently via SVD as the largest singular value squared.

  Args:
    representations: Input representations. We expect rank 2 input.

  Returns:
    Metric value as scalar `tf.Tensor`.
  """
  if representations.shape.rank != 2:
    raise ValueError(f"Expected 2D tensor (got shape {representations.shape})")
  sigma = tf.linalg.svd(representations, compute_uv=False)
  trace = tf.reduce_sum(tf.math.square(representations))
  denominator = tf.math.square(sigma[0])
  return tf.math.divide_no_nan(trace, denominator)


@tf.function
def rankme(
    representations: tf.Tensor,
    *,
    epsilon: float = 1e-12,
) -> tf.Tensor:
  """RankMe metric implementation.

  Computes a metric that measures the decay rate of the singular values.
  For the paper, see https://arxiv.org/abs/2210.02885.

  Args:
    representations: Input representations as rank-2 tensor.
    epsilon: Epsilon for numerican stability.

  Returns:
    Metric value as scalar `tf.Tensor`.
  """
  if representations.shape.rank != 2:
    raise ValueError(f"Expected 2D tensor (got shape {representations.shape})")
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
def coherence(representations: tf.Tensor) -> tf.Tensor:
  """Coherence metric implementation.

  Coherence measures how easy it is to construct a linear classifier on top of
  data without knowing downstream labels.
  Refer to https://arxiv.org/abs/2305.16562 for more details.

  Args:
    representations: Input representations, a rank-2 tensor.

  Returns:
    Metric value as scalar `tf.Tensor`.
  """
  if representations.shape.rank != 2:
    raise ValueError(f"Expected 2D tensor (got shape {representations.shape})")
  _, u, _ = tf.linalg.svd(representations, compute_uv=True, full_matrices=False)
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
