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
"""Unsupervised losses implementations as tensor-to-tensor functions."""
from typing import Optional, Union

import tensorflow as tf


def _variance_loss(representations: tf.Tensor, eps: float = 1e-4) -> tf.Tensor:
  """Variance loss component of VicReg loss.

  Computes truncated per-dimension standard deviation of input representations.
  For the purpose of the loss, standard deviation of 1 or more is optimal.

  Args:
    representations: Input representations.
    eps: Epsilon for the standard deviation computation.

  Returns:
    Loss value as scalar `tf.Tensor`.
  """
  std = tf.math.sqrt(tf.math.reduce_variance(representations, axis=0) + eps)
  return tf.reduce_mean(tf.nn.relu(1 - std))


def _covariance_loss(representations: tf.Tensor) -> tf.Tensor:
  """Covariance loss component of VicReg loss.

  Computes normalized square of the off-diagonal elements of the covariance
  matrix of input representations.

  Args:
    representations: Input representations.

  Returns:
    Loss value as scalar `tf.Tensor`.
  """
  batch_size, feature_dim = tf.unstack(
      tf.cast(tf.shape(representations), tf.float32)
  )
  representations = _normalize(representations, scale=False)
  covariance = (
      tf.matmul(representations, representations, transpose_a=True) / batch_size
  )
  covariance = tf.pow(covariance, 2)
  return (
      tf.reduce_sum(
          tf.linalg.set_diag(
              covariance, tf.zeros(tf.cast(feature_dim, tf.int32))
          )
      )
      / feature_dim
  )


def _normalize(
    representations: tf.Tensor, *, scale=True, epsilon: float = 1e-6
) -> tf.Tensor:
  """Standardizes the representations across the first (batch) dimension.

  Args:
    representations: Representations to normalize.
    scale: Whether to scale representations by the standard deviation. If
      `False`, simply remove the mean from `features`. Default: `True`.
    epsilon: Numerical epsilon to avoid division by zero.

  Returns:
    A `tf.Tensor` with representations normalized to zero mean and
    (subject to `scale`) unit variance.
  """
  representations_mean = tf.reduce_mean(representations, axis=0)
  if scale:
    # Avoid reduce_std(), which started to produce NaN in gradients
    # after TF2.13, see b/281569559.
    representations_variance = tf.math.reduce_variance(representations, axis=0)
    return tf.math.divide_no_nan(
        representations - representations_mean,
        tf.math.sqrt(representations_variance + epsilon),
    )
  else:
    return representations - representations_mean


def vicreg_loss(
    representations_clean: tf.Tensor,
    representations_corrupted: tf.Tensor,
    *,
    sim_weight: Union[tf.Tensor, float] = 25.0,
    var_weight: Union[tf.Tensor, float] = 25.0,
    cov_weight: Union[tf.Tensor, float] = 1.0,
) -> tf.Tensor:
  """VICReg loss implementation.

  Implements VICReg loss from the paper https://arxiv.org/abs/2105.04906.

  Args:
    representations_clean: Representations from the clean view of the data.
    representations_corrupted: Representations from the corrupted view of the
      data.
    sim_weight: Weight of the invariance (similarity) loss component of the
      VICReg loss.
    var_weight: Weight of the variance loss component of the VICReg loss.
    cov_weight: Weight of the covariance loss component of the VICReg loss.

  Returns:
    VICReg loss value as `tf.Tensor`.
  """
  losses = []
  if tf.get_static_value(sim_weight) != 0.0:
    losses.append(
        sim_weight
        * tf.math.reduce_mean(
            tf.math.squared_difference(
                representations_clean, representations_corrupted
            )
        )
    )
  if tf.get_static_value(var_weight) != 0.0:
    losses.append(var_weight * _variance_loss(representations_clean))
    losses.append(var_weight * _variance_loss(representations_corrupted))
  if tf.get_static_value(cov_weight) != 0.0:
    losses.append(cov_weight * _covariance_loss(representations_clean))
    losses.append(cov_weight * _covariance_loss(representations_corrupted))
  return tf.add_n(losses)


def barlow_twins_loss(
    representations_clean: tf.Tensor,
    representations_corrupted: tf.Tensor,
    *,
    lambda_: Optional[Union[tf.Tensor, float]] = None,
    normalize_batch: bool = True,
) -> tf.Tensor:
  """Barlow Twins loss implementation.

  Implements BarlowTwins loss from the paper https://arxiv.org/abs/2103.03230.
  Note that BarlowTwins loss scale is sensitive to the batch size.

  Args:
    representations_clean: Representations from the clean view of the data.
    representations_corrupted: Representations from the corrupted view of the
      data.
    lambda_: Parameter lambda of the BarlowTwins model. If None (default), uses
      `1 / feature_dim`.
    normalize_batch: If `True` (default), normalizes representations per-batch.

  Returns:
    BarlowTwins loss value as `tf.Tensor`.
  """
  if normalize_batch:
    representations_clean = _normalize(representations_clean)
    representations_corrupted = _normalize(representations_corrupted)
  batch_size, feature_dim = tf.unstack(
      tf.cast(tf.shape(representations_clean), tf.float32)
  )
  lambda_ = 1 / feature_dim if lambda_ is None else lambda_
  correlation = (
      tf.linalg.matmul(
          representations_clean, representations_corrupted, transpose_a=True
      )
      / batch_size
  )

  loss_matrix = tf.pow(correlation - tf.eye(feature_dim), 2)
  loss_diagonal_sum = tf.linalg.trace(loss_matrix)
  loss_sum = tf.reduce_sum(loss_matrix)
  return (1.0 - lambda_) * loss_diagonal_sum + lambda_ * loss_sum
