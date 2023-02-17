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
"""Tests for losses."""
import functools
from typing import Any
from absl.testing import parameterized

import tensorflow as tf
from tensorflow_gnn.models.contrastive_losses import losses


def all_losses():
  return [
      dict(
          testcase_name="VicReg",
          loss_fn=losses.vicreg_loss,
      ),
      dict(
          testcase_name="BarlowTwins",
          loss_fn=losses.barlow_twins_loss,
      ),
      dict(
          testcase_name="BarlowTwinsNoBN",
          loss_fn=functools.partial(
              losses.barlow_twins_loss, normalize_batch=False
          ),
      ),
  ]


class CommonConstrastiveLossesTests(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(all_losses())
  def test_loss_shape(self, loss_fn: Any):
    """Verifies output logits shape."""
    x_clean = tf.random.uniform((2, 5))
    x_corrupted = tf.random.uniform((2, 5))
    loss_value = loss_fn(x_clean, x_corrupted)

    # Loss is a scalar.
    self.assertEqual(loss_value.shape, ())

  @parameterized.named_parameters(all_losses())
  def test_error_if_different_shape(self, loss_fn: Any):
    """Verifies the error if two embedding have different dimensionalities."""
    x_clean = tf.random.uniform((1, 4))
    x_corrupted = tf.random.uniform((1, 5))
    with self.assertRaisesRegex(
        tf.errors.InvalidArgumentError, ".*Incompatible shapes: .*"
    ):
      _ = loss_fn(x_clean, x_corrupted)


class BarlowTwinsLossTest(tf.test.TestCase, parameterized.TestCase):

  def test_loss_value(self):
    """Verifies outputs in simple cases."""
    x_clean = tf.random.uniform((1, 4))
    x_corrupted = tf.random.uniform((1, 4))
    loss_value = losses.barlow_twins_loss(x_clean, x_corrupted)

    # Loss is equal to dimensionality in case with normalization and 1D inputs.
    self.assertAllClose(loss_value, 4.0)

    x = tf.random.uniform((3, 4))
    loss_value = losses.barlow_twins_loss(x, x, lambda_=0)
    # Loss is equal to 0 for lambda=0 and orthonormal inputs.
    self.assertAllClose(loss_value, 0.0)

    x_clean = tf.convert_to_tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    x_corrupted = tf.convert_to_tensor([[1.0, 0.0, 1.0], [-1.0, 0.0, 0.0]])
    # In this example, the loss matrix is:
    # [ 1  ,  0,  0.5]
    # [-0.5,  1,  0  ]
    # [-0.5,  0,  1  ]
    # Then, the first component of the loss is the trace of loss squared = 3.
    # Therefore, we expect loss with lambda=0 to be 3.
    loss_value = losses.barlow_twins_loss(
        x_clean, x_corrupted, lambda_=0, normalize_batch=False
    )
    self.assertAllClose(loss_value, 3.)

    # The second component of the loss is the element-wise square of the matrix,
    # which is 3*(1^2) + 3*(0.5^2) = 3.75
    loss_value = losses.barlow_twins_loss(
        x_clean, x_corrupted, lambda_=1, normalize_batch=False
    )
    self.assertAllClose(loss_value, 3.75)


class VicRegLossTest(tf.test.TestCase, parameterized.TestCase):

  def test_loss_value(self):
    """Verifies outputs in simple cases."""
    x_clean = tf.random.uniform((1, 4))
    x_corrupted = tf.random.uniform((1, 4))
    loss_value = losses.vicreg_loss(
        x_clean, x_corrupted, sim_weight=0, var_weight=0
    )

    # Loss is equal to 0 for covariance loss for 1D inputs.
    self.assertAllClose(loss_value, 0)

    loss_value = losses.vicreg_loss(
        x_clean, x_clean, var_weight=0, cov_weight=0
    )
    # Loss is equal to 0 for the same inputs (L2 loss=0).
    self.assertAllClose(loss_value, 0)

    x_clean = losses._normalize(x_clean)
    loss_value = losses.vicreg_loss(
        x_clean, x_clean, sim_weight=0, var_weight=1, cov_weight=0
    )
    # Loss is equal to 0 for the same inputs (L2 loss=0).
    # 1e-4 factor is from the epsilon parameter of the loss
    self.assertAllClose(loss_value, 2 - 2 * tf.sqrt(1e-4))

    x_clean = tf.convert_to_tensor([[0.0, -1.0, -1.0], [1.0, 1.0, 1.0]])
    x_corrupted = tf.convert_to_tensor([[1.0, 1.0, 1.0], [-1.0, 0.0, -1.0]])
    # In this example, the squared difference is:
    # [1, 4, 4]
    # [4, 1, 4]
    # Then, the first component of the loss is the mean of the squared
    # differences. In the current example (4*4+2)/6 = 3.
    loss_value = losses.vicreg_loss(
        x_clean, x_corrupted, sim_weight=1, var_weight=0, cov_weight=0
    )
    self.assertAllClose(loss_value, 3.)

    # Both x_clean and x_corrupted are designed to be orthogonal when the
    # features are normalized. Therefore, both covariance losses are expected to
    # be 1. Since they are summed up, we expect 2 as the final answer.
    # Note that we test var_weight separately in a randomized test.
    loss_value = losses.vicreg_loss(
        x_clean, x_corrupted, sim_weight=0, var_weight=0, cov_weight=1
    )
    self.assertAllClose(loss_value, 2)


if __name__ == "__main__":
  tf.test.main()
