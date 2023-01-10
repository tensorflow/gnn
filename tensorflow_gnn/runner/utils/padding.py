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
"""Helpers for size constraints."""
import abc
import functools
from typing import Callable, Mapping, Optional

import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn.runner import interfaces
from tensorflow_gnn.runner.utils import parsing as parsing_utils

SizeConstraints = tfgnn.SizeConstraints


def one_node_per_component(gtspec: tfgnn.GraphTensorSpec) -> Mapping[str, int]:
  return {k: 1 for k in gtspec.node_sets_spec.keys()}


class _GraphTensorPadding(interfaces.GraphTensorPadding):
  """Calculates `SizeConstraints` for `GraphTensor` padding."""

  def __init__(
      self,
      gtspec: tfgnn.GraphTensorSpec,
      dataset_provider: interfaces.DatasetProvider,
      min_nodes_per_component: Optional[Mapping[str, int]] = None):
    self._gtspec = gtspec
    self._dataset_provider = dataset_provider
    if min_nodes_per_component is None:
      # For readout at least one node per component must be present: we do
      # not know the readout node set a priori.
      self._min_nodes_per_component = one_node_per_component(gtspec)
    else:
      self._min_nodes_per_component = dict(min_nodes_per_component)

  @abc.abstractmethod
  def get_filter_fn(self,
                    size_constraints: SizeConstraints) -> Callable[..., bool]:
    raise NotImplementedError()

  @abc.abstractmethod
  def get_size_constraints(self, target_batch_size: int) -> SizeConstraints:
    raise NotImplementedError()


class FitOrSkipPadding(_GraphTensorPadding):
  """Calculates fit or skip `SizeConstraints` for `GraphTensor` padding.

  See: `tfgnn.learn_fit_or_skip_size_constraints.`
  """

  def __init__(
      self,
      gtspec: tfgnn.GraphTensorSpec,
      dataset_provider: interfaces.DatasetProvider,
      min_nodes_per_component: Optional[Mapping[str, int]] = None,
      fit_or_skip_sample_sample_size: int = 10_000,
      fit_or_skip_success_ratio: float = 0.99):
    super().__init__(gtspec, dataset_provider, min_nodes_per_component)

    self._fit_or_skip_sample_sample_size = fit_or_skip_sample_sample_size
    self._fit_or_skip_success_ratio = fit_or_skip_success_ratio

  def get_filter_fn(self,
                    size_constraints: SizeConstraints) -> Callable[..., bool]:
    return functools.partial(
        tfgnn.satisfies_size_constraints,
        total_sizes=size_constraints)

  def get_size_constraints(self, target_batch_size: int) -> SizeConstraints:
    dataset = self._dataset_provider.get_dataset(tf.distribute.InputContext())
    return tfgnn.learn_fit_or_skip_size_constraints(  # pytype: disable=bad-return-type
        parsing_utils.maybe_parse_graph_tensor_dataset(dataset, self._gtspec),
        target_batch_size,
        min_nodes_per_component=self._min_nodes_per_component,
        sample_size=self._fit_or_skip_sample_sample_size,
        success_ratio=self._fit_or_skip_success_ratio)


class TightPadding(_GraphTensorPadding):
  """Calculates tight `SizeConstraints` for `GraphTensor` padding.

  See: `tfgnn.find_tight_size_constraints.`
  """

  def get_filter_fn(self,
                    size_constraints: SizeConstraints) -> Callable[..., bool]:
    return lambda *args, **kwargs: True

  def get_size_constraints(self, target_batch_size: int) -> SizeConstraints:
    dataset = self._dataset_provider.get_dataset(tf.distribute.InputContext())
    return tfgnn.find_tight_size_constraints(
        parsing_utils.maybe_parse_graph_tensor_dataset(dataset, self._gtspec),
        min_nodes_per_component=self._min_nodes_per_component,
        target_batch_size=target_batch_size)
