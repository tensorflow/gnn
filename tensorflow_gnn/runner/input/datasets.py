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
"""`tf.data.Dataset` of `tfgnn.GraphTensor` methods."""
from typing import Callable, List, Optional, Sequence

import tensorflow as tf
from tensorflow_gnn.runner import interfaces


def _process_dataset(
    dataset: tf.data.Dataset,
    *,
    num_shards: int,
    index: int,
    shuffle_dataset: bool = False,
    interleave_fn: Callable[..., tf.data.Dataset],
    examples_shuffle_size: Optional[int] = None) -> tf.data.Dataset:
  """Implements `SimpleDatasetsProvider.get_dataset(...)`.

  Args:
    dataset: A `tf.data.Dataset` to process.
    num_shards: The number of shards operating in parallel.
    index: The worker index.
    shuffle_dataset: If enabled, shuffle the dataset before applying the
      `interleave_fn.` NOTE: if enabled, `dataset` must be of known, finite
      cardinality.
    interleave_fn: A function that takes a dataset element and returns a
      `tf.data.Dataset.`
    examples_shuffle_size: An optional buffer size for example shuffling.

  Returns:
    A processed `tf.data.Dataset.`
  """
  # Shard first before running any randomization operators (e.g. shuffle)
  dataset = dataset.shard(num_shards, index)
  if shuffle_dataset:
    dataset = dataset.shuffle(dataset.cardinality())
  dataset = dataset.interleave(
      interleave_fn,
      deterministic=False,
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  if examples_shuffle_size is not None:
    dataset = dataset.shuffle(examples_shuffle_size)
  return dataset.prefetch(tf.data.AUTOTUNE)


class PassthruDatasetProvider(interfaces.DatasetProvider):
  """Builds a `tf.data.Dataset` from a pass thru dataset.

  Passes any `dataset` thru: omitting any sharding. For detailed documentation,
  see the filename dataset provider complement: `SimpleDatasetsProvider.`
  """

  def __init__(self,
               dataset: tf.data.Dataset,
               *,
               shuffle_datasets: bool = False,
               examples_shuffle_size: Optional[int] = None):
    self._dataset = dataset
    self._shuffle_datasets = shuffle_datasets
    self._examples_shuffle_size = examples_shuffle_size

  def get_dataset(self, _: tf.distribute.InputContext) -> tf.data.Dataset:
    """Gets a `tf.data.Dataset` omitting any input context."""
    return _process_dataset(
        self._dataset,
        num_shards=1,
        index=0,
        shuffle_dataset=self._shuffle_datasets,
        interleave_fn=lambda x: x,
        examples_shuffle_size=self._examples_shuffle_size)


class SimpleDatasetProvider(interfaces.DatasetProvider):
  """Builds a `tf.data.Dataset` from a list of files.

  This `SimpleDatasetProvider` builds a `tf.data.Dataset` as follows:
   - The object is initialized with a list of filenames. For convenience,
     a file pattern can be specified instead, which will be expanded to a
     sorted list.
   - The filenames are sharded between replicas according to the `InputContext`
     (order matters).
   - Filenames are shuffled per replica (if requested).
   - The files in each shard are interleaved after being read by the
     `interleave_fn`.
   - Examples are shuffled (if requested), auto-prefetched, and returned for use
     in one replica of the trainer.
  """

  def __init__(self,
               file_pattern: Optional[str] = None,
               *,
               filenames: Optional[Sequence[str]] = None,
               shuffle_filenames: bool = False,
               interleave_fn: Callable[..., tf.data.Dataset],
               examples_shuffle_size: Optional[int] = None):
    """Captures the args shared across `get_dataset(...)` calls.

    Args:
      file_pattern: A file pattern, to be expanded by `tf.io.gfile.glob`
        and sorted into the list of all `filenames`.
      filenames: A list of all filenames, specified explicitly.
        This argument is mutually exclusive with `file_pattern`.
      shuffle_filenames: If enabled, filenames will be shuffled after sharding
        between replicas, before any file reads. Through interleaving, some
        files may be read in parallel: the details are auto-tuned for
        throughput.
      interleave_fn: A callback that receives a single filename and returns
        a `tf.data.Dataset` with the `tf.Example` values from that file.
      examples_shuffle_size: An optional buffer size for example shuffling.
    """
    self._file_pattern = file_pattern
    self._filenames = filenames
    if (self._file_pattern is not None) + (self._filenames is not None) != 1:
      raise ValueError(
          "Please provide either `_file_pattern` or `_filenames` argument, "
          "but not both.")
    self._shuffle_filenames = shuffle_filenames
    self._interleave_fn = interleave_fn
    self._examples_shuffle_size = examples_shuffle_size

  def get_dataset(self, context: tf.distribute.InputContext) -> tf.data.Dataset:
    """Gets a `tf.data.Dataset` by `context` per replica."""
    if self._file_pattern is not None:
      filenames = _sorted_glob_or_raise(self._file_pattern)
    else:
      filenames = self._filenames
    return _process_dataset(
        tf.data.Dataset.from_tensor_slices(filenames),
        num_shards=context.num_input_pipelines,
        index=context.input_pipeline_id,
        shuffle_dataset=self._shuffle_filenames,
        interleave_fn=self._interleave_fn,
        examples_shuffle_size=self._examples_shuffle_size)


def _sorted_glob_or_raise(file_pattern: str,
                          pattern_name="pattern") -> List[str]:
  filenames = tf.io.gfile.glob(file_pattern)
  if not filenames:
    raise FileNotFoundError(f"No files match {pattern_name} {file_pattern}")
  return sorted(filenames)


def _process_sampled_dataset(
    principal_dataset: tf.data.Dataset,
    extra_datasets: Sequence[tf.data.Dataset],
    principal_weight: Optional[float] = None,
    extra_weights: Optional[Sequence[float]] = None,
    *,
    num_shards: int = 1,
    index: int = 0,
    # TODO(b/196880966): `principal_cardinality` should be adjusted in the
    # presence of sharding (e.g., if `num_replicas` > 1 then
    # `principal_cardinality_per_shard` < `principal_cardinality`).
    principal_cardinality: Optional[int] = None,
    fixed_cardinality: bool = False,
    shuffle_dataset: bool = False,
    interleave_fn: Callable[..., tf.data.Dataset],
    examples_shuffle_size: Optional[int] = None) -> tf.data.Dataset:
  """Implements `SimpleSampleDatasetsProvider.get_dataset(...)`."""
  if extra_weights is not None and principal_weight is not None:
    weights = [*extra_weights, principal_weight]
  elif extra_weights is None and principal_weight is not None:
    raise ValueError("`extra_weights` required a `principal_weight`")
  elif extra_weights is not None and principal_weight is None:
    raise ValueError("`principal_weight` is required with `extra_weights`")
  else:
    weights = None

  if fixed_cardinality and principal_cardinality is None:
    msg = "`principal_cardinality` is required with `fixed_cardinality`"
    raise ValueError(msg)

  def dataset_fn(dataset):
    return _process_dataset(
        dataset,
        num_shards=num_shards,
        index=index,
        shuffle_dataset=shuffle_dataset,
        interleave_fn=interleave_fn,
        examples_shuffle_size=examples_shuffle_size)

  if fixed_cardinality:
    datasets = [
        *(dataset_fn(p).repeat() for p in extra_datasets),
        dataset_fn(principal_dataset).repeat()
    ]
    sampled_dataset = tf.data.Dataset.sample_from_datasets(
        datasets,
        weights=weights,
        stop_on_empty_dataset=False)
    weight = (principal_weight or 1 / len(datasets))
    count = int(principal_cardinality / weight)
    sampled_dataset = sampled_dataset.take(count)
  else:
    count = tf.int64.max
    datasets = [
        # `sample_from_datasets` produces a dataset that reports infinite
        # cardinality if any of the underlying datasets are of infinite
        # cardinality: repeat the datasets `tf.int64.max` times instead.
        *(dataset_fn(p).repeat(count) for p in extra_datasets),
        # If there are `weights`, the `principal_weight` comes last,
        # so the principal dataset has to be in the same place.
        dataset_fn(principal_dataset)
    ]
    sampled_dataset = tf.data.Dataset.sample_from_datasets(
        datasets,
        weights=weights,
        stop_on_empty_dataset=True)

  return sampled_dataset.prefetch(tf.data.AUTOTUNE)


class PassthruSampleDatasetsProvider(interfaces.DatasetProvider):
  """Builds a sampled `tf.data.Dataset` from multiple pass thru datasets.

  Passes any `principal_dataset` and `extra_datasets` thru: omitting any
  sharding. For detailed documentation, see the filename dataset provider
  complement: `SimpleSampleDatasetsProvider.`
  """

  def __init__(self,
               principal_dataset: tf.data.Dataset,
               extra_datasets: Sequence[tf.data.Dataset],
               principal_weight: Optional[float] = None,
               extra_weights: Optional[Sequence[float]] = None,
               *,
               principal_cardinality: Optional[int] = None,
               fixed_cardinality: bool = False,
               shuffle_dataset: bool = False,
               examples_shuffle_size: Optional[int] = None):
    self._principal_dataset = principal_dataset
    self._extra_datasets = extra_datasets
    self._principal_weight = principal_weight
    self._extra_weights = extra_weights
    self._principal_cardinality = principal_cardinality
    self._fixed_cardinality = fixed_cardinality
    self._shuffle_dataset = shuffle_dataset
    if examples_shuffle_size is not None:
      denominator = len(extra_datasets) + 1
      self._examples_shuffle_size = examples_shuffle_size // denominator
    else:
      self._examples_shuffle_size = None

  def get_dataset(self, _: tf.distribute.InputContext) -> tf.data.Dataset:
    """Gets a sampled `tf.data.Dataset` omitting any input context."""
    return _process_sampled_dataset(
        self._principal_dataset,
        self._extra_datasets,
        self._principal_weight,
        self._extra_weights,
        num_shards=1,
        index=0,
        principal_cardinality=self._principal_cardinality,
        fixed_cardinality=self._fixed_cardinality,
        shuffle_dataset=self._shuffle_dataset,
        interleave_fn=lambda x: x,
        examples_shuffle_size=self._examples_shuffle_size)


class SimpleSampleDatasetsProvider(interfaces.DatasetProvider):
  """Builds a sampling `tf.data.Dataset` from multiple filenames.

  For complete explanations regarding sampling see `_process_sampled_dataset()`.

  This `SimpleSampleDatasetsProvider` builds a `tf.data.Dataset` as follows:

   - The object is initialized with a list of filenames specified by
     `principle_filenames` and `extra_filenames` argument. For convenience,
     the corresponding file pattern `principal_file_pattern` and
     `extra_file_patterns` can be specified instead, which will be expanded to a
     sorted list.
   - The filenames are sharded between replicas according to the `InputContext`
     (order matters).
   - Filenames are shuffled per replica (if requested).
   - Examples from all file patterns are sampled according to `principal_weight`
     and `extra_weights.`
   - The files in each shard are interleaved after being read by the
     `interleave_fn`.
   - Examples are shuffled (if requested), auto-prefetched, and returned for
     use in one replica of the trainer.
  """

  def __init__(self,
               principal_file_pattern: Optional[str] = None,
               extra_file_patterns: Optional[Sequence[str]] = None,
               principal_weight: Optional[float] = None,
               extra_weights: Optional[Sequence[float]] = None,
               *,
               principal_filenames: Optional[Sequence[str]] = None,
               extra_filenames: Optional[Sequence[Sequence[str]]] = None,
               principal_cardinality: Optional[int] = None,
               fixed_cardinality: bool = False,
               shuffle_filenames: bool = False,
               interleave_fn: Callable[..., tf.data.Dataset],
               examples_shuffle_size: Optional[int] = None):
    """Captures the args shared across `get_dataset(...)` calls.

    Args:
      principal_file_pattern: A principal file pattern for sampling, to be
        expanded by `tf.io.gfile.glob` and sorted into the list of
        `principal_filenames`.
      extra_file_patterns: File patterns, to be expanded by `tf.io.gfile.glob`
        and sorted into the list of `extra_filenames`.
      principal_weight: An optional weight for the dataset corresponding to
        `principal_file_pattern.` Required iff `extra_weights` are also
        provided.
      extra_weights: Optional weights corresponding to `file_patterns` for
        sampling. Required iff `principal_weight` is also provided.
      principal_filenames: A list of principal filenames, specified explicitly.
        This argument is mutually exclusive with `principal_file_pattern`.
      extra_filenames: A list of extra filenames, specified explicitly.
        This argument is mutually exclusive with `extra_file_patterns`.
      principal_cardinality: Iff `fixed_cardinality`=True, the size of the
        returned dataset is computed as `principal_cardinality` /
        `principal_weight` (with a default of uniform weights).
      fixed_cardinality: Whether to take a fixed number of elements.
      shuffle_filenames: If enabled, filenames will be shuffled after sharding
        between replicas, before any file reads. Through interleaving, some
       files may be read in parallel: the details are auto-tuned for throughput.
      interleave_fn: A fn applied with `tf.data.Dataset.interleave.`
      examples_shuffle_size: An optional buffer size for example shuffling. If
        specified, the size is adjusted to `shuffle_size //
        (len(file_patterns) + 1).`
    """
    self._principal_file_pattern = principal_file_pattern
    self._extra_file_patterns = extra_file_patterns
    self._principal_filenames = principal_filenames
    self._extra_filenames = extra_filenames
    if (self._principal_file_pattern is not None) + (self._principal_filenames
                                                     is not None) != 1:
      raise ValueError(
          "Please provide either `_principal_file_pattern` or "
          "`_principal_filenames` argument, but not both.")
    self._principal_weight = principal_weight
    self._extra_weights = extra_weights
    self._principal_cardinality = principal_cardinality
    self._fixed_cardinality = fixed_cardinality
    self._shuffle_filenames = shuffle_filenames
    self._interleave_fn = interleave_fn
    if examples_shuffle_size is not None:
      denominator = len(extra_file_patterns) + 1
      self._examples_shuffle_size = examples_shuffle_size // denominator
    else:
      self._examples_shuffle_size = None

  def get_dataset(self, context: tf.distribute.InputContext) -> tf.data.Dataset:
    """Creates a `tf.data.Dataset` by sampling.

    The contents of the resulting `tf.data.Dataset` are sampled from several
    sources, each stored as a sharded dataset:
      * one principal input, whose size determines the size of the resulting
        `tf.data.Dataset`;
      * zero or more side inputs, which are repeated if necessary to preserve
        the requested samping weights.

    Each input dataset is shared before interleaving. The result of interleaving
    is only shuffled if a `examples_shuffle_size` is provided.

    Datasets are sampled from with `tf.data.Dataset.sample_from_datasets.` For
    sampling details, please refer to the TensorFlow documentation at:
    https://www.tensorflow.org/api_docs/python/tf/data/Dataset#sample_from_datasets.

    Two methods are supported to determine the end of the resulting
    `tf.data.Dataset`:

    fixed_cardinality=True) Returns a dataset with a fixed cardinality, set at
      `principal_cardinality` // `principal_weight.` `principal_dataset` and
      `principal_cardinality` are required for this method. `principal_weight`
      is required iff `extra_weights` are also provided.

    fixed_cardinality=False) Returns a dataset that ends after the principal
      input has been exhausted, subject to the random selection of samples.
      `principal_dataset` is required for this method. `principal_weight` is
      required iff `extra_weights` are also provided.

    The choice of `principal_dataset` is important and should, in most
    cases, be chosen as the largest underlying dataset as compared to
    `extra_datasets.` `positives` and `negatives` where `len(negatives)` >>
    `len(positives)` and with `positives` corresponding to `principal_dataset,`
    the desired behavior of epochs determined by the exhaustion of `positives`
    and the continued mixing of unique elements from `negatives` may not occur:
    On sampled dataset reiteration `positives` will again be exhausted but
    elements from `negatives` may be those same seen in the previous epoch
    (as they occur at the beginning of the same, reiterated underlying
    `negatives` dataset). In this case, the recommendations are to:

    1) Reformulate the sampling in terms of the larger dataset (`negatives`),
       where, with `fixed_cardinality=False`, if the exhaustion of `negatives`
       is desired, or, with `fixed_cardinality=True`, when
       `principal_cardinality` can be used to specify the desired number of
       elements from `negatives.`
    2) Ensure that the underlying `principal_dataset` of `negatives` are
       well-sharded. In this way, the nondeterminism of interleaving will
       randomly access elements of `negatives` on reiteration.

    Args:
      context: An `tf.distribute.InputContext` for sharding.

    Returns:
      A `tf.data.Dataset.`
    """
    if self._principal_file_pattern is not None:
      principal_filenames = _sorted_glob_or_raise(self._principal_file_pattern,
                                                  "principal file pattern")
      extra_filenames = [
          _sorted_glob_or_raise(f, "extra file pattern")
          for f in self._extra_file_patterns
      ]
    else:
      principal_filenames = self._principal_filenames
      extra_filenames = self._extra_filenames
    return _process_sampled_dataset(
        tf.data.Dataset.from_tensor_slices(principal_filenames),
        [tf.data.Dataset.from_tensor_slices(f) for f in extra_filenames],
        self._principal_weight,
        self._extra_weights,
        num_shards=context.num_input_pipelines,
        index=context.input_pipeline_id,
        principal_cardinality=self._principal_cardinality,
        fixed_cardinality=self._fixed_cardinality,
        shuffle_dataset=self._shuffle_filenames,
        interleave_fn=self._interleave_fn,
        examples_shuffle_size=self._examples_shuffle_size)


class TFRecordDatasetProvider(SimpleDatasetProvider):

  def __init__(self, *args, **kwargs):
    super().__init__(
        *args,
        interleave_fn=tf.data.TFRecordDataset,
        **kwargs)


class SampleTFRecordDatasetsProvider(SimpleSampleDatasetsProvider):

  def __init__(self, *args, **kwargs):
    super().__init__(
        *args,
        interleave_fn=tf.data.TFRecordDataset,
        **kwargs)
