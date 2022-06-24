"""`tf.data.Dataset` of `tfgnn.GraphTensor` methods."""
from typing import Callable, Optional, Sequence

import tensorflow as tf


def _get_dataset(
    file_pattern: str,
    *,
    num_shards: int,
    index: int,
    shuffle_filenames: bool = False,
    interleave_fn: Callable[..., tf.data.Dataset],
    examples_shuffle_size: Optional[int] = None) -> tf.data.Dataset:
  """Gets a `tf.data.Dataset` with sharding, interleaving, shuffling and prefetching."""
  filenames = sorted(tf.io.gfile.glob(file_pattern))
  dataset = tf.data.Dataset.from_tensor_slices(filenames)
  # Shard first before running any randomization operators (e.g. shuffle)
  dataset = dataset.shard(num_shards, index)
  if shuffle_filenames:
    dataset = dataset.shuffle(dataset.cardinality())
  dataset = dataset.interleave(
      interleave_fn,
      deterministic=False,
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  if examples_shuffle_size is not None:
    dataset = dataset.shuffle(examples_shuffle_size)
  return dataset.prefetch(tf.data.AUTOTUNE)


class SimpleDatasetProvider:
  """Builds a `tf.data.Dataset` from a file pattern.

  This `SimpleDatasetProvider` builds a `tf.data.Dataset` as follows:
   - The filenames matching the given `file_pattern` are sharded according to
     the `InputContext`.
   - Filenames are shuffled (if requested).
   - The files in each shard are interleaved after being read by the
     `interleave_fn`.
   - Examples are shuffled (if requested), auto-prefetched, and returned for use
     in one replica of the trainer.
  """

  def __init__(self,
               file_pattern: str,
               *,
               shuffle_filenames: bool = False,
               interleave_fn: Callable[..., tf.data.Dataset],
               examples_shuffle_size: Optional[int] = None):
    """Captures the args shared across `get_dataset(...)` calls.

    Args:
      file_pattern: A file pattern (to be glob with `tf.io.gfile.glob`).
      shuffle_filenames: If enabled, filenames will be shuffled before any file
        reads. Through interleaving, some files may be read in parallel: the
        details are auto-tuned for throughput.
      interleave_fn: A callback that receives a single filename and returns
        a `tf.data.Dataset` with the `tf.Example` values from that file.
      examples_shuffle_size: An optional buffer size for example shuffling.
    """
    self._file_pattern = file_pattern
    self._shuffle_filenames = shuffle_filenames
    self._interleave_fn = interleave_fn
    self._examples_shuffle_size = examples_shuffle_size

  def get_dataset(self, context: tf.distribute.InputContext) -> tf.data.Dataset:
    """Gets a `tf.data.Dataset` by `context` per replica."""
    return _get_dataset(
        self._file_pattern,
        num_shards=context.num_input_pipelines,
        index=context.input_pipeline_id,
        shuffle_filenames=self._shuffle_filenames,
        interleave_fn=self._interleave_fn,
        examples_shuffle_size=self._examples_shuffle_size)


def _get_sampled_dataset(
    principal_file_pattern: str,
    extra_file_patterns: Sequence[str],
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
    shuffle_filenames: bool = False,
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

  def dataset_fn(file_pattern):
    return _get_dataset(
        file_pattern,
        num_shards=num_shards,
        index=index,
        shuffle_filenames=shuffle_filenames,
        interleave_fn=interleave_fn,
        examples_shuffle_size=examples_shuffle_size)

  if fixed_cardinality:
    datasets = [
        *(dataset_fn(p).repeat() for p in extra_file_patterns),
        dataset_fn(principal_file_pattern).repeat()
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
        *(dataset_fn(p).repeat(count) for p in extra_file_patterns),
        # If there are `weights`, the `principal_weight` comes last,
        # so the principal dataset has to be in the same place.
        dataset_fn(principal_file_pattern)
    ]
    sampled_dataset = tf.data.Dataset.sample_from_datasets(
        datasets,
        weights=weights,
        stop_on_empty_dataset=True)

  return sampled_dataset.prefetch(tf.data.AUTOTUNE)


class SimpleSampleDatasetsProvider:
  """Builds a sampling `tf.data.Dataset` from a multiple file patterns.

  For complete explanations regarding sampling see `_get_sampled_dataset()`.

  This `SimpleSampleDatasetsProvider` builds a `tf.data.Dataset` as follows:
   - The filenames matching the given `principal_file_pattern` and
     `extra_file_patterns` are sharded according to the `InputContext`.
   - Filenames are shuffled (if requested).
   - Examples from all file patterns are sampled according to `principal_weight`
     and `extra_weights.`
   - The files in each shard are interleaved after being read by the
     `interleave_fn`.
   - Examples are shuffled (if requested), auto-prefetched, and returned for
     use in one replica of the trainer.
  """

  def __init__(self,
               principal_file_pattern: str,
               extra_file_patterns: Sequence[str],
               principal_weight: Optional[float] = None,
               extra_weights: Optional[Sequence[float]] = None,
               *,
               principal_cardinality: Optional[int] = None,
               fixed_cardinality: bool = False,
               shuffle_filenames: bool = False,
               interleave_fn: Callable[..., tf.data.Dataset],
               examples_shuffle_size: Optional[int] = None):
    """Captures the args shared across `get_dataset(...)` calls.

    Args:
      principal_file_pattern: A principal file pattern for sampling
        (to be globed with `tf.io.gfile.glob`).
      extra_file_patterns: File patterns (to be globed with `tf.io.gfile.glob`).
      principal_weight: An optional weight for the dataset corresponding to
        `principal_file_pattern.` Required iff `extra_weights` are also
        provided.
      extra_weights: Optional weights corresponding to `file_patterns` for
        sampling. Required iff `principal_weight` is also provided.
      principal_cardinality: Iff `fixed_cardinality`=True, the size of the
        returned dataset is computed as `principal_cardinality` /
        `principal_weight` (with a default of uniform weights).
      fixed_cardinality: Whether to take a fixed number of elements.
      shuffle_filenames: If enabled, filenames will be shuffled before any file
        reads. Through interleaving, some files may be read in parallel: the
        details are auto-tuned for throughput.
      interleave_fn: A fn applied with `tf.data.Dataset.interleave.`
      examples_shuffle_size: An optional buffer size for example shuffling. If
        specified, the size is adjusted to `shuffle_size //
        (len(file_patterns) + 1).`
    """
    self._principal_file_pattern = principal_file_pattern
    self._extra_file_patterns = extra_file_patterns
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
    and the contined mixing of unique elements from `negatives` may not occur:
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
    return _get_sampled_dataset(
        self._principal_file_pattern,
        self._extra_file_patterns,
        self._principal_weight,
        self._extra_weights,
        num_shards=context.num_input_pipelines,
        index=context.input_pipeline_id,
        principal_cardinality=self._principal_cardinality,
        fixed_cardinality=self._fixed_cardinality,
        shuffle_filenames=self._shuffle_filenames,
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
