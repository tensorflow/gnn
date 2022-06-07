"""Model helpers."""
import collections
import functools
from typing import Any

import tensorflow as tf


def _chain_first_output(inputs: Any, m: tf.keras.Model) -> Any:
  if isinstance(inputs, collections.Sequence):
    return m(inputs[0])
  return m(inputs)


def chain(*args: tf.keras.Model) -> tf.keras.Model:
  """Concatenates many `tf.keras.Model` by chaining their output.

  - For `Sequence` output, the first element is used for chaining.
  - For all other output, the entire output is used for chaining.

  Args:
    *args: Keras model(s) for chaining.

  Returns:
    A new chained Keras model.
  """
  if not args:
    raise ValueError("At least one `tf.keras.Model` is requirred")
  first, *rest = args
  output = functools.reduce(_chain_first_output, rest, first(first.input))
  return tf.keras.Model(first.input, output)
