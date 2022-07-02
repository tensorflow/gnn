"""Model helpers."""
import collections
from typing import Any, Callable

import tensorflow as tf

Model = tf.keras.Model


def _if_nested(maybe_nested: Any,
               true_fn: Callable[..., Any],
               false_fn: Callable[..., Any]) -> Any:
  if tf.nest.is_nested(maybe_nested):
    if isinstance(maybe_nested, collections.abc.Sequence):
      return true_fn(maybe_nested)
    else:
      raise ValueError("Only Sequence nested structures are supported")
  return false_fn(maybe_nested)


def chain_first_output(m1: Model,
                       m2: Model,
                       first_output_only: bool = True) -> Model:
  """Concatenates two `tf.keras.Model` by chaining the first output.

  Note: Outputs of `m1` and `m2` maybe be atoms or nested structures of kind
  `collections.abc.Sequence` only. Example psuedocode, where single outputs are
  treated as lists of length one:

  out1first, *out1rest = m1(m1.input)
  out2first, *out2rest = m2(out1first)
  return out2first if first_output_only else [out2first, *out1rest, *out2rest]

  Args:
    m1: A Keras model for concatenation.
    m2: A Keras model for concatenation.
    first_output_only: Whether to nest the outputs of `m1` and `m2`: if True,
      only the first output of `m2` is used.

  Returns:
    A new concatenated Keras model.
  """
  x, y = _if_nested(m1(m1.input), lambda o: (o[0], o[1:]), lambda o: (o, []))
  output = _if_nested(m2(x), lambda o: [o[0], *y, *o[1:]], lambda o: [o, *y])
  return Model(m1.input, not first_output_only and output or output[0])
