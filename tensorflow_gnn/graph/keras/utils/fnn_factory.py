"""The get_fnn_factory helper and associated logic."""

from typing import Any, Callable, List, Optional, Union

import tensorflow as tf


# TODO(b/192858913): Build this for real (incl. caching for reuse) and test it.
def get_fnn_factory(
    *,
    output_dim: int,
    hidden_dims: Optional[List[int]] = None,
    activation: Optional[Union[str, Callable[..., Any]]] = "relu",
    activate_output: bool = True,
    name: Optional[str] = None,
) -> Callable[[], tf.keras.layers.Layer]:
  """Returns a factory for feed-forward networks (FNNs) in Keras.

  Args:
    output_dim: a positive integer. The number of units in the output layer
      of the FNN.
    hidden_dims: a list of positive integers. The FNN contains hidden layers
      of these sizes before the output layer. Defaults to the empty list.
    activation: Passed as `activation=` argument to the Keras layers that make
      up the FNN (but see 'acticvate_output`).
    activate_output: If set to false, the output layer pf the FNN has no
      activation function (instead of `activation`).
    name: an optional Python string. Used as a name for the returned
      Sequential and as a prefix for its layers.

  Returns:
    A function that can be called without arguments any number of times
    to build the FNN described by the args as a fresh Keras model.
  """
  def fnn_factory():
    fnn = tf.keras.Sequential(name=name)
    for i, units in enumerate(hidden_dims or []):
      fnn.add(tf.keras.layers.Dense(units=units, activation=activation,
                                    name=f"{name}/hidden_{i}"))
    fnn.add(tf.keras.layers.Dense(
        units=output_dim,
        activation=activation if activate_output else None,
        name=f"{name}/output"))
    return fnn

  return fnn_factory
