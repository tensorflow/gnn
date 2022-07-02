"""Model export helpers."""
import os
from typing import Any, Optional, Union

import tensorflow as tf

Field = Union[tf.Tensor, tf.RaggedTensor]


def _rename_output(output: Any, names: Any) -> Any:
  """Renames atoms of `output` with `names` for two matching structures."""
  tf.nest.assert_same_structure(output, names, check_types=False)

  if not tf.nest.is_nested(output):
    if names is not None:
      return tf.keras.layers.Layer(name=names)(output)
    else:
      return output

  renamed_output = [
      tf.keras.layers.Layer(name=atom2)(atom1) if atom2 is not None else atom1
      for atom1, atom2 in zip(tf.nest.flatten(output), tf.nest.flatten(names))
  ]

  return tf.nest.pack_sequence_as(names, renamed_output)


class KerasModelExporter:
  """Exports a Keras model (with Keras API) via tf.keras.models.save_model."""

  def __init__(self,
               output_names: Optional[Any] = None):
    """Captures the args shared across `save(...)` calls.

    Args:
      output_names: The name(s) for any output Tensor(s). Can be a single `str`
        name or a nested structure of `str` names. If a nested structure is
        given, it must match the structure of the exported model's output
        (as asserted by `tf.nest.assert_same_structure`): model output is
        renamed by flattening (`tf.nest.flatten`) and zipping the two
        structures. Any `None` values in `output_names` are ignored (leaving
        that corresponding atom with its original name). Renamed atoms are
        packed into the structure of `output_names`: in this way, `dict(...)`
        keys of a model output can also be renamed.
    """
    self._output_names = output_names

  def save(self, model: tf.keras.Model, export_dir: str):
    if self._output_names is not None:
      output = _rename_output(model.output, self._output_names)
      model = tf.keras.Model(model.input, output)
    tf.keras.models.save_model(model, export_dir)


class SubmoduleExporter:
  """Exports a Keras model submodule (`getarr(model, 'submodules')`) by name."""

  def __init__(self,
               submodule_name: str,
               *,
               output_names: Optional[Any] = None,
               subdirectory: Optional[str] = None):
    """Captures the args shared across `save(...)` calls.

    Args:
      submodule_name: The name of the submodule to export.
      output_names: The names for any output Tensor(s), see:
        `KerasModelExporter.`
      subdirectory: An optional subdirectory, if set: submodules are exported
        to `os.path.join(export_dir, subdirectory).`
    """
    self._output_names = output_names
    self._subdirectory = subdirectory
    self._submodule_name = submodule_name

  def save(self, model: tf.keras.Model, export_dir: str):
    """Saves a Keras model submodule."""
    submodules = [m for m in model.submodules if self._submodule_name == m.name]

    if not submodules:
      raise ValueError(f"No submodule `{self._submodule_name}` found")
    elif len(submodules) > 1:
      raise ValueError(f"More than one submodule found ({submodules})")
    elif isinstance(submodules[0], tf.keras.Model):
      [submodel] = submodules
    elif isinstance(submodules[0], tf.keras.layers.Layer):
      [sublayer] = submodules
      submodel = tf.keras.Model(sublayer.input, sublayer.output)
    else:
      raise ValueError(f"Submodule ({submodules}) is neither a `tf.keras.Model`"
                       " nor a `tf.keras.layers.Layer`")

    if self._subdirectory:
      export_dir = os.path.join(export_dir, self._subdirectory)

    KerasModelExporter(self._output_names).save(submodel, export_dir)
