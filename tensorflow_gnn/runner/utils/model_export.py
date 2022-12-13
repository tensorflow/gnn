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
"""Model export helpers."""
import os
from typing import Any, Optional, Union

import tensorflow as tf
from tensorflow_gnn.runner import interfaces
from tensorflow_gnn.runner.utils import model as model_utils

Field = Union[tf.Tensor, tf.RaggedTensor]


# TODO(b/196880966) Move to `model.py` and add unit tests.
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


class KerasModelExporter(interfaces.ModelExporter):
  """Exports a Keras model (with Keras API) via tf.keras.models.save_model."""

  def __init__(self,
               *,
               output_names: Optional[Any] = None,
               subdirectory: Optional[str] = None,
               include_preprocessing: bool = True,
               options: Optional[tf.saved_model.SaveOptions] = None):
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
      subdirectory: An optional subdirectory, if set: models are exported to
        `os.path.join(export_dir, subdirectory).`
      include_preprocessing: Whether to include any `preprocess_model.`
      options: Options for saving to SavedModel.
    """
    self._output_names = output_names
    self._subdirectory = subdirectory
    self._include_preprocessing = include_preprocessing
    self._options = options

  def save(self,
           preprocess_model: Optional[tf.keras.Model],
           model: tf.keras.Model,
           export_dir: str):
    """Exports a Keras model (with Keras API) via tf.keras.models.save_model.

    Importantly: the `preprocess_model`, if provided, and `model` are
    concatenated before any export. Concatenation involves the chaining of the
    first output of `preprocess_model` to the only input of `model.` The result
    is a model with the input of `preprocess_model` and the output of `model.`

    Args:
      preprocess_model: An optional `tf.keras.Model` for preprocessing.
      model: A `tf.keras.Model` to save.
      export_dir: A destination directory for the model.
    """
    if preprocess_model is not None and self._include_preprocessing:
      model = model_utils.chain_first_output(preprocess_model, model)
    if self._output_names is not None:
      output = _rename_output(model.output, self._output_names)
      model = tf.keras.Model(model.input, output)
    if self._subdirectory:
      export_dir = os.path.join(export_dir, self._subdirectory)
    tf.keras.models.save_model(model, export_dir, options=self._options)


class SubmoduleExporter(interfaces.ModelExporter):
  """Exports a Keras model submodule (`getarr(model, 'submodules')`) by name."""

  def __init__(self,
               submodule_name: str,
               *,
               output_names: Optional[Any] = None,
               subdirectory: Optional[str] = None,
               include_preprocessing: bool = False,
               options: Optional[tf.saved_model.SaveOptions] = None):
    """Captures the args shared across `save(...)` calls.

    Args:
      submodule_name: The name of the submodule to export.
      output_names: The names for output Tensor(s), see: `KerasModelExporter.`
      subdirectory: An optional subdirectory, if set: submodules are exported
        to `os.path.join(export_dir, subdirectory).`
      include_preprocessing: Whether to include any `preprocess_model.`
      options: Options for saving to SavedModel.
    """
    self._output_names = output_names
    self._subdirectory = subdirectory
    self._submodule_name = submodule_name
    self._include_preprocessing = include_preprocessing
    self._options = options

  def save(self,
           preprocess_model: Optional[tf.keras.Model],
           model: tf.keras.Model,
           export_dir: str):
    """Saves a Keras model submodule.

    Importantly: the `preprocess_model`, if provided, and `model` are
    concatenated before any export.

    Args:
      preprocess_model: An optional `tf.keras.Model` for preprocessing.
      model: A `tf.keras.Model` to save.
      export_dir: A destination directory for the model.
    """
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
      [submodel] = submodules
      raise ValueError(
          f"Submodule ({submodel}) is neither a Keras Model nor Layer`")

    exporter = KerasModelExporter(
        output_names=self._output_names,
        subdirectory=self._subdirectory,
        include_preprocessing=self._include_preprocessing,
        options=self._options)

    exporter.save(preprocess_model, submodel, export_dir)
