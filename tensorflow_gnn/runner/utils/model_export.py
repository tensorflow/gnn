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

Field = Union[tf.Tensor, tf.RaggedTensor]


# TODO(b/196880966) Move to `model.py` and add unit tests.
def _rename_output(output: Any, names: Any) -> Any:
  """Renames atoms of `output` with `names` for two matching structures."""
  tf.nest.assert_same_structure(output, names, check_types=False)
  renamed_output = [
      tf.keras.layers.Layer(name=atom2)(atom1) if atom2 is not None else atom1
      for atom1, atom2 in zip(tf.nest.flatten(output), tf.nest.flatten(names))
  ]
  return tf.nest.pack_sequence_as(names, renamed_output)


class KerasModelExporter(interfaces.ModelExporter):
  """Exports a Keras model (with Keras API) via `tf.keras.models.save_model`."""

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
        that corresponding atom with its original name).
      subdirectory: An optional subdirectory, if set: models are exported to
        `os.path.join(export_dir, subdirectory).`
      include_preprocessing: Whether to include any `preprocess_model.`
      options: Options for saving to a TensorFlow `SavedModel`.
    """
    self._output_names = output_names
    self._subdirectory = subdirectory
    self._include_preprocessing = include_preprocessing
    self._options = options

  def save(self, run_result: interfaces.RunResult, export_dir: str):
    """Exports a Keras model (with Keras API) via tf.keras.models.save_model.

    Importantly: the `run_result.preprocess_model`, if provided, and
    `run_result.trained_model` are stacked before any export. Stacking involves
    the chaining of the first output of `run_result.preprocess_model` to the
    only input of `run_result.trained_model.` The result is a model with the
    input of `run_result.preprocess_model` and the output of
    `run_result.trained_model.`

    Args:
      run_result: A `RunResult` from training.
      export_dir: A destination directory.
    """
    preprocess_model = run_result.preprocess_model
    model = run_result.trained_model

    if preprocess_model and not preprocess_model.built:
      raise ValueError("`preprocess_model` is expected to have been built")
    elif not model.built:
      raise ValueError("`model` is expected to have been built")

    _save_model(export_dir,
                preprocess_model,
                model,
                self._include_preprocessing,
                self._output_names,
                self._subdirectory,
                self._options)


class SubmoduleExporter(interfaces.ModelExporter):
  """Exports a Keras submodule.

  Given a `RunResult`, this exporter creates and exports a submodule with
  inputs identical to the trained model and outputs from some intermediate layer
  (named `sublayer_name`). For example, with pseudocode:

  `trained_model = tf.keras.Sequential([layer1, layer2, layer3, layer4])`
  and
  `SubmoduleExporter(sublayer_name='layer2')`

  The exported submodule is:

  `submodule = tf.keras.Sequential([layer1, layer2])`
  """

  def __init__(self,
               sublayer_name: str,
               *,
               output_names: Optional[Any] = None,
               subdirectory: Optional[str] = None,
               include_preprocessing: bool = False,
               options: Optional[tf.saved_model.SaveOptions] = None):
    """Captures the args shared across `save(...)` calls.

    Args:
      sublayer_name: The name of the submodule's final layer.
      output_names: The names for output Tensor(s), see: `KerasModelExporter`.
      subdirectory: An optional subdirectory, if set: submodules are exported
        to `os.path.join(export_dir, subdirectory)`.
      include_preprocessing: Whether to include any `preprocess_model`.
      options: Options for saving to a TensorFlow `SavedModel`.
    """
    self._sublayer_name = sublayer_name
    self._output_names = output_names
    self._subdirectory = subdirectory
    self._include_preprocessing = include_preprocessing
    self._options = options

  def save(self, run_result: interfaces.RunResult, export_dir: str):
    """Saves a Keras model submodule.

    Importantly: the `run_result.preprocess_model`, if provided, and
    `run_result.trained_model` are stacked before any export. Stacking involves
    the chaining of the first output of `run_result.preprocess_model` to the
    only input of `run_result.trained_model.` The result is a model with the
    input of `run_result.preprocess_model` and the output of
    `run_result.trained_model.`

    Args:
      run_result: A `RunResult` from training.
      export_dir: A destination directory.
    """
    preprocess_model = run_result.preprocess_model
    model = run_result.trained_model

    if preprocess_model and not preprocess_model.built:
      raise ValueError("`preprocess_model` is expected to have been built")
    elif not model.built:
      raise ValueError("`model` is expected to have been built")

    layers = [l for l in model.layers if self._sublayer_name == l.name]

    if not layers:
      raise ValueError(f"No intermediate layer `{self._sublayer_name}` found")
    elif len(layers) > 1:
      raise ValueError(f"More than one intermediate layer found ({layers})")

    [layer] = layers
    submodule = tf.keras.Model(model.input, layer.output)

    _save_model(export_dir,
                preprocess_model,
                submodule,
                self._include_preprocessing,
                self._output_names,
                self._subdirectory,
                self._options)


def _save_model(export_dir: str,
                preprocess_model: tf.keras.Model,
                model: tf.keras.Model,
                include_preprocessing: bool,
                output_names: Optional[Any] = None,
                subdirectory: Optional[str] = None,
                options: Optional[tf.saved_model.SaveOptions] = None):
  """Saves a Keras model."""
  if preprocess_model and include_preprocessing:
    xs, *_ = preprocess_model.output
    model = tf.keras.Model(preprocess_model.input, model(xs))
  if output_names:
    output = _rename_output(model.output, output_names)
    model = tf.keras.Model(model.input, output)
  if subdirectory:
    export_dir = os.path.join(export_dir, subdirectory)
  tf.keras.models.save_model(model, export_dir, options=options)
