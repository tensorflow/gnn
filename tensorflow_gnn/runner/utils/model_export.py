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
from typing import Any, Optional, Sequence, Union

import tensorflow as tf
from tensorflow_gnn.runner import interfaces

Field = Union[tf.Tensor, tf.RaggedTensor]


class KerasModelExporter(interfaces.ModelExporter):
  """Exports a Keras model (with Keras API) via `tf.keras.models.save_model`."""

  def __init__(self,
               *,
               output_names: Optional[Any] = None,
               subdirectory: Optional[str] = None,
               include_preprocessing: bool = True,
               options: Optional[tf.saved_model.SaveOptions] = None,
               use_legacy_model_save: Optional[bool] = None):
    """Captures the args shared across `save(...)` calls.

    Args:
      output_names: By default, each output of the exported model uses the name
        of the final Keras layer that created it as its key in the SavedModel
        signature. This argument can be set to a single `str` name or a nested
        structure of `str` names to override the output names. Its nesting
        structure must match the exported model's output (as checked by
        `tf.nest.assert_same_structure`). Any `None` values in `output_names`
        are ignored, leaving that output with its default name.
      subdirectory: An optional subdirectory, if set: models are exported to
        `os.path.join(export_dir, subdirectory).`
      include_preprocessing: Whether to include any `preprocess_model.`
      options: Options for saving to a TensorFlow `SavedModel`.
      use_legacy_model_save: Optional; most users can leave it unset to get a
        useful default for export to inference. See `runner.export_model()`
        for more.
    """
    self._output_names = output_names
    self._subdirectory = subdirectory
    self._include_preprocessing = include_preprocessing
    self._options = options
    self._use_legacy_model_save = use_legacy_model_save

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

    _export_model(export_dir,
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

    _export_model(export_dir,
                  preprocess_model,
                  submodule,
                  self._include_preprocessing,
                  self._output_names,
                  self._subdirectory,
                  self._options)


def export_model(model: tf.keras.Model,
                 export_dir: str,
                 *,
                 output_names: Optional[Any] = None,
                 options: Optional[tf.saved_model.SaveOptions] = None,
                 use_legacy_model_save: Optional[bool] = None) -> None:
  """Exports a Keras model without traces s.t. it is loadable without TF-GNN.

  Args:
    model: Keras model instance to be saved.
    export_dir: Path where to save the model.
    output_names: Optionally, a nest of `str` values or `None` with the same
      structure as the outputs of `model`. A non-`None` value is used as that
      output's key in the SavedModel signature. By default, an output gets
      the name of the final Keras layer creating it as its key (matching the
      behavior of legacy `Model.save(save_format="tf")`).
    options: An optional `tf.saved_model.SaveOptions` argument.
    use_legacy_model_save: Optional; most users can leave it unset to get a
      useful default for export to inference. If set to `True`, forces the use
      of `Model.save()`, which exports a SavedModel suitable for inference and
      potentially also for reloading as a Keras model (depending on its Layers).
      If set to `False`, forces the use of `tf.keras.export.ExportArchive`,
      which is usable as of TensorFlow 2.13 and is advertised as the more
      streamlined way of exporting to SavedModel for inference only. Currently,
      `None` behaves like `True`, but the long-term plan is to migrate towards
      `False`.
  """
  if use_legacy_model_save is None:
    use_legacy_model_save = True

  # Create a tf.function for export as the default signature.
  #
  # The goal is to replicate the traditional behavior of model.save()
  # and its sole "serving_default" signature, even if we use...
  #  * model.save(save_traces=False), which removes that default and requires
  #    some boilerplate to get it back,
  #  * model.export(), or rather the ExportArchive formalism behind it, which
  #      * by default fails to retrieve the output names for the signature
  #        in the expected way from layer names (b/312907301);
  #      * by default creates two equal signatures "serve" and "serving_default"
  #        but we'd rather not get clients to depend on "serve" before we are
  #        ready to abandon the traditonal model.save() behavior.
  nested_arg_specs, nested_kwarg_specs = model.save_spec()
  flat_arg_specs = tf.nest.flatten((nested_arg_specs, nested_kwarg_specs))

  if output_names is None:
    flat_output_names = model.output_names
    if not (isinstance(flat_output_names, Sequence)
            and not isinstance(flat_output_names, (str, bytes))
            and all(isinstance(name, str) for name in flat_output_names)):
      raise ValueError("Expected Model.output_names to be a Sequence[str], "
                       f"got: {flat_output_names}")
  else:
    tf.nest.assert_same_structure(model.output, output_names)
    flat_output_names = tf.nest.flatten(output_names)
    assert len(flat_output_names) == len(model.output_names)
    for i in range(len(flat_output_names)):
      if flat_output_names[i] is None:
        flat_output_names[i] = model.output_names[i]

  @tf.function(input_signature=flat_arg_specs)
  def serving_default(*flat_args):
    nested_args, nested_kwargs = tf.nest.pack_sequence_as(
        (nested_arg_specs, nested_kwarg_specs),
        flat_args)
    nested_outputs = model(*nested_args, **nested_kwargs)
    return dict(zip(flat_output_names, tf.nest.flatten(nested_outputs)))

  # Do the export.
  if use_legacy_model_save:
    model.save(
        export_dir,
        save_format="tf", include_optimizer=False, save_traces=False,
        signatures={
            tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY: serving_default
        },
        options=options)
  else:
    export_archive = tf.keras.export.ExportArchive()
    export_archive.track(model)
    export_archive.add_endpoint(
        tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY, serving_default)
    export_archive.write_out(export_dir, options=options)


def _export_model(export_dir: str,
                  preprocess_model: tf.keras.Model,
                  model: tf.keras.Model,
                  include_preprocessing: bool,
                  output_names: Optional[Any] = None,
                  subdirectory: Optional[str] = None,
                  options: Optional[tf.saved_model.SaveOptions] = None,
                  use_legacy_model_save: Optional[bool] = None):
  """Exports a Keras model."""
  if preprocess_model and include_preprocessing:
    xs, *_ = preprocess_model.output
    model = tf.keras.Model(preprocess_model.input, model(xs))
  if subdirectory:
    export_dir = os.path.join(export_dir, subdirectory)
  export_model(model, export_dir, output_names=output_names, options=options,
               use_legacy_model_save=use_legacy_model_save)
