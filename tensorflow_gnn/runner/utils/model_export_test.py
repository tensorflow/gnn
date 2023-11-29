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
"""Tests for model_export."""
import itertools
import os
from typing import Any, Optional

from absl.testing import parameterized
import tensorflow as tf
import tensorflow_gnn as tfgnn

from tensorflow_gnn.runner import interfaces
from tensorflow_gnn.runner.utils import model_export


def _duplicate_submodules(name: str) -> tf.keras.Model:
  inputs = tf.keras.Input(shape=(4,))
  outputs = tf.keras.layers.Dense(2)(inputs)
  model = tf.keras.Model(inputs, outputs)
  model.submodule1 = _no_submodule(1, 1, name)
  model.submodule2 = _no_submodule(1, 1, name)
  return model


def _layer_submodule(ninputs: int, noutputs: int, name: str) -> tf.keras.Model:
  def fn(x):
    if ninputs > 1:
      x = tf.math.add_n(x)
    if noutputs > 1:
      return [x * (i + 8191) for i in range(noutputs)]
    else:
      return x
  if ninputs > 1:
    inputs = [tf.keras.Input(shape=(4,)) for _ in range(ninputs)]
  else:
    inputs = tf.keras.Input(shape=(4,))
  outputs = tf.keras.layers.Lambda(fn, name=name)(inputs)
  return tf.keras.Model(inputs, outputs)


def _model_submodule(ninputs: int, noutputs: int, name: str) -> tf.keras.Model:
  submodel = _no_submodule(ninputs, noutputs, name)
  if noutputs > 1:
    output = tf.math.add_n(submodel(submodel.input)) + 8191
  else:
    output = submodel(submodel.input) + 8191
  model = tf.keras.Model(submodel.input, output)
  return model


def _no_submodule(ninputs: int, noutputs: int, name: str) -> tf.keras.Model:
  inputs = [tf.keras.Input(shape=(4,)) for _ in range(ninputs)]
  x = tf.add_n(tf.nest.flatten(inputs))
  delta = itertools.count(1)
  outputs = [next(delta) + x for _ in range(noutputs)]
  if len(inputs) == 1:
    [inputs] = inputs
  if len(outputs) == 1:
    [outputs] = outputs
  return tf.keras.Model(inputs, outputs, name=name)


def _structure_like(inputs: Any, outputs: Any) -> tf.keras.Model:
  inputs = tf.nest.map_structure(lambda _: tf.keras.Input(shape=(4,)), inputs)
  x = tf.add_n(tf.nest.flatten(inputs))
  delta = itertools.count(1)
  outputs = tf.nest.map_structure(lambda _: next(delta) + x, outputs)
  return tf.keras.Model(inputs, outputs)


def _tf_module_as_submodule(name: str) -> tf.keras.Model:
  inputs = tf.keras.Input(shape=(4,))
  outputs = tf.keras.layers.Dense(2)(inputs)
  submodule = tf.Module(name=name)
  model = tf.keras.Model(inputs, outputs)
  model.submodule = submodule
  return model


def _named_inputs_outputs() -> tf.keras.Model:
  left = tf.keras.Input(shape=(1,), name="left")
  right = tf.keras.Input(
      type_spec=tf.TensorSpec((None, 1), tf.float32, name="right"),
      name="ignored")
  summation = tf.keras.layers.Add(name="summation")([left, right])
  subtraction = tf.keras.layers.Subtract(name="subtraction")([left, right])
  return tf.keras.Model([left, right], [summation, subtraction])


class ModelExportTests(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters([
      dict(
          testcase_name="SingleInputOutput",
          model=_no_submodule(1, 1, "abc123"),
          output_names="output_a",
      ),
      dict(
          testcase_name="SingleInputMultipleOutput",
          model=_no_submodule(1, 2, "abc123"),
          output_names=["output_b", "output_a"],
      ),
      dict(
          testcase_name="MultipleInputSingleOutput",
          model=_no_submodule(2, 1, "abc123"),
          output_names="output_b",
      ),
      dict(
          testcase_name="MultipleInputOutput",
          model=_no_submodule(2, 2, "abc123"),
          output_names=["output_a", "output_b"],
      ),
      dict(
          testcase_name="MultipleInputOutputLegacySave",
          model=_no_submodule(2, 2, "abc123"),
          output_names=["output_a", "output_b"],
          use_legacy_model_save=True,
      ),
      dict(
          testcase_name="MultipleInputOutputExport",
          model=_no_submodule(2, 2, "abc123"),
          output_names=["output_a", "output_b"],
          use_legacy_model_save=False,
      ),
      dict(
          testcase_name="MappingOutput",
          model=_structure_like(None, {"x": None, "y": None}),
          output_names={"y": "output_b", "x": "output_a"},
      ),
      dict(
          testcase_name="MappingOutputLegacySave",
          model=_structure_like(None, {"x": None, "y": None}),
          output_names={"y": "output_b", "x": "output_a"},
          use_legacy_model_save=True,
      ),
      dict(
          testcase_name="MappingOutputExport",
          model=_structure_like(None, {"x": None, "y": None}),
          output_names={"y": "output_b", "x": "output_a"},
          use_legacy_model_save=False,
      ),
      dict(
          testcase_name="NamedInputsOutputs",
          model=_named_inputs_outputs(),
          expected_input_names=["left", "right"],
          expected_output_names=["summation", "subtraction"],
          use_legacy_model_save=False,
      ),
      dict(
          testcase_name="NamedInputsOutputsLegacySave",
          model=_named_inputs_outputs(),
          expected_input_names=["left", "right"],
          expected_output_names=["summation", "subtraction"],
          use_legacy_model_save=True,
      ),
  ])
  def test_keras_model_exporter(
      self,
      model: tf.keras.Model,
      output_names: Any = None,
      expected_output_names: Any = None,  # Defaults to flatten(output_names).
      expected_input_names: Any = None,  # Defaults to model.input[*].name.
      use_legacy_model_save: Optional[bool] = None):

    if (use_legacy_model_save is False  # Don't trigger on None. pylint: disable=g-bool-id-comparison
        and tf.__version__.startswith("2.12.")):
      self.skipTest("Model.export() does not work for TF < 2.13")

    export_dir = self.create_tempdir()
    exporter = model_export.KerasModelExporter(
        output_names=output_names,
        use_legacy_model_save=use_legacy_model_save)
    exporter.save(interfaces.RunResult(None, None, model), export_dir)

    if expected_input_names is None:
      expected_input_names = [i.name for i in tf.nest.flatten(model.input)]
    func = lambda i: tf.random.uniform((1, *i.shape[1:]))
    args = [func(i) for i in tf.nest.flatten(model.input)]
    kwargs = dict(zip(expected_input_names, args))

    model_output = model(args)

    flat_model_outputs = tf.nest.flatten(model_output)
    if expected_output_names is None:
      tf.nest.assert_same_structure(model_output, output_names)
      expected_output_names = tf.nest.flatten(output_names)
    else:
      self.assertLen(flat_model_outputs, len(expected_output_names))
    zipped = zip(flat_model_outputs, expected_output_names)

    saved_model = tf.saved_model.load(export_dir)
    saved_model_outputs = saved_model.signatures["serving_default"](**kwargs)

    self.assertCountEqual(expected_output_names, saved_model_outputs.keys())
    for output, name in zipped:
      self.assertAllClose(
          output,
          saved_model_outputs[name],
          msg=f"Testing output {name}")

  @parameterized.named_parameters([
      dict(
          testcase_name="SingleInputOutputLayerSubmodule",
          model=_layer_submodule(1, 1, "abc123"),
          submodule_name="abc123",
          subdirectory=None,
          output_names="output",
      ),
      dict(
          testcase_name="MultipleInputOutputLayerSubmodule",
          model=_layer_submodule(2, 2, "abc123"),
          submodule_name="abc123",
          subdirectory=None,
          output_names=["output_0", "output_1"]
      ),
      dict(
          testcase_name="SingleInputMultipleOutputLayerSubmodule",
          model=_layer_submodule(1, 2, "abc123"),
          submodule_name="abc123",
          subdirectory=None,
          output_names=["output_0", "output_1"]
      ),
      dict(
          testcase_name="MultipleInputSingleOutputLayerSubmodule",
          model=_layer_submodule(2, 1, "abc123"),
          submodule_name="abc123",
          subdirectory=None,
          output_names="output",
      ),
      dict(
          testcase_name="SingleInputOutputModelSubmodule",
          model=_model_submodule(1, 1, "abc123"),
          submodule_name="abc123",
          subdirectory=None,
          output_names="output",
      ),
      dict(
          testcase_name="MultipleInputOutputModelSubmodule",
          model=_model_submodule(2, 2, "abc123"),
          submodule_name="abc123",
          subdirectory=None,
          output_names=["output_0", "output_1"],
      ),
      dict(
          testcase_name="SingleInputMultipleOutputModelSubmodule",
          model=_model_submodule(1, 2, "abc123"),
          submodule_name="abc123",
          subdirectory=None,
          output_names=["output_0", "output_1"],
      ),
      dict(
          testcase_name="MultipleInputSingleOutputModelSubmodule",
          model=_model_submodule(2, 1, "abc123"),
          submodule_name="abc123",
          subdirectory=None,
          output_names="output",
      ),
      dict(
          testcase_name="WithSubdirectory",
          model=_layer_submodule(1, 1, "abc123"),
          submodule_name="abc123",
          subdirectory="tmp",
          output_names="output",
      ),
  ])
  def test_submodule_exporter(
      self,
      model: tf.keras.Model,
      submodule_name: str,
      subdirectory: str,
      output_names: Any):
    export_dir = self.create_tempdir()
    exporter = model_export.SubmoduleExporter(
        submodule_name,
        output_names=output_names,
        subdirectory=subdirectory)
    exporter.save(interfaces.RunResult(None, None, model), export_dir)

    submodule = next(m for m in model.submodules if submodule_name == m.name)

    if subdirectory:
      saved_model = tf.saved_model.load(os.path.join(export_dir, subdirectory))
    else:  # `export_dir` only
      saved_model = tf.saved_model.load(export_dir)

    if tf.nest.is_nested(submodule.input):
      args = [tf.random.uniform([1] + i.shape[1:]) for i in submodule.input]
      kwargs = {i.name: v for i, v in zip(submodule.input, args)}
    else:  # Single input
      args = tf.random.uniform([1] + submodule.input.shape[1:])
      kwargs = {submodule.input.name: args}

    submodule_output = submodule(args)
    saved_model_outputs = saved_model.signatures["serving_default"](**kwargs)

    if tf.nest.is_nested(submodule.output):
      submodule_outputs = submodule_output
    else:  # Single output
      output_names = [output_names]
      submodule_outputs = [submodule_output]

    for output, name in zip(submodule_outputs, output_names):
      self.assertAllClose(
          output,
          saved_model_outputs[name], msg=f"Testing {name}...")

  @parameterized.named_parameters([
      dict(
          testcase_name="DuplicateSubmodules",
          model=_duplicate_submodules("abc123"),
          submodule_name="abc123",
          expected_error=r"More than one intermediate layer found \(\[.*\]\)",
      ),
      dict(
          testcase_name="NoSubmodule",
          model=_no_submodule(1, 1, "abc123"),
          submodule_name="abc123",
          expected_error="No intermediate layer `.*` found",
      ),
      dict(
          testcase_name="TFModuleAsSubmodule",
          model=_tf_module_as_submodule("abc123"),
          submodule_name="abc123",
          expected_error="No intermediate layer `.*` found",
      ),
  ])
  def test_submodule_exporter_fails(
      self,
      model: tf.keras.Model,
      submodule_name: str,
      expected_error: str):
    exporter = model_export.SubmoduleExporter(submodule_name)
    with self.assertRaisesRegex(ValueError, expected_error):
      run_result = interfaces.RunResult(None, None, model)
      exporter.save(run_result, self.create_tempdir())

  def test_simple_graph_piece(self):
    export_dir = self.create_tempdir()

    inputs = tf.keras.Input([6, 4], name="inputs")
    node_set = tfgnn.NodeSet.from_fields(
        features={tfgnn.HIDDEN_STATE: inputs},
        sizes=[6])
    node_set = node_set.replace_features({
        tfgnn.HIDDEN_STATE: node_set[tfgnn.HIDDEN_STATE] * 2
    })
    model = tf.keras.Model(inputs, node_set[tfgnn.HIDDEN_STATE])

    model_export.export_model(model, export_dir)
    saved_model = tf.saved_model.load(export_dir)

    inputs = tf.random.uniform([6, 4])
    results = saved_model.signatures["serving_default"](inputs=inputs)
    self.assertLen(results, 1)

    [actual] = results.values()
    self.assertAllClose(actual, inputs * 2)


if __name__ == "__main__":
  tf.test.main()
