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
import os
from typing import Any

from absl.testing import parameterized
import tensorflow as tf
from tensorflow_gnn.runner import orchestration
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
  x = tf.math.add_n(inputs)
  outputs = [tf.keras.layers.Dense(2)(x) for _ in range(noutputs)]
  if len(inputs) == 1:
    [inputs] = inputs
  if len(outputs) == 1:
    [outputs] = outputs
  return tf.keras.Model(inputs, outputs, name=name)


def _tf_module_as_submodule(name: str) -> tf.keras.Model:
  inputs = tf.keras.Input(shape=(4,))
  outputs = tf.keras.layers.Dense(2)(inputs)
  submodule = tf.Module(name=name)
  model = tf.keras.Model(inputs, outputs)
  model.submodule = submodule
  return model


class ModelExportTests(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters([
      dict(
          testcase_name="SingleInputOutput",
          model=_no_submodule(1, 1, "abc123"),
          output_names="output",
      ),
      dict(
          testcase_name="SingleInputMultipleOutput",
          model=_no_submodule(1, 2, "abc123"),
          output_names=["output_0", "output_1"],
      ),
      dict(
          testcase_name="MultipleInputSingleOutput",
          model=_no_submodule(2, 1, "abc123"),
          output_names="output",
      ),
      dict(
          testcase_name="MultipleInputOutput",
          model=_no_submodule(2, 2, "abc123"),
          output_names=["output_0", "output_1"],
      ),
  ])
  def test_keras_model_exporter(
      self,
      model: tf.keras.Model,
      output_names: Any):
    export_dir = self.create_tempdir()
    exporter = model_export.KerasModelExporter(output_names=output_names)
    exporter.save(None, model, export_dir)

    saved_model = tf.saved_model.load(export_dir)

    if tf.nest.is_nested(model.input):
      args = [tf.random.uniform([1] + i.shape[1:]) for i in model.input]
      kwargs = {i.name: v for i, v in zip(model.input, args)}
    else:  # Single input
      args = tf.random.uniform([1] + model.input.shape[1:])
      kwargs = {model.input.name: args}

    model_output = model(args)
    saved_model_outputs = saved_model.signatures["serving_default"](**kwargs)

    if tf.nest.is_nested(model_output):
      model_outputs = model_output
    else:  # Single output
      output_names = [output_names]
      model_outputs = [model_output]

    for output, name in zip(model_outputs, output_names):
      self.assertAllClose(
          output,
          saved_model_outputs[name], msg=f"Testing {name}...")

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
    exporter.save(None, model, export_dir)

    submodule = next(m for m in model.submodules if submodule_name == m.name)

    if subdirectory:
      directory = os.path.join(export_dir, subdirectory)
      saved_model = tf.saved_model.load(directory)
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
          expected_error=r"More than one submodule found \(\[.*\]\)",
      ),
      dict(
          testcase_name="NoSubmodule",
          model=_no_submodule(1, 1, "abc123"),
          submodule_name="abc123",
          expected_error="No submodule `abc123` found",
      ),
      dict(
          testcase_name="TFModuleAsSubmodule",
          model=_tf_module_as_submodule("abc123"),
          submodule_name="abc123",
          expected_error=r"Submodule \(.*\) is neither a Keras Model nor Layer",
      ),
  ])
  def test_submodule_exporter_fails(
      self,
      model: tf.keras.Model,
      submodule_name: str,
      expected_error: str):
    exporter = model_export.SubmoduleExporter(submodule_name)
    with self.assertRaisesRegex(ValueError, expected_error):
      exporter.save(None, model, self.create_tempdir())

  @parameterized.named_parameters([
      dict(
          testcase_name="KerasModelExporter",
          klass=model_export.KerasModelExporter,
      ),
      dict(
          testcase_name="SubmoduleExporter",
          klass=model_export.SubmoduleExporter,
      ),
  ])
  def test_protocol(self, klass: object):
    self.assertIsInstance(klass, orchestration.ModelExporter)


if __name__ == "__main__":
  tf.test.main()
