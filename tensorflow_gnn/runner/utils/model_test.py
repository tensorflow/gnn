"""Tests for model."""
import itertools
from typing import Any, Mapping, Sequence, Union

from absl.testing import parameterized
import tensorflow as tf
from tensorflow_gnn.runner.utils import model as model_utils

NestedStructure = Union[Any, Sequence[Any], Mapping[str, Any]]


def model(input_shape: Sequence[int],
          input_structure: NestedStructure,
          output_structure: NestedStructure,
          *,
          tag_value: int = 1) -> tf.keras.Model:
  inputs = tf.nest.map_structure(
      lambda _: tf.keras.Input(shape=input_shape),
      input_structure)
  tensors = itertools.cycle(tf.nest.flatten(inputs))
  offsets = itertools.count(tag_value)
  outputs = tf.nest.map_structure(
      lambda _: next(tensors) + next(offsets),
      output_structure)
  return tf.keras.Model(inputs, outputs)


class ModelTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters([
      dict(
          testcase_name="ChainAtom",
          m1=model((4,), None, None, tag_value=8191),
          m2=model((4,), None, None, tag_value=4),
          inputs=tf.range(0, 4),
          expected_outputs=tf.range(0, 4) + 8191 + 4,
      ),
      dict(
          testcase_name="ChainMapping",
          m1=model((4,), None, {"a": None, "b": None}, tag_value=8191),
          m2=model(
              (4,),
              {"a": None, "b": None},
              {"a": None, "b": None},
              tag_value=4),
          inputs=tf.range(0, 4),
          expected_outputs={
              "a": tf.range(0, 4) + 8191 + 4,
              "b": tf.range(0, 4) + 8191 + 4 + 2,
          },
      ),
      dict(
          testcase_name="ChainSequence",
          m1=model((4,), None, [None, None], tag_value=8191),
          m2=model((4,), None, None, tag_value=4),
          inputs=tf.range(0, 4),
          expected_outputs=tf.range(0, 4) + 8191 + 4,
      ),
      dict(
          testcase_name="ChainAtomFromSequence",
          m1=model((4,), None, [None, None], tag_value=8191),
          m2=model((4,), None, None, tag_value=4),
          inputs=tf.range(0, 4),
          expected_outputs=tf.range(0, 4) + 8191 + 4,
      ),
      dict(
          testcase_name="ChainMappingFromSequence",
          m1=model((4,), None, [{"a": None, "b": None}, None], tag_value=8191),
          m2=model((4,), {"a": None, "b": None}, [None, None], tag_value=4),
          inputs=tf.range(0, 4),
          expected_outputs=[
              tf.range(0, 4) + 8191 + 4,
              tf.range(0, 4) + 8191 + 4 + 2,
          ],
      ),
      dict(
          testcase_name="ChainSequenceFromSequence",
          m1=model((4,), None, [[None, None], None], tag_value=8191),
          m2=model((4,), [None, None], [None, None], tag_value=4),
          inputs=tf.range(0, 4),
          expected_outputs=[
              tf.range(0, 4) + 8191 + 4,
              tf.range(0, 4) + 8191 + 4 + 2,
          ],
      ),
  ])
  def test_chain(
      self,
      m1: tf.keras.Model,
      m2: tf.keras.Model,
      inputs: tf.Tensor,
      expected_outputs: Union[tf.Tensor, Sequence[tf.Tensor]]):
    actual = model_utils.chain(m1, m2)
    self.assertAllClose(actual(inputs), expected_outputs)


if __name__ == "__main__":
  tf.test.main()
