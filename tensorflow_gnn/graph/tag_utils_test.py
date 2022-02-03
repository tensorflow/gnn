"""Tests for tag_utils."""

from absl.testing import absltest
from absl.testing import parameterized

from tensorflow_gnn.graph import graph_constants as const
from tensorflow_gnn.graph import tag_utils


class ReverseTagTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("Source", const.SOURCE, const.TARGET),
      ("Target", const.TARGET, const.SOURCE))
  def test(self, tag, expected):
    actual = tag_utils.reverse_tag(tag)
    self.assertEqual(expected, actual)

  def testError(self):
    with self.assertRaisesRegex(ValueError, r"Expected tag .*got: 3"):
      _ = tag_utils.reverse_tag(3)


if __name__ == "__main__":
  absltest.main()
