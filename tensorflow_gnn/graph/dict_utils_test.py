"""Tests for dict_utils."""

from absl.testing import absltest
from tensorflow_gnn.graph import dict_utils


class KeyPrefixTest(absltest.TestCase):

  def testWithKeyPrefix(self):
    d1 = {"a": 1, "b": 2}
    d2 = dict_utils.with_key_prefix(d1, "p/")
    self.assertDictEqual(d1, {"a": 1, "b": 2})  # Unchanged.
    self.assertDictEqual(d2, {"p/a": 1, "p/b": 2})

  def testPopByPrefix(self):
    d1 = {"p/a": 1, "p/b": 2, "q/c": 3, "q/d": 4}
    d2 = dict_utils.pop_by_prefix(d1, "p/")
    self.assertDictEqual(d1, {"q/c": 3, "q/d": 4})  # Changed in-place.
    self.assertDictEqual(d2, {"a": 1, "b": 2})


if __name__ == "__main__":
  absltest.main()
