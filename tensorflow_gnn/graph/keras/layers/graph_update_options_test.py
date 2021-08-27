"""Tests for GraphUpdateOptions and its pieces.

These tests cover only generalities of storage and field access.
The tests for GraphUpdate and its pieces cover the use of individual options.
"""

from absl.testing import absltest

from tensorflow_gnn.graph.keras.layers import graph_update_options as opt


class GraphUpdateOptionsTest(absltest.TestCase):

  def testEdgeSetOptions(self):
    options = opt.GraphUpdateEdgeSetOptions(update_combiner_fn="concatenate")
    options.update_output_feature = "extra_state"
    self.assertEqual(options.update_combiner_fn, "concatenate")
    self.assertEqual(options.update_output_feature, "extra_state")

  def testNodeSetOptions(self):
    options = opt.GraphUpdateNodeSetOptions(update_combiner_fn="concatenate")
    self.assertEqual(options.update_combiner_fn, "concatenate")
    self.assertIsNone(options.update_output_feature)

  def testContextOptions(self):
    options = opt.GraphUpdateNodeSetOptions()
    options.update_output_feature = "extra_state"
    self.assertEqual(options.update_output_feature, "extra_state")

  def testGraphOptionsData(self):
    options = opt.GraphUpdateOptions(update_context=False)
    self.assertEqual(options.update_context, False)

  def testGraphOptionsNodeSets(self):
    options = opt.GraphUpdateOptions()
    # Set values for "users".
    options.node_sets["users"].update_output_feature = "user_state"
    options.node_sets["users"].context_pool_factory = lambda: "u_pooler"
    # Set default values.
    options.node_set_default.update_output_feature = "state"
    options.node_set_default.update_combiner_fn = "concatenate"
    # Set values for "channels".
    options.node_sets["channels"].update_output_feature = "channel_state"
    options.node_sets["channels"].context_pool_factory = lambda: "c_pooler"

    # update_output_feature is set in all places.
    self.assertEqual(
        options.node_set_default.update_output_feature,
        "state")
    self.assertEqual(
        options.node_sets.get("users").update_output_feature,
        "user_state")
    self.assertEqual(
        options.node_set_with_defaults("users").update_output_feature,
        "user_state")
    self.assertEqual(
        options.node_sets.get("channels").update_output_feature,
        "channel_state")
    self.assertEqual(
        options.node_set_with_defaults("channels").update_output_feature,
        "channel_state")

    # update_combiner_fn is only set in defaults.
    self.assertEqual(
        options.node_set_default.update_combiner_fn,
        "concatenate")
    self.assertIsNone(options.node_sets.get("users").update_combiner_fn)
    self.assertEqual(
        options.node_set_with_defaults("users").update_combiner_fn,
        "concatenate")
    # ...and access "with_defaults" does not change that:
    self.assertIsNone(options.node_sets.get("users").update_combiner_fn)
    self.assertIsNone(options.node_sets.get("channels").update_combiner_fn)
    self.assertEqual(
        options.node_set_with_defaults("channels").update_combiner_fn,
        "concatenate")
    self.assertIsNone(options.node_sets.get("channels").update_combiner_fn)

    # context_pool_factory is set separately per node set.
    self.assertIsNone(options.node_set_default.context_pool_factory)
    self.assertEqual(
        options.node_sets.get("users").context_pool_factory(),
        "u_pooler")
    self.assertEqual(
        options.node_set_with_defaults("users").context_pool_factory(),
        "u_pooler")
    self.assertEqual(
        options.node_sets.get("channels").context_pool_factory(),
        "c_pooler")
    self.assertEqual(
        options.node_set_with_defaults("channels").context_pool_factory(),
        "c_pooler")

  def testGraphOptionsEdgeSets(self):
    # The implementation was tested for node sets already,
    # so it is enough to do a simple smoke test.
    options = opt.GraphUpdateOptions()
    options.edge_sets["edges"].update_output_feature = "edge_state"
    options.edge_set_default.update_combiner_fn = "concatenate"

    self.assertEqual(
        options.edge_sets.get("edges").update_output_feature,
        "edge_state")
    self.assertIsNone(options.edge_set_default.update_output_feature)
    self.assertEqual(
        options.edge_set_with_defaults("edges").update_output_feature,
        "edge_state")

    self.assertIsNone(options.edge_sets.get("edges").update_combiner_fn)
    self.assertEqual(
        options.edge_set_default.update_combiner_fn,
        "concatenate")
    self.assertEqual(
        options.edge_set_with_defaults("edges").update_combiner_fn,
        "concatenate")
    self.assertIsNone(options.edge_sets.get("edges").update_combiner_fn)

  def testGraphOptionsEqual(self):
    def _get_options():
      options = opt.GraphUpdateOptions()
      options.update_context = True
      options.context.update_output_feature = "ctx_state"
      options.node_sets["users"].update_output_feature = "user_state"
      return options

    # Difference in a top-level field.
    options1 = _get_options()
    options2 = _get_options()
    # pylint: disable=g-generic-assert
    # Let's spell out == and != like users do.
    self.assertTrue(options1 == options2)
    self.assertFalse(options1 != options2)
    options2.update_context = False
    self.assertFalse(options1 == options2)
    self.assertTrue(options1 != options2)

    # Difference in a direct subobject.
    options2 = _get_options()
    self.assertTrue(options1 == options2)
    options2.context.update_output_feature = "other_state"
    self.assertFalse(options1 == options2)

    # Difference in a subobject from a dict.
    options2 = _get_options()
    self.assertTrue(options1 == options2)
    options2.node_sets["users"].update_output_feature = "other_state"
    self.assertFalse(options1 == options2)

    # defaultdict doesn't change equality until new entries are written.
    options2 = _get_options()
    self.assertTrue(options1 == options2)
    channel_options = options2.node_sets["channels"]
    self.assertTrue(options1 == options2)
    channel_options.update_output_feature = "channel_state"
    self.assertFalse(options1 == options2)


if __name__ == "__main__":
  absltest.main()
