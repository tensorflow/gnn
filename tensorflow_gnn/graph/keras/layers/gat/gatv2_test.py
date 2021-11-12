from absl.testing import parameterized
import tensorflow as tf
import tensorflow_gnn as tfgnn

ct = tf.constant
rt = tf.ragged.constant


class GATv2Test(tf.test.TestCase, parameterized.TestCase):

  def testGAT_single_head(self):
    """Tests that a single-headed GAT is correct given predefined weights."""
    gt_input = tfgnn.GraphTensor.from_pieces(
        node_sets={
            'signals': tfgnn.NodeSet.from_fields(
                sizes=[3],
                # Node states have dimension 4.
                # The first three dimensions one-hot encode the node_id i.
                # The fourth dimension holds a distinct payload value 2**i.
                features={tfgnn.DEFAULT_STATE_NAME: ct(
                    [[1., 0., 0., 1.],
                     [0., 1., 0., 2.],
                     [0., 0., 1., 4.]])}),
        },
        edge_sets={
            'edges': tfgnn.EdgeSet.from_fields(
                # The edges contain a cycle 0->1->2->0 (let's call it clockwise)
                # and the reverse cycle 0->2->1->0 (counterclockwise).
                sizes=[6],
                adjacency=tfgnn.Adjacency.from_indices(
                    ('signals', ct([0, 1, 2, 0, 2, 1])),
                    ('signals', ct([1, 2, 0, 2, 1, 0])))),
        })

    log10 = tf.math.log(10.).numpy()
    got_gt = tfgnn.keras.layers.GATv2(
        num_heads=1,
        per_head_channels=4,
        edge_set_name='edges',
        attention_activation='relu',  # Let's keep it simple.
        # The space of attention computation of the single head has dimension 4.
        # The last dimension is used only in the key to carry the node's value,
        # multiplied by 11/10.
        # The first three dimensions are used to hand-construct attention scores
        # (see the running example below) that favor the counterclockwise
        # incoming edge over the other. Recall that weight matrices are
        # multiplied from the right onto batched inputs (in rows).
        #
        # For example, the query vector of node 0 is [0, 1, 0, 0], and ...
        query_kernel_initializer=tf.keras.initializers.Constant(
            [[0., 1., 0., 0.],
             [0., 0., 1., 0.],
             [1., 0., 0., 0.],
             [0., 0., 0., 0.]]),
        # ... the key vectors of node 1 and 2, resp., are [-1, 0, -1, 2.2]
        # and [-1, -1, 0, 4.4]. Therefore, ...
        key_kernel_initializer=tf.keras.initializers.Constant(
            [[0., -1., -1., 0.],
             [-1., 0., -1., 0.],
             [-1., -1., 0., 0.],
             [0., 0., 0., 1.1]]),
        # ... attention from node 0 to node 1 has a sum of key and query vector
        # [-1, 1, -1, 2.2], which gets turned by ReLU and the attention weights
        # below into a pre-softmax score of log(10). Likewise,
        # attention from node 0 to node 2 has a vector sum [-1, 0, 0, 4.4]
        # and pre-softmax score of 0. Altogether, this means: ...
        attention_kernel_initializers=tf.keras.initializers.Constant(
            [[log10], [log10], [log10], [0.]])
    )(gt_input)
    got = got_gt.node_sets['signals'][tfgnn.DEFAULT_STATE_NAME]

    # ... The softmax-weighed key vectors on the incoming edges of node 0
    # are  10/11 * [-1, 0, -1, 2.2]  +  1/11 * [-1, -1, 0, 4.4].
    # The final ReLU takes out the non-positive components and leaves 2 + 0.4
    # in the last component of the first row in the resulting node states.
    want = ct([[0., 0., 0., 2.4],  # Node 0.
               [0., 0., 0., 4.1],  # Node 1.
               [0., 0., 0., 1.2]])  # Node 2.
    self.assertAllEqual(got.shape, (3, 4))
    self.assertAllClose(got, want, atol=.0001)

  def testGAT_multihead(self):
    """Tests that a multi-headed GAT is correct given predefined weights."""
    gt_input = tfgnn.GraphTensor.from_pieces(
        node_sets={
            'signals': tfgnn.NodeSet.from_fields(
                sizes=[3],
                # The same node states as in the single_head test above.
                features={tfgnn.DEFAULT_STATE_NAME: ct(
                    [[1., 0., 0., 1.],
                     [0., 1., 0., 2.],
                     [0., 0., 1., 4.]])}),
        },
        edge_sets={
            'edges': tfgnn.EdgeSet.from_fields(
                # The same edges as in the single_head test above.
                sizes=[6],
                adjacency=tfgnn.Adjacency.from_indices(
                    ('signals', ct([0, 1, 2, 0, 2, 1])),
                    ('signals', ct([1, 2, 0, 2, 1, 0])))),
        })

    log10 = tf.math.log(10.).numpy()
    got_gt = tfgnn.keras.layers.GATv2(
        num_heads=2,
        per_head_channels=4,
        edge_set_name='edges',
        attention_activation=tf.keras.layers.LeakyReLU(alpha=0.0),
        # Attention head 0 uses the first four dimensions, which are used
        # in the same way as for the single_head test above.
        # Attention head 1 uses the last four dimensions, in which we
        # now favor the clockwise incoming edges and omit the scaling by 11/10.
        query_kernel_initializer=tf.keras.initializers.Constant(
            [[0., 1., 0., 0., 0., 0., 1., 0,],
             [0., 0., 1., 0., 1., 0., 0., 0.],
             [1., 0., 0., 0., 0., 1., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0.]]),
        key_kernel_initializer=tf.keras.initializers.Constant(
            [[0., -1., -1., 0., 0., -1., -1., 0.],
             [-1., 0., -1., 0., -1., 0., -1., 0.],
             [-1., -1., 0., 0., -1., -1., 0., 0.],
             [0., 0., 0., 1.1, 0., 0., 0., 1.]]),
        # Attention head 0 works out to softmax weights 10/11 and 1/11 as above.
        # Attention head 1 creates very large pre-softmax scores that
        # work out to weights 1 and 0 within floating-point precision.
        attention_kernel_initializers=tf.keras.initializers.Constant(
            [[log10, 100.],
             [log10, 100.],
             [log10, 100.],
             [0., 0.]])
    )(gt_input)
    got = got_gt.node_sets['signals'][tfgnn.DEFAULT_STATE_NAME]

    # Attention head 0 generates the first four outout dimensions as in the
    # single_head test above, with weights 10/11 and 1/11,
    # Attention head 1 uses weights 0 and 1 (note the reversed preference).
    want = ct([[0., 0., 0., 2.4, 0., 0., 0., 4.0],
               [0., 0., 0., 4.1, 0., 0., 0., 1.0],
               [0., 0., 0., 1.2, 0., 0., 0., 2.0]])
    self.assertAllEqual(got.shape, (3, 8))
    self.assertAllClose(got, want, atol=.0001)

  def testGAT_single_head_heterogeneous(self):
    """Tests that a single-headed GAT is correct on a heterogeneous graph."""
    gt_input = tfgnn.GraphTensor.from_pieces(
        node_sets={
            'signals':
                tfgnn.NodeSet.from_fields(
                    features={
                        tfgnn.DEFAULT_STATE_NAME: ct([[1.], [2.], [3.]]),
                    },
                    sizes=[3]),
            'signals2':
                tfgnn.NodeSet.from_fields(
                    features={
                        tfgnn.DEFAULT_STATE_NAME: ct([[1.], [2.], [3.]]),
                    },
                    sizes=[3]),
        },
        edge_sets={
            'edges':
                tfgnn.EdgeSet.from_fields(
                    sizes=[9],
                    adjacency=tfgnn.Adjacency.from_indices(
                        ('signals', ct([0, 0, 0, 1, 1, 1, 2, 2, 2])),
                        ('signals', ct([0, 1, 2, 0, 1, 2, 0, 1, 2])))),
            'edges2':
                tfgnn.EdgeSet.from_fields(
                    sizes=[9],
                    adjacency=tfgnn.Adjacency.from_indices(
                        ('signals', ct([0, 0, 0, 1, 1, 1, 2, 2, 2])),
                        ('signals2', ct([0, 1, 2, 0, 1, 2, 0, 1, 2])))),
        })

    model_homogeneous = tfgnn.keras.layers.GATv2(
        num_heads=1, per_head_channels=4, edge_set_name='edges')
    model_heterogeneous = tfgnn.keras.layers.GATv2(
        num_heads=1, per_head_channels=4, edge_set_name='edges2')
    # Initialize model weights.
    model_homogeneous(gt_input)
    model_heterogeneous(gt_input)
    # Transfer the weights from the homogeneous to the heterogeneous model.
    model_heterogeneous.set_weights(model_homogeneous.get_weights())

    # Run the same model on both 'edges' and 'edges2', which index different but
    # identical node sets.
    got_homogeneous = model_homogeneous(gt_input).node_sets['signals'][
        tfgnn.DEFAULT_STATE_NAME]
    got_heterogeneous = model_heterogeneous(gt_input).node_sets['signals2'][
        tfgnn.DEFAULT_STATE_NAME]

    # Ensure that these outputs are the same.
    self.assertAllClose(got_homogeneous, got_heterogeneous)

  def testGAT_multihead_equivalent_to_two_single_heads_constant(self):
    """Tests that a multihead GAT == concatenating two single head GATs."""
    gt_input = tfgnn.GraphTensor.from_pieces(
        node_sets={
            'signals':
                tfgnn.NodeSet.from_fields(
                    features={
                        tfgnn.DEFAULT_STATE_NAME: ct([[1.], [2.], [3.]]),
                    },
                    sizes=[3]),
        },
        edge_sets={
            'edges':
                tfgnn.EdgeSet.from_fields(
                    sizes=[9],
                    adjacency=tfgnn.Adjacency.from_indices(
                        ('signals', ct([0, 0, 0, 1, 1, 1, 2, 2, 2])),
                        ('signals', ct([0, 1, 2, 0, 1, 2, 0, 1, 2])))),
        })
    got_single_head_1 = tfgnn.keras.layers.GATv2(
        num_heads=1,
        per_head_channels=2,
        edge_set_name='edges',
        key_kernel_initializer=tf.keras.initializers.Constant([[1.], [3.]]),
        query_kernel_initializer=tf.keras.initializers.Constant([[2.], [4.]]),
        attention_kernel_initializers=tf.keras.initializers.Constant([[.1],
                                                                      [.2]]),
    )(gt_input).node_sets['signals'][tfgnn.DEFAULT_STATE_NAME]
    got_single_head_2 = tfgnn.keras.layers.GATv2(
        num_heads=1,
        per_head_channels=2,
        edge_set_name='edges',
        key_kernel_initializer=tf.keras.initializers.Constant([[5.], [7.]]),
        query_kernel_initializer=tf.keras.initializers.Constant([[6.], [8.]]),
        attention_kernel_initializers=tf.keras.initializers.Constant([[.3],
                                                                      [.4]]),
    )(gt_input).node_sets['signals'][tfgnn.DEFAULT_STATE_NAME]
    self.assertAllEqual(got_single_head_1.shape, (3, 2))
    self.assertAllEqual(got_single_head_2.shape, (3, 2))

    got_double_head = tfgnn.keras.layers.GATv2(
        num_heads=2,
        per_head_channels=2,
        edge_set_name='edges',
        key_kernel_initializer=tf.keras.initializers.Constant(
            [[1.], [3.], [5.], [7.]]),
        query_kernel_initializer=tf.keras.initializers.Constant(
            [[2.], [4.], [6.], [8.]]),
        attention_kernel_initializers=tf.keras.initializers.Constant(
            [[.1], [.3], [.2], [.4]]),
    )(gt_input).node_sets['signals'][tfgnn.DEFAULT_STATE_NAME]
    self.assertAllEqual(got_double_head.shape, (3, 4))

    self.assertAllClose(
        got_double_head,
        tf.concat([got_single_head_1, got_single_head_2], axis=-1))

  def testGAT_multihead_equivalent_to_two_single_heads_random_weights(self):
    """Tests that a multihead GAT == 2 single-head GATs, with random init."""
    gt_input = tfgnn.GraphTensor.from_pieces(
        node_sets={
            'signals':
                tfgnn.NodeSet.from_fields(
                    features={
                        tfgnn.DEFAULT_STATE_NAME: ct([[1.], [2.], [3.]]),
                    },
                    sizes=[3]),
        },
        edge_sets={
            'edges':
                tfgnn.EdgeSet.from_fields(
                    sizes=[9],
                    adjacency=tfgnn.Adjacency.from_indices(
                        ('signals', ct([0, 0, 0, 1, 1, 1, 2, 2, 2])),
                        ('signals', ct([0, 1, 2, 0, 1, 2, 0, 1, 2])))),
        })

    w_query_weights_1 = tf.keras.initializers.GlorotUniform(seed=42)((2,))
    w_query_weights_2 = tf.keras.initializers.GlorotUniform(seed=84)((2,))
    w_key_weights_1 = tf.keras.initializers.GlorotUniform(seed=1)((2,))
    w_key_weights_2 = tf.keras.initializers.GlorotUniform(seed=2)((2,))
    attention_logits_1 = tf.keras.initializers.GlorotUniform(seed=10)((2, 1))
    attention_logits_2 = tf.keras.initializers.GlorotUniform(seed=20)((2, 1))
    got_single_head_1 = tfgnn.keras.layers.GATv2(
        num_heads=1,
        per_head_channels=2,
        edge_set_name='edges',
        key_kernel_initializer=tf.keras.initializers.Constant(
            w_query_weights_1),
        query_kernel_initializer=tf.keras.initializers.Constant(
            w_key_weights_1),
        attention_kernel_initializers=tf.keras.initializers.Constant(
            attention_logits_1),
    )(gt_input).node_sets['signals'][tfgnn.DEFAULT_STATE_NAME]
    got_single_head_2 = tfgnn.keras.layers.GATv2(
        num_heads=1,
        per_head_channels=2,
        edge_set_name='edges',
        key_kernel_initializer=tf.keras.initializers.Constant(
            w_query_weights_2),
        query_kernel_initializer=tf.keras.initializers.Constant(
            w_key_weights_2),
        attention_kernel_initializers=tf.keras.initializers.Constant(
            attention_logits_2),
    )(gt_input).node_sets['signals'][tfgnn.DEFAULT_STATE_NAME]
    self.assertAllEqual(got_single_head_1.shape, (3, 2))
    self.assertAllEqual(got_single_head_2.shape, (3, 2))

    got_double_head = tfgnn.keras.layers.GATv2(
        num_heads=2,
        per_head_channels=2,
        edge_set_name='edges',
        key_kernel_initializer=tf.keras.initializers.Constant(
            tf.concat([w_query_weights_1, w_query_weights_2], axis=-1)),
        query_kernel_initializer=tf.keras.initializers.Constant(
            tf.concat([w_key_weights_1, w_key_weights_2], axis=-1)),
        attention_kernel_initializers=tf.keras.initializers.Constant(
            tf.concat([attention_logits_1, attention_logits_2], axis=-1)))(
                gt_input).node_sets['signals'][tfgnn.DEFAULT_STATE_NAME]
    self.assertAllEqual(got_double_head.shape, (3, 4))

    self.assertAllClose(
        got_double_head,
        tf.concat([got_single_head_1, got_single_head_2], axis=-1))

  @parameterized.named_parameters(
      ('single_head_batch',
       tfgnn.GraphTensor.from_pieces(
           node_sets={
               'signals':
                   tfgnn.NodeSet.from_fields(
                       features={
                           tfgnn.DEFAULT_STATE_NAME:
                               ct([[1., 2.], [2., 3.], [3., 4.]]),
                       },
                       sizes=[1, 2]),
           },
           edge_sets={
               'edges':
                   tfgnn.EdgeSet.from_fields(
                       sizes=[1, 4],
                       adjacency=tfgnn.Adjacency.from_indices(
                           ('signals', ct([0, 0, 0, 1, 1])),
                           ('signals', ct([0, 0, 1, 0, 1])))),
           }), 1, (3, 4)),
      ('multi_head',
       tfgnn.GraphTensor.from_pieces(
           node_sets={
               'signals':
                   tfgnn.NodeSet.from_fields(
                       features={
                           tfgnn.DEFAULT_STATE_NAME:
                               ct([[1., 2.], [2., 3.], [3., 4.]]),
                       },
                       sizes=[3]),
           },
           edge_sets={
               'edges':
                   tfgnn.EdgeSet.from_fields(
                       sizes=[9],
                       adjacency=tfgnn.Adjacency.from_indices(
                           ('signals', ct([0, 0, 0, 1, 1, 1, 2, 2, 2])),
                           ('signals', ct([0, 1, 2, 0, 1, 2, 0, 1, 2])))),
           }), 2, (3, 4)),
      ('single_head_batch_extra_dim',
       tfgnn.GraphTensor.from_pieces(
           node_sets={
               'signals':
                   tfgnn.NodeSet.from_fields(
                       features={
                           tfgnn.DEFAULT_STATE_NAME:
                               ct([[[1., 2.]], [[2., 3.]], [[3., 4.]]]),
                       },
                       sizes=[1, 2]),
           },
           edge_sets={
               'edges':
                   tfgnn.EdgeSet.from_fields(
                       sizes=[1, 4],
                       adjacency=tfgnn.Adjacency.from_indices(
                           ('signals', ct([0, 0, 0, 1, 1])),
                           ('signals', ct([0, 0, 1, 0, 1])))),
           }), 1, (3, 1, 4)),
      ('multi_head_extra_dim',
       tfgnn.GraphTensor.from_pieces(
           node_sets={
               'signals':
                   tfgnn.NodeSet.from_fields(
                       features={
                           tfgnn.DEFAULT_STATE_NAME:
                               ct([[[1., 2.]], [[2., 3.]], [[3., 4.]]]),
                       },
                       sizes=[3]),
           },
           edge_sets={
               'edges':
                   tfgnn.EdgeSet.from_fields(
                       sizes=[9],
                       adjacency=tfgnn.Adjacency.from_indices(
                           ('signals', ct([0, 0, 0, 1, 1, 1, 2, 2, 2])),
                           ('signals', ct([0, 1, 2, 0, 1, 2, 0, 1, 2])))),
           }), 2, (3, 1, 4)))
  def testGAT_shapes(self, gt_input, num_heads, want_shape):
    """Tests that the output shapes are as expected."""
    got = tfgnn.keras.layers.GATv2(
        num_heads=num_heads,
        per_head_channels=4 // num_heads,
        edge_set_name='edges',
    )(
        gt_input)
    self.assertAllEqual(
        got.node_sets['signals'][tfgnn.DEFAULT_STATE_NAME].shape, want_shape)

  def testGAT_dropout(self):
    """Sanity tests that a GAT with dropout behaves as expected."""
    gt_input = tfgnn.GraphTensor.from_pieces(
        node_sets={
            'signals':
                tfgnn.NodeSet.from_fields(
                    features={
                        tfgnn.DEFAULT_STATE_NAME: ct([[1.], [2.], [3.]]),
                    },
                    sizes=[3]),
        },
        edge_sets={
            'edges':
                tfgnn.EdgeSet.from_fields(
                    sizes=[9],
                    adjacency=tfgnn.Adjacency.from_indices(
                        ('signals', ct([0, 0, 0, 1, 1, 1, 2, 2, 2])),
                        ('signals', ct([0, 1, 2, 0, 1, 2, 0, 1, 2])))),
        })

    tf.random.set_seed(42)

    # Run a GAT that is almost guaranteed to drop all edges.
    full_dropout_gat = tfgnn.keras.layers.GATv2(
        num_heads=1,
        per_head_channels=4,
        edge_set_name='edges',
        edge_dropout=.999999,
        key_kernel_initializer=tf.keras.initializers.Constant(
            [[1.], [3.], [5.], [7.]]),
        query_kernel_initializer=tf.keras.initializers.Constant(
            [[2.], [4.], [6.], [8.]]),
        attention_kernel_initializers=tf.keras.initializers.Constant(
            [[.1], [.2], [.3], [.4]]))
    # Get results for training (dropout enabled) & evaluation (dropout disabled)
    got_with_dropout = full_dropout_gat(
        gt_input, training=True).node_sets['signals'][tfgnn.DEFAULT_STATE_NAME]
    got_without_dropout = full_dropout_gat(
        gt_input, training=False).node_sets['signals'][tfgnn.DEFAULT_STATE_NAME]

    # Ensure that the GAT without dropout has correct values, and that the GAT
    # with dropout is just zeros.
    self.assertAllClose(
        got_without_dropout,
        ct([[2.9932172, 8.979652, 14.966086, 20.95252],
            [2.9932172, 8.979652, 14.966086, 20.95252],
            [2.9932172, 8.979652, 14.966086, 20.95252]]))
    self.assertAllClose(got_with_dropout, tf.zeros((3, 4)))

    # Now, create a GAT with 50% edge dropout.
    partial_dropout_gat = tfgnn.keras.layers.GATv2(
        num_heads=1,
        per_head_channels=4,
        edge_set_name='edges',
        edge_dropout=.5,
        key_kernel_initializer=tf.keras.initializers.Constant(
            [[1.], [3.], [5.], [7.]]),
        query_kernel_initializer=tf.keras.initializers.Constant(
            [[2.], [4.], [6.], [8.]]),
        attention_kernel_initializers=tf.keras.initializers.Constant(
            [[.1], [.2], [.3], [.4]]))
    # Get results for training (dropout enabled) & evaluation (dropout disabled)
    partial_with_dropout = partial_dropout_gat(
        gt_input, training=True).node_sets['signals'][tfgnn.DEFAULT_STATE_NAME]
    partial_without_dropout = partial_dropout_gat(
        gt_input, training=False).node_sets['signals'][tfgnn.DEFAULT_STATE_NAME]

    # Ensure the 50% dropout model is equal to the GAT without dropout when
    # training.
    self.assertAllClose(partial_without_dropout, got_without_dropout)
    # Ensure the 50% dropout model is not equal to the GAT without dropout when
    # evaluating..
    self.assertNotAllClose(partial_without_dropout, partial_with_dropout)

  def testGAT_get_config(self):
    """Tests that the get_config call returns all expected parameters."""
    key_kernel_initializer = tf.keras.initializers.Constant([[1.], [3.], [5.],
                                                             [7.]])
    query_kernel_initializer = tf.keras.initializers.Constant([[2.], [4.], [6.],
                                                               [8.]])
    attention_kernel_initializers = tf.keras.initializers.Constant([[.1], [.2],
                                                                    [.3], [.4]])
    model = tfgnn.keras.layers.GATv2(
        num_heads=1,
        per_head_channels=4,
        edge_set_name='edges',
        attention_activation=tf.keras.layers.LeakyReLU(alpha=0.5, name='sigma'),
        key_kernel_initializer=key_kernel_initializer,
        query_kernel_initializer=query_kernel_initializer,
        attention_kernel_initializers=attention_kernel_initializers,
        name='foo')

    self.assertDictEqual(
        model.get_config(), {
            'edge_dropout': 0.0,
            'attention_activation': {
                'class_name': 'LeakyReLU',
                'config': {'name': 'sigma', 'trainable': True,
                           'dtype': 'float32', 'alpha': 0.5}},
            'edge_set_name': 'edges',
            'feature_name': 'hidden_state',
            'query_kernel_initializer': query_kernel_initializer,
            'num_heads': 1,
            'output_feature_name': tfgnn.DEFAULT_STATE_NAME,
            'per_head_channels': 4,
            'key_kernel_initializer': key_kernel_initializer,
            'attention_kernel_initializers': attention_kernel_initializers,
            'use_bias': True,
            'dtype': 'float32',
            'name': 'foo',
            'trainable': True
        })

  def testGAT_invalid_shape(self):
    """Tests that a non-scalar GraphTensor raises a ValueError."""
    input_gt = tfgnn.GraphTensor.from_pieces(
        node_sets={
            'signals':
                tfgnn.NodeSet.from_fields(
                    features={
                        tfgnn.DEFAULT_STATE_NAME:
                            rt([[1., 5., 3., 2., 4.], [1., 2., 3.]]),
                    },
                    sizes=[[5], [3]]),
        })
    with self.assertRaisesRegex(ValueError, 'had rank 1'):
      tfgnn.keras.layers.GATv2(
          num_heads=1,
          per_head_channels=4,
          edge_set_name='edges',
      )(
          input_gt)

  def testGAT_invalid_edge_dropout(self):
    """Tests that an edge dropout of too high or low raises a ValueError."""
    with self.assertRaisesRegex(ValueError, 'Edge dropout -1'):
      tfgnn.keras.layers.GATv2(
          num_heads=1,
          per_head_channels=4,
          edge_dropout=-1,
          edge_set_name='edges')
    with self.assertRaisesRegex(ValueError, 'Edge dropout 1'):
      tfgnn.keras.layers.GATv2(
          num_heads=1,
          per_head_channels=4,
          edge_dropout=1,
          edge_set_name='edges')

  def testGAT_invalid_num_heads(self):
    """Tests that a non-positive num_heads raises a ValueError."""
    with self.assertRaisesRegex(ValueError, 'Number of heads 0'):
      tfgnn.keras.layers.GATv2(
          num_heads=0, per_head_channels=4, edge_set_name='edges')
    with self.assertRaisesRegex(ValueError, 'Number of heads -1'):
      tfgnn.keras.layers.GATv2(
          num_heads=-1, per_head_channels=4, edge_set_name='edges')

  def testGAT_invalid_num_channels(self):
    """Tests that a non-positive num_channels raises a ValueError."""
    with self.assertRaisesRegex(ValueError, 'Per-head channels 0'):
      tfgnn.keras.layers.GATv2(
          num_heads=1, per_head_channels=0, edge_set_name='edges')
    with self.assertRaisesRegex(ValueError, 'Per-head channels -1'):
      tfgnn.keras.layers.GATv2(
          num_heads=1, per_head_channels=-1, edge_set_name='edges')


if __name__ == '__main__':
  tf.test.main()
