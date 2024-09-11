# Model Template "Albis"

## Overview

Looking for guidance how build a GNN model?

A TF-GNN **model template** provides a selection of field-tested
GNN architectures, accompanied by instructions for users how to
choose between them and tune the respective hyperparameters.

This model template is code-named `MtAlbis` _(Model Template "Albis")_ .
It works on heterogeneous graphs (that is, graphs subdivided into multiple
node sets and edge sets) by passing messages along edges and updating node
states from incoming messages. Its main architectural choices are:

  * how to aggregate the incoming messages from each node set:
      * by element-wise averaging (reduce type `"mean"`),
      * by a concatenation of the average with other fixed expressions
        (e.g., `"mean|max_no_inf"`, `"mean|sum"`), or
      * with attention, that is, a trained, data-dependent weighting;
  * whether to use residual connections for updating node states;
  * if and how to normalize node states.

Like other TF-GNN models, MtAlbis is used in your GNN model by calling
a `MtAlbisGraphUpdate` layer for each round of message passing.
See its docstring for a detailed documentation of the available
hyperparameters.

## How to run

For an end-to-end example of training one instance of `MtAlbis`, see
[tensorflow_gnn/runner/examples/ogbn/mag/train.py](../../runner/examples/ogbn/mag/train.py).

For a *declaration* of hyperparameter searches with
[Vizier](https://github.com/google/vizier), see
[tensorflow_gnn/models/mt_albis/hparams_vizier.py](./hparams_vizier.py).
At this time, it is left to OSS/Cloud users to create their own orchestration
for running a `train.py` script for various hyperparameters, report the
resulting quality back to Vizier, and receive new hyperparameters to try.

## Hyperparameter tuning strategy (sketch)

We recommend to first tune `MtAlbis` without using any attention
(`attention_type="none"`), using the Adam optimizer with a CosineDecay of the
learning rate as seen in the example above. Not using attention makes it easier
and faster to train, and provides a useful baseline. We recommend to leave
`edge_set_combine_type` at its default `"concat"`, unless there are node sets
that receive messages from an unusually large number of edge sets, in which case
`"sum"` helps to save model weights.

If enough training runs are affordable (possibly on a subset of the data), we
invite users to tune all other hyperparameters. If not, it may make sense to
restrict

  * `simple_conv_reduce_type` to the values `"mean"` and `"mean|sum"`,
  * `normalization_type` to the fixed value `"layer"`,
  * `edge_dropout_rate` to `0` (emulating VanillaMPNN) or to the same
    tunable number as `state_dropout_rate` – crude but effective.
    (State dropout is the conventional dropout of individual entries in
    the network's hidden activations, while edge dropout discards the
    entire messages from a random subset of edges.)

Attention can be enabled by setting `attention_type`, possibly restricted to
a list of `attention_edge_set_names`. It introduces the new hyperparameter
`attention_num_heads` and (if applied to all edge sets) disables the
hyperparameter `simple_conv_reduce_type`. We recommend to evaluate attention
models carefully, relative to the non-attention baseline. Not all problems
benefit from attention, even those that do may become more sensitive to skews
between the training and inference distributions.

## API stability

The public API of this model (`from tensorflow_gnn.models import mt_albis`)
is covered by [semantic versioning](https://semver.org/spec/v2.0.0.html) of
TensorFlow GNN's open-source releases: new minor versions do not break existing
users.

Major changes to the architecture or the like would happen under a new code name
`MtB...`.

<!-- PLACEHOLDER FOR README GOOGLE EXTRAS -->
