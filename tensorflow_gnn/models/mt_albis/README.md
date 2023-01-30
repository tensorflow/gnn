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
    simply by averaging (TODO(b/265760014): or other fixed expressions),
    or with attention (that is, a trained, data-dependent weighting);
  * whether to use residual connections for updating node states;
  * how to normalize hidden activations.

Like other TF-GNN models, MtAlbis is used in your GNN model by calling
a `MtAlbisGraphUpdate` layer for each round of message passing.
See its docstring for a detailed documentation of the available
hyperparameters.

TODO(b/261835577): Code examples and instructions for hyperparameter tuning
are forthcoming.

## Maintenance and stability

For now, this code is experimental, with no promises of stability yet.
We intend to develop a stable version of `MtAlbis`, eventually freezing its
selection of architectures. Further evolution of TF-GNN's model template
would then happen under a new code name, `MtB...`.
