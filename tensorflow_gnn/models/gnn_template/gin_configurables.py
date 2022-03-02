"""Gin bindings for the public symbols of model gnn_template."""

import gin
from tensorflow_gnn.models import gnn_template

gin.external_configurable(gnn_template.vanilla_mpnn_model,
                          "VanillaMPNNModel")
gin.external_configurable(gnn_template.init_states_by_embed_and_transform,
                          "InitStatesByEmbedAndTransform")
gin.external_configurable(gnn_template.pass_simple_messages,
                          "PassSimpleMessages")
