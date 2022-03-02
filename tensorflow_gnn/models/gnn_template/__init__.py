"""The tensorflow_gnn.models.gnn_template package."""

from tensorflow_gnn.models.gnn_template import modeling

vanilla_mpnn_model = modeling.vanilla_mpnn_model
init_states_by_embed_and_transform = modeling.init_states_by_embed_and_transform
pass_simple_messages = modeling.pass_simple_messages

# Prune imported module symbols so they're not accessible implicitly.
del modeling
