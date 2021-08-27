"""The tfgnn.keras.utils package."""

from tensorflow_gnn.graph.keras.utils import fnn_factory

get_fnn_factory = fnn_factory.get_fnn_factory

# Prune imported module symbols so they're not accessible implicitly.
del fnn_factory
