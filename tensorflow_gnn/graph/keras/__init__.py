"""The tfgnn.keras package."""

from tensorflow_gnn.graph.keras import keras_tensors  # To register the types.
from tensorflow_gnn.graph.keras import layers  # Provided as subpackage.
from tensorflow_gnn.graph.keras import utils  # Provided as subpackage.

# Prune imported module symbols so they're not accessible implicitly,
# except those meant to be used as subpackages, like tfgnn.keras.layers.
del keras_tensors
