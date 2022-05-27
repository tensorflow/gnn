"""The tfgnn.keras package."""

from tensorflow_gnn.keras import builders  #  To register the types.
from tensorflow_gnn.keras import keras_tensors  # To register the types.
from tensorflow_gnn.keras import layers  # Provided as subpackage.

ConvGNNBuilder = builders.ConvGNNBuilder

# Prune imported module symbols so they're not accessible implicitly,
# except those meant to be used as subpackages, like tfgnn.keras.layers.
# Please use the same order as for the import statements at the top.
del builders
del keras_tensors
