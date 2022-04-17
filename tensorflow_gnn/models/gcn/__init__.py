"""The tensorflow_gnn.models.gcn package."""
from tensorflow_gnn.models.gcn import gcn_conv

GCNConv = gcn_conv.GCNConv
GCNConvGraphUpdate = gcn_conv.GCNConvGraphUpdate

del gcn_conv
