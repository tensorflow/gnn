"""Public interface for GNN Sampler."""
from tensorflow_gnn.sampler import sampling_spec_builder
from tensorflow_gnn.sampler import sampling_spec_pb2

SamplingOp = sampling_spec_pb2.SamplingOp
SamplingSpec = sampling_spec_pb2.SamplingSpec
SamplingSpecBuilder = sampling_spec_builder.SamplingSpecBuilder
make_sampling_spec_tree = sampling_spec_builder.make_sampling_spec_tree

del sampling_spec_pb2
del sampling_spec_builder
