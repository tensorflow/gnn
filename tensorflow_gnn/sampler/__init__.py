"""Public interface for GNN Sampler."""
from tensorflow_gnn.sampler import sampling_spec_builder
from tensorflow_gnn.sampler import sampling_spec_pb2


SamplingSpec = sampling_spec_pb2.SamplingSpec
SamplingSpecBuilder = sampling_spec_builder.SamplingSpecBuilder

del sampling_spec_pb2
del sampling_spec_builder
