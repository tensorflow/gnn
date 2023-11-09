"""Dependency on TensorFlow for build."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def tf_setup():
    """Define tensorflow>=2.13.0 dependency for Bazel build.

    This downloads the TensorFlow files required for building TFGNN protos
    (examples and graph_schema).This TF version should always be within our supported range and gets
    updated manually as our TF dependency advances. This version is somewhat flexible since TFGNN
    protos only depend on tensorflow.Example, tensorflow.Feature, tensorflow.TensorShapeProto, and
    tensorflow.DataType, which are all very stable definitions in TensorFlow.
    """
    http_archive(
        name = "org_tensorflow",
        sha256 = "e58c939079588623e6fa1d054aec2f90f95018266e0a970fd353a5244f5173dc",
        urls = [
            "https://github.com/tensorflow/tensorflow/archive/refs/tags/v2.13.0.tar.gz",
        ],
        strip_prefix = "tensorflow-2.13.0",
    )
