"""Dependency on TensorFlow for build."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def tf_setup():
    """Define tensorflow>=2.11.0 dependency for Bazel build.

    This downloads the TensorFlow files required for building TFGNN protos
    (examples and graph_schema).This TF version should always be within our supported range and gets
    updated manually as our TF dependency advances. This version is somewhat flexible since TFGNN
    protos only depend on tensorflow.Example, tensorflow.Feature, tensorflow.TensorShapeProto, and
    tensorflow.DataType, which are all very stable definitions in TensorFlow.
    """
    http_archive(
        name = "org_tensorflow",
        sha256 = "c030cb1905bff1d2446615992aad8d8d85cbe90c4fb625cee458c63bf466bc8e",
        urls = [
            "https://github.com/tensorflow/tensorflow/archive/refs/tags/v2.12.0.tar.gz",
        ],
        strip_prefix = "tensorflow-2.12.0",
    )
