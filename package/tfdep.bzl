"""Dependency on TensorFlow for build."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def tf_setup():
    """Define tensorflow>=2.6.0 dependency for Bazel build."""
    http_archive(
        name = "org_tensorflow",
        sha256 = "41b32eeaddcbc02b0583660bcf508469550e4cd0f86b22d2abe72dfebeacde0f",
        urls = [
            "https://github.com/tensorflow/tensorflow/archive/refs/tags/v2.6.0.tar.gz",
        ],
        strip_prefix = "tensorflow-2.6.0",
    )
