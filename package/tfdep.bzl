"""Dependency on TensorFlow for build."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def tf_setup():
    """Define tensorflow>=2.8.0 dependency for Bazel build."""
    http_archive(
        name = "org_tensorflow",
        sha256 = "66b953ae7fba61fd78969a2e24e350b26ec116cf2e6a7eb93d02c63939c6f9f7",
        urls = [
            "https://github.com/tensorflow/tensorflow/archive/refs/tags/v2.8.0.tar.gz",
        ],
        strip_prefix = "tensorflow-2.8.0",
    )
