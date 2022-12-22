"""Dependency on TensorFlow for build."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def tf_setup():
    """Define tensorflow>=2.8.0 dependency for Bazel build."""
    http_archive(
        name = "org_tensorflow",
        sha256 = "99c732b92b1b37fc243a559e02f9aef5671771e272758aa4aec7f34dc92dac48",
        urls = [
            "https://github.com/tensorflow/tensorflow/archive/refs/tags/v2.11.0.tar.gz",
        ],
        strip_prefix = "tensorflow-2.11.0",
    )
