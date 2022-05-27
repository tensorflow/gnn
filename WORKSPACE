workspace(name = "tensorflow_gnn")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# Define the TensorFlow archive.
load("@tensorflow_gnn//package:tfdep.bzl", "tf_setup")
tf_setup()

# Initialize the TensorFlow repository and all dependencies.
# See @org_tensorflow//WORKSPACE for details.
load("@org_tensorflow//tensorflow:workspace3.bzl", "tf_workspace3")
tf_workspace3()
load("@org_tensorflow//tensorflow:workspace2.bzl", "tf_workspace2")
tf_workspace2()
load("@org_tensorflow//tensorflow:workspace1.bzl", "tf_workspace1")
tf_workspace1()
load("@org_tensorflow//tensorflow:workspace0.bzl", "tf_workspace0")
tf_workspace0()

# This seems required to avoid a proxy setup.
# TODO(blais): Find a way to get rid of this.
http_archive(
    name = "io_bazel_rules_docker",
    sha256 = "1698624e878b0607052ae6131aa216d45ebb63871ec497f26c67455b34119c80",
    strip_prefix = "rules_docker-0.15.0",
    urls = ["https://github.com/bazelbuild/rules_docker/archive/v0.15.0.tar.gz"],
)
