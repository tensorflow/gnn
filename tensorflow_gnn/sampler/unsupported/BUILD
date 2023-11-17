package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

load("@rules_proto//proto:defs.bzl", "proto_library")
load("@rules_python//python:proto.bzl", "py_proto_library")
load("@tensorflow_gnn//tensorflow_gnn:tensorflow_gnn.bzl", "pytype_strict_contrib_test")

proto_library(
    name = "subgraph_proto",
    srcs = ["subgraph.proto"],
    deps = ["@org_tensorflow//tensorflow/core:protos_all"],
)

py_proto_library(
    name = "subgraph_py_proto",
    deps = [
        ":subgraph_proto",
    ],
)

# Only the tests will be run by Bazel, and those will be run with no deps.
pytype_strict_contrib_test(
    name = "subgraph_test",
    srcs = ["subgraph_test.py"],
    python_version = "PY3",
    srcs_version = "PY3ONLY",
    deps = [],
)

pytype_strict_contrib_test(
    name = "sampling_lib_test",
    srcs = ["sampling_lib_test.py"],
    python_version = "PY3",
    srcs_version = "PY3ONLY",
    deps = [],
)

pytype_strict_contrib_test(
    name = "graph_sampler_test",
    srcs = ["graph_sampler_test.py"],
    data = [
        "@tensorflow_gnn//testdata/heterogeneous",
        "@tensorflow_gnn//testdata/homogeneous",
        "@tensorflow_gnn//testdata/node_vs_edge",
    ],
    python_version = "PY3",
    srcs_version = "PY3ONLY",
    deps = [],
)

pytype_strict_contrib_test(
    name = "sampling_utils_test",
    srcs = ["sampling_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3ONLY",
    deps = [],
)
