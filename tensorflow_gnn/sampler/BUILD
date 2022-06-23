package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

load("@com_google_protobuf//:protobuf.bzl", "py_proto_library")
load("@tensorflow_gnn//tensorflow_gnn:tensorflow_gnn.bzl", "pytype_strict_contrib_test")

py_proto_library(
    name = "subgraph_py_proto",
    srcs = ["subgraph.proto"],
    default_runtime = "@com_google_protobuf//:protobuf_python",
    protoc = "@com_google_protobuf//:protoc",
    srcs_version = "PY3",
    deps = [
        "@com_google_protobuf//:protobuf_python",
        "@org_tensorflow//tensorflow/core:protos_all_py",
    ],
)

py_proto_library(
    name = "sampling_spec_py_proto",
    srcs = ["sampling_spec.proto"],
    default_runtime = "@com_google_protobuf//:protobuf_python",
    protoc = "@com_google_protobuf//:protoc",
    srcs_version = "PY3",
    deps = [
        "@com_google_protobuf//:protobuf_python",
        "@org_tensorflow//tensorflow/core:protos_all_py",
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
