package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

load("@com_google_protobuf//:protobuf.bzl", "py_proto_library")

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
    ],
)
