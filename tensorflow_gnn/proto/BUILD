package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

load("@com_google_protobuf//:protobuf.bzl", "py_proto_library")

py_proto_library(
    name = "graph_schema_py_proto",
    srcs = ["graph_schema.proto"],
    default_runtime = "@com_google_protobuf//:protobuf_python",
    protoc = "@com_google_protobuf//:protoc",
    srcs_version = "PY3",
    deps = [
        "@com_google_protobuf//:protobuf_python",
        "@org_tensorflow//tensorflow/core:protos_all_py",
    ],
)

py_library(
    name = "graph_schema",
    srcs = ["graph_schema.py"],
    srcs_version = "PY3",
    deps = [
        ":graph_schema_py_proto",
    ],
)

py_proto_library(
    name = "examples_py_proto",
    srcs = ["examples.proto"],
    default_runtime = "@com_google_protobuf//:protobuf_python",
    protoc = "@com_google_protobuf//:protoc",
    srcs_version = "PY3",
    deps = [
        "@com_google_protobuf//:protobuf_python",
        "@org_tensorflow//tensorflow/core:protos_all_py",
    ],
)
