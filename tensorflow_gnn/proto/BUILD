package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

load("@rules_proto//proto:defs.bzl", "proto_library")
load("@rules_python//python:proto.bzl", "py_proto_library")

py_library(
    name = "proto",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":graph_schema",
        "//tensorflow_gnn/utils:api_utils",
    ],
)

proto_library(
    name = "graph_schema_proto",
    srcs = ["graph_schema.proto"],
    deps = [
        "@org_tensorflow//tensorflow/core:protos_all",
    ],
)

py_proto_library(
    name = "graph_schema_py_proto",
    deps = [
        ":graph_schema_proto",
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

proto_library(
    name = "examples_proto",
    srcs = ["examples.proto"],
    deps = ["@org_tensorflow//tensorflow/core:protos_all"],
)

py_proto_library(
    name = "examples_py_proto",
    deps = [
        ":examples_proto",
    ],
)
