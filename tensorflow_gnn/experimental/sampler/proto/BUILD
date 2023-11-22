load("@rules_proto//proto:defs.bzl", "proto_library")
load("@rules_python//python:proto.bzl", "py_proto_library")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

py_library(
    name = "proto",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":eval_dag_py_proto",
        "//tensorflow_gnn/utils:api_utils",
    ],
)


proto_library(
    name = "eval_dag_proto",
    srcs = ["eval_dag.proto"],
    deps = [
        "@org_tensorflow//tensorflow/core:protos_all",
    ],
)

py_proto_library(
    name = "eval_dag_py_proto",
    deps = [
        ":eval_dag_proto",
    ],
)
