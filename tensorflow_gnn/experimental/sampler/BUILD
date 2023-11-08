load("@rules_proto//proto:defs.bzl", "proto_library")
load("@rules_python//python:proto.bzl", "py_proto_library")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

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

py_test(
    name = "core_test",
    srcs = ["core_test.py"],
    python_version = "PY3",
    deps = [],
)

py_test(
    name = "ext_ops_test",
    srcs = ["ext_ops_test.py"],
    python_version = "PY3",
    deps = [],
)

py_test(
    name = "eval_dag_test",
    srcs = ["eval_dag_test.py"],
    python_version = "PY3",
    deps = [],
)