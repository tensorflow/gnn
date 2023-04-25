load("@com_google_protobuf//:protobuf.bzl", "py_proto_library")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

py_proto_library(
    name = "eval_dag_py_proto",
    srcs = ["eval_dag.proto"],
    default_runtime = "@com_google_protobuf//:protobuf_python",
    protoc = "@com_google_protobuf//:protoc",
    srcs_version = "PY3",
    deps = [
        "@com_google_protobuf//:protobuf_python",
        "@org_tensorflow//tensorflow/core:protos_all_py",
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