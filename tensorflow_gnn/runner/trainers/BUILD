load("@tensorflow_gnn//tensorflow_gnn:tensorflow_gnn.bzl", "pytype_strict_library")
load("@tensorflow_gnn//tensorflow_gnn:tensorflow_gnn.bzl", "py_strict_test")

licenses(["notice"])

pytype_strict_library(
    name = "keras_fit",
    srcs = ["keras_fit.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_gnn/runner:__pkg__"],
    deps = [
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn/runner:orchestration",
    ],
)

py_strict_test(
    name = "keras_fit_test",
    srcs = ["keras_fit_test.py"],
    python_version = "PY3",
    deps = [
        ":keras_fit",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn/runner:orchestration",
    ],
)
