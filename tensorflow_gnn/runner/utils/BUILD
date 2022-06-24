load("@tensorflow_gnn//tensorflow_gnn:tensorflow_gnn.bzl", "pytype_strict_library")
load("@tensorflow_gnn//tensorflow_gnn:tensorflow_gnn.bzl", "py_strict_test")

licenses(["notice"])

pytype_strict_library(
    name = "model_dir",
    srcs = ["model_dir.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_gnn/runner:__pkg__"],
    deps = [
        "//:expect_tensorflow_installed",
    ],
)

pytype_strict_library(
    name = "model_export",
    srcs = ["model_export.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_gnn/runner:__pkg__"],
    deps = [
        "//:expect_tensorflow_installed",
    ],
)

pytype_strict_library(
    name = "strategies",
    srcs = ["strategies.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_gnn/runner:__pkg__"],
    deps = [
        "//:expect_tensorflow_installed",
    ],
)

py_strict_test(
    name = "model_export_test",
    srcs = ["model_export_test.py"],
    python_version = "PY3",
    deps = [
        ":model_export",
        "//:expect_absl_installed",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn/runner:orchestration",
    ],
)

pytype_strict_library(
    name = "model_templates",
    srcs = ["model_templates.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_gnn/runner:__pkg__"],
    deps = [
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
    ],
)
