load("@tensorflow_gnn//tensorflow_gnn:tensorflow_gnn.bzl", "pytype_strict_library")
load("@tensorflow_gnn//tensorflow_gnn:tensorflow_gnn.bzl", "py_strict_test")

licenses(["notice"])

pytype_strict_library(
    name = "attribution",
    srcs = ["attribution.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_gnn/runner:__pkg__"],
    deps = [
        ":model",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
        "//tensorflow_gnn/runner:interfaces",
    ],
)

pytype_strict_library(
    name = "model",
    srcs = ["model.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_gnn/runner:__pkg__"],
    deps = [
        "//:expect_tensorflow_installed",
    ],
)

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
        ":model",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn/runner:interfaces",
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

pytype_strict_library(
    name = "padding",
    srcs = ["padding.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_gnn/runner:__pkg__"],
    deps = [
        ":parsing",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
        "//tensorflow_gnn/runner:interfaces",
    ],
)

pytype_strict_library(
    name = "parsing",
    srcs = ["parsing.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_gnn/runner:__pkg__"],
    deps = [
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
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
    name = "attribution_test",
    srcs = ["attribution_test.py"],
    srcs_version = "PY3",
    tags = ["tf_at_least_2_10"],
    deps = [
        ":attribution",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
    ],
)

py_strict_test(
    name = "model_test",
    srcs = ["model_test.py"],
    python_version = "PY3",
    deps = [
        ":model",
        "//:expect_absl_installed",
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
    ],
)

py_strict_test(
    name = "parsing_test",
    srcs = ["parsing_test.py"],
    srcs_version = "PY3",
    deps = [
        ":parsing",
        "//:expect_absl_installed",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
    ],
)
