load("@tensorflow_gnn//tensorflow_gnn:tensorflow_gnn.bzl", "pytype_strict_library")
load("@tensorflow_gnn//tensorflow_gnn:tensorflow_gnn.bzl", "py_strict_test")

licenses(["notice"])

pytype_strict_library(
    name = "attribution",
    srcs = ["attribution.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_gnn/runner:__pkg__"],
    deps = [
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
    ],
)

py_strict_test(
    name = "attribution_test",
    srcs = ["attribution_test.py"],
    srcs_version = "PY3",
    deps = [
        ":attribution",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
        "//tensorflow_gnn/runner:orchestration",
    ],
)

pytype_strict_library(
    name = "classification",
    srcs = ["classification.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_gnn/runner:__pkg__"],
    deps = [
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
    ],
)

py_strict_test(
    name = "classification_test",
    srcs = ["classification_test.py"],
    srcs_version = "PY3",
    deps = [
        ":classification",
        "//:expect_absl_installed",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
        "//tensorflow_gnn/runner:orchestration",
    ],
)

pytype_strict_library(
    name = "dgi",
    srcs = ["dgi.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_gnn/runner:__pkg__"],
    deps = [
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
    ],
)

py_strict_test(
    name = "dgi_test",
    srcs = ["dgi_test.py"],
    srcs_version = "PY3",
    deps = [
        ":dgi",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
        "//tensorflow_gnn/runner:orchestration",
    ],
)

pytype_strict_library(
    name = "regression",
    srcs = ["regression.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_gnn/runner:__pkg__"],
    deps = [
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
    ],
)

py_strict_test(
    name = "regression_test",
    srcs = ["regression_test.py"],
    srcs_version = "PY3",
    deps = [
        ":regression",
        "//:expect_absl_installed",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
        "//tensorflow_gnn/runner:orchestration",
    ],
)