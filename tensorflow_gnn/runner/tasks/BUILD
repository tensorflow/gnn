load("@tensorflow_gnn//tensorflow_gnn:tensorflow_gnn.bzl", "pytype_strict_library")
load("@tensorflow_gnn//tensorflow_gnn:tensorflow_gnn.bzl", "py_strict_test")
load("@tensorflow_gnn//tensorflow_gnn:tensorflow_gnn.bzl", "distribute_py_test")

licenses(["notice"])

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
    tags = [
        "tf_2_10_and_up",  # TODO(b/237419537) remove on tf 2.10 release
    ],
    deps = [
        ":classification",
        "//:expect_absl_installed",
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

distribute_py_test(
    name = "dgi_test",
    srcs = ["dgi_test.py"],
    srcs_version = "PY3",
    xla_enable_strict_auto_jit = False,
    deps = [
        ":dgi",
        "//:expect_absl_installed",
        "//:expect_tensorflow_installed",
        "//:expect_tensorflow_installed:tensorflow_no_contrib",
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
    tags = [
        "tf_2_10_and_up",  # TODO(b/237419537) remove on tf 2.10 release
    ],
    deps = [
        ":regression",
        "//:expect_absl_installed",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
        "//tensorflow_gnn/runner:orchestration",
    ],
)
