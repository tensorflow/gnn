load("//third_party/bazel_rules/rules_python/python:py_library.bzl", "py_library")
# Placeholder: load py_binary

# Copybara rewrites load() statements back and forth; do not reformat.
# buildifier: disable=out-of-order-load, disable=same-origin-load

# buildifier: disable=out-of-order-load, disable=same-origin-load

# buildifier: disable=out-of-order-load, disable=same-origin-load
load("@tensorflow_gnn//tensorflow_gnn:tensorflow_gnn.bzl", "tf_py_test")

licenses(["notice"])

package(
    default_applicable_licenses = ["//tensorflow_gnn:license"],
    default_visibility = ["//visibility:public"],
)

py_library(
    name = "train_lib",
    srcs = ["train.py"],
    type_checking = True,
    visibility = [
        ":__subpackages__",
    ],
    deps = [
        ":utils",
        "//:expect_absl_installed_app",
        "//:expect_absl_installed_flags",
        "//third_party/py/absl/logging",
        "//third_party/py/ml_collections/config_dict",
        "//third_party/py/ml_collections/config_flags",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
        "//tensorflow_gnn/models/gat_v2",
        "//tensorflow_gnn/models/hgt",
        "//tensorflow_gnn/models/mt_albis",
        "//tensorflow_gnn/models/multi_head_attention",
        "//tensorflow_gnn/models/vanilla_mpnn",
        "//tensorflow_gnn/runner",
    ],
)

py_binary(
    name = "train",
    srcs = ["train.py"],
    type_checking = True,
    deps = [":train_lib"],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    type_checking = True,
    deps = [
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
    ],
)

tf_py_test(
    name = "utils_test",
    srcs = ["utils_test.py"],
    deps = [
        ":utils",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
    ],
)
