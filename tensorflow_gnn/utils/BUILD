load("//third_party/bazel_rules/rules_python/python:py_library.bzl", "py_library")

# Copybara rewrites load() statements back and forth; do not reformat.
# buildifier: disable=out-of-order-load, disable=same-origin-load

licenses(["notice"])

package(
    default_applicable_licenses = ["//tensorflow_gnn:license"],
    default_visibility = ["//visibility:public"],
)

py_library(
    name = "test_utils",
    srcs = ["test_utils.py"],
    type_checking = True,
)

py_library(
    name = "api_utils",
    srcs = ["api_utils.py"],
    type_checking = True,
)

py_library(
    name = "tf_test_utils",
    srcs = ["tf_test_utils.py"],
    type_checking = True,
    deps = [
        "//:expect_tensorflow_installed",
    ],
)
