load("//third_party/bazel_rules/rules_python/python:py_test.bzl", "py_test")
load("//third_party/bazel_rules/rules_python/python:py_library.bzl", "py_library")

# Copybara rewrites load() statements back and forth; do not reformat.
# buildifier: disable=out-of-order-load, disable=same-origin-load

# buildifier: disable=out-of-order-load, disable=same-origin-load

licenses(["notice"])

package(
    default_applicable_licenses = ["//tensorflow_gnn:license"],
    default_visibility = ["//visibility:public"],
)

GOOGLE_INTERNAL_UNIGRAPH_DEPENDENCIES = []

py_library(
    name = "unigraph",
    srcs = ["unigraph.py"],
    type_checking = True,
    deps = [
        "//third_party/py/apache_beam",
        "//third_party/py/pyarrow",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
    ] + GOOGLE_INTERNAL_UNIGRAPH_DEPENDENCIES,
)

py_test(
    name = "unigraph_test",
    srcs = ["unigraph_test.py"],
    data = [
        "@tensorflow_gnn//testdata/heterogeneous",
        "@tensorflow_gnn//testdata/homogeneous",
    ],
    type_checking = True,
    deps = [
        ":unigraph",
        "//:expect_absl_installed_testing",
        "//third_party/py/apache_beam",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
        "//tensorflow_gnn/utils:test_utils",
    ],
)
