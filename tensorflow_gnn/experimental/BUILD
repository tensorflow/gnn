load("//third_party/bazel_rules/rules_python/python:py_library.bzl", "py_library")

# Copybara rewrites load() statements back and forth; do not reformat.
# buildifier: disable=out-of-order-load, disable=same-origin-load

licenses(["notice"])

package(
    default_applicable_licenses = ["//tensorflow_gnn:license"],
    default_visibility = [
        "//tensorflow_gnn:__subpackages__",
    ],
)

py_library(
    name = "experimental",
    srcs = ["__init__.py"],
    type_checking = True,
    deps = [
        "//tensorflow_gnn/graph:readout",
        "//tensorflow_gnn/graph:tensor_utils",
    ],
)
