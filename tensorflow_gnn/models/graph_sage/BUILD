load("//third_party/bazel_rules/rules_python/python:py_library.bzl", "py_library")

# Copybara rewrites load() statements back and forth; do not reformat.
# buildifier: disable=out-of-order-load, disable=same-origin-load

# buildifier: disable=out-of-order-load, disable=same-origin-load
load("@tensorflow_gnn//tensorflow_gnn:tensorflow_gnn.bzl", "tf_py_test")

licenses(["notice"])

package(
    default_applicable_licenses = ["//tensorflow_gnn:license"],  # copybara:strip
    default_visibility = [
        "//tensorflow_gnn:__pkg__",
        "//tensorflow_gnn/graph:__subpackages__",
    ],
)

package_group(name = "users")

py_library(
    name = "graph_sage",
    srcs = ["__init__.py"],
    type_checking = True,
    visibility = [":users"],
    deps = [
        ":layers",
        "//tensorflow_gnn/utils:api_utils",
    ],
)

py_library(
    name = "layers",
    srcs = ["layers.py"],
    type_checking = True,
    deps = [
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
    ],
)

tf_py_test(
    name = "layers_test",
    srcs = ["layers_test.py"],
    deps = [
        ":layers",
        "//:expect_absl_installed_testing",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
        "//tensorflow_gnn/utils:tf_test_utils",
        "//:expect_ai_edge_litert_installed",
    ],
)
