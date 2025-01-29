# Copybara rewrites load() statements back and forth; do not reformat.
# buildifier: disable=out-of-order-load, disable=same-origin-load
load("@tensorflow_gnn//tensorflow_gnn:tensorflow_gnn.bzl", "pytype_strict_library")

# buildifier: disable=out-of-order-load, disable=same-origin-load
load("@tensorflow_gnn//tensorflow_gnn:tensorflow_gnn.bzl", "tf_py_test")

licenses(["notice"])

package(default_visibility = [
    "//tensorflow_gnn:__pkg__",
    "//tensorflow_gnn/graph:__subpackages__",
])

package_group(name = "users")

pytype_strict_library(
    name = "graph_sage",
    srcs = ["__init__.py"],
    visibility = [":users"],
    deps = [
        ":layers",
        "//tensorflow_gnn/utils:api_utils",
    ],
)

pytype_strict_library(
    name = "layers",
    srcs = ["layers.py"],
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
