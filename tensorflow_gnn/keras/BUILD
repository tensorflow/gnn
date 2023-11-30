load("@tensorflow_gnn//tensorflow_gnn:tensorflow_gnn.bzl", "pytype_strict_library")
load("@tensorflow_gnn//tensorflow_gnn:tensorflow_gnn.bzl", "tf_py_test")

licenses(["notice"])

package(
    default_applicable_licenses = ["//tensorflow_gnn:license"],
    default_visibility = [
        ":__subpackages__",
        "//tensorflow_gnn:__pkg__",
        "//tensorflow_gnn/graph:__subpackages__",
    ],
)

pytype_strict_library(
    name = "keras",
    srcs = ["__init__.py"],
    deps = [
        ":builders",
        ":initializers",
        ":keras_tensors",
        "//tensorflow_gnn/keras/layers",
        "//tensorflow_gnn/utils:api_utils",
    ],
)

pytype_strict_library(
    name = "keras_tensors",
    srcs = ["keras_tensors.py"],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn/graph:adjacency",
        "//tensorflow_gnn/graph:graph_constants",
        "//tensorflow_gnn/graph:graph_piece",
        "//tensorflow_gnn/graph:graph_tensor",
        "//tensorflow_gnn/graph:tf_internal",
    ],
)

tf_py_test(
    name = "keras_tensors_test",
    srcs = ["keras_tensors_test.py"],
    python_version = "PY3",
    shard_count = 3,
    deps = [
        ":keras_tensors",
        "//:expect_absl_installed_testing",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn/graph:adjacency",
        "//tensorflow_gnn/graph:broadcast_ops",
        "//tensorflow_gnn/graph:graph_constants",
        "//tensorflow_gnn/graph:graph_tensor",
        "//tensorflow_gnn/graph:graph_tensor_ops",
        "//tensorflow_gnn/graph:pool_ops",
        "//tensorflow_gnn/runner",
        "//tensorflow_gnn/utils:tf_test_utils",
    ],
)

pytype_strict_library(
    name = "builders",
    srcs = ["builders.py"],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn/graph:adjacency",
        "//tensorflow_gnn/graph:graph_constants",
        "//tensorflow_gnn/graph:graph_tensor",
        "//tensorflow_gnn/keras/layers:graph_update",
        "//tensorflow_gnn/keras/layers:next_state",
    ],
)

tf_py_test(
    name = "builders_test",
    srcs = ["builders_test.py"],
    python_version = "PY3",
    deps = [
        ":builders",
        "//:expect_absl_installed_testing",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn/graph:adjacency",
        "//tensorflow_gnn/graph:graph_constants",
        "//tensorflow_gnn/graph:graph_tensor",
        "//tensorflow_gnn/keras/layers:convolutions",
        "//tensorflow_gnn/keras/layers:graph_update",
        "//tensorflow_gnn/keras/layers:next_state",
        "//tensorflow_gnn/utils:tf_test_utils",
    ],
)

pytype_strict_library(
    name = "initializers",
    srcs = ["initializers.py"],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
    ],
)

tf_py_test(
    name = "initializers_test",
    srcs = ["initializers_test.py"],
    python_version = "PY3",
    deps = [
        ":initializers",
        "//:expect_absl_installed_testing",
        "//:expect_tensorflow_installed",
    ],
)
