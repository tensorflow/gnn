load("//third_party/bazel_rules/rules_python/python:py_test.bzl", "py_test")
load("//third_party/bazel_rules/rules_python/python:py_library.bzl", "py_library")

# Copybara rewrites load() statements back and forth; do not reformat.
# buildifier: disable=out-of-order-load, disable=same-origin-load

# buildifier: disable=out-of-order-load, disable=same-origin-load

# buildifier: disable=out-of-order-load, disable=same-origin-load

# buildifier: disable=out-of-order-load, disable=same-origin-load
load("@tensorflow_gnn//tensorflow_gnn:tensorflow_gnn.bzl", "tf_py_test")

licenses(["notice"])

package(
    default_applicable_licenses = ["//tensorflow_gnn:license"],
    default_visibility = [
        "//tensorflow_gnn:__pkg__",
        "//tensorflow_gnn/compat:__subpackages__",
        "//tensorflow_gnn/converters:__subpackages__",
        "//tensorflow_gnn/examples:__subpackages__",
        "//tensorflow_gnn/experimental:__subpackages__",
        "//tensorflow_gnn/graph:__subpackages__",
        "//tensorflow_gnn/keras:__subpackages__",
        "//tensorflow_gnn/sampler:__subpackages__",
    ],
)

py_library(
    name = "graph_constants",
    srcs = ["graph_constants.py"],
    type_checking = True,
    deps = [
        "//:expect_tensorflow_installed",
    ],
)

py_library(
    name = "graph_tensor",
    srcs = ["graph_tensor.py"],
    type_checking = True,
    deps = [
        ":adjacency",
        ":graph_constants",
        ":graph_piece",
        ":tensor_utils",
        ":tf_internal",
        "//:expect_tensorflow_installed",
    ],
)

py_library(
    name = "graph_tensor_ops",
    srcs = ["graph_tensor_ops.py"],
    type_checking = True,
    deps = [
        ":adjacency",
        ":broadcast_ops",
        ":graph_constants",
        ":graph_tensor",
        ":pool_ops",
        ":tag_utils",
        ":tensor_utils",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn/keras:keras_tensors",
    ],
)

py_library(
    name = "graph_tensor_io",
    srcs = ["graph_tensor_io.py"],
    type_checking = True,
    deps = [
        ":adjacency",
        ":graph_constants",
        ":graph_piece",
        ":graph_tensor",
        "//:expect_tensorflow_installed",
    ],
)

py_library(
    name = "graph_tensor_encode",
    srcs = ["graph_tensor_encode.py"],
    type_checking = True,
    deps = [
        ":adjacency",
        ":graph_constants",
        ":graph_piece",
        ":graph_tensor",
        ":graph_tensor_io",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn/keras:keras_tensors",
    ],
)

py_library(
    name = "graph_tensor_random",
    srcs = ["graph_tensor_random.py"],
    type_checking = True,
    deps = [
        ":adjacency",
        ":graph_constants",
        ":graph_tensor",
        ":tensor_utils",
        "//:expect_tensorflow_installed",
    ],
)

py_library(
    name = "graph_tensor_pprint",
    srcs = ["graph_tensor_pprint.py"],
    type_checking = True,
    deps = [
        ":graph_tensor",
        "//:expect_numpy_installed",
        "//tensorflow_gnn/keras:keras_tensors",
    ],
)

py_library(
    name = "tensor_utils",
    srcs = ["tensor_utils.py"],
    type_checking = True,
    deps = [
        ":tf_internal",
        "//:expect_tensorflow_installed",
    ],
)

tf_py_test(
    name = "tensor_utils_test",
    srcs = ["tensor_utils_test.py"],
    deps = [
        ":tensor_utils",
        "//:expect_absl_installed_testing",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
    ],
)

py_library(
    name = "readout",
    srcs = ["readout.py"],
    type_checking = True,
    deps = [
        ":adjacency",
        ":broadcast_ops",
        ":graph_constants",
        ":graph_tensor",
        ":pool_ops",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn/keras:keras_tensors",
    ],
)

tf_py_test(
    name = "readout_test",
    srcs = ["readout_test.py"],
    deps = [
        ":adjacency",
        ":batching_utils",
        ":graph_constants",
        ":graph_tensor",
        ":graph_tensor_ops",
        ":readout",
        "//:expect_absl_installed_testing",
        "//:expect_tensorflow_installed",
    ],
)

py_library(
    name = "schema_utils",
    srcs = ["schema_utils.py"],
    type_checking = True,
    deps = [
        ":adjacency",
        ":graph_constants",
        ":graph_tensor",
        ":graph_tensor_io",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn/keras:keras_tensors",
        "//tensorflow_gnn/proto:graph_schema_py_proto",
    ],
)

tf_py_test(
    name = "schema_utils_test",
    srcs = ["schema_utils_test.py"],
    data = ["@tensorflow_gnn//testdata/homogeneous"],
    deps = [
        ":adjacency",
        ":graph_tensor",
        ":schema_utils",
        "//:expect_absl_installed_testing",
        "//third_party/py/google/protobuf",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn/proto:graph_schema_py_proto",
        "//tensorflow_gnn/utils:test_utils",
    ],
)

py_library(
    name = "schema_validation",
    srcs = ["schema_validation.py"],
    type_checking = True,
    deps = [
        ":adjacency",
        ":graph_constants",
        ":graph_tensor",
        ":readout",
        ":schema_utils",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn/proto:graph_schema_py_proto",
    ],
)

tf_py_test(
    name = "schema_validation_test",
    srcs = ["schema_validation_test.py"],
    deps = [
        ":adjacency",
        ":graph_constants",
        ":graph_tensor",
        ":schema_validation",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn/proto:graph_schema_py_proto",
    ],
)

py_library(
    name = "graph_tensor_test_utils",
    srcs = ["graph_tensor_test_utils.py"],
    type_checking = True,
    visibility = [
        "//tensorflow_gnn/compat:__subpackages__",
        "//tensorflow_gnn/tools:__subpackages__",
    ],
    deps = [
        ":graph_constants",
        ":graph_tensor_encode",
        ":graph_tensor_random",
        ":schema_utils",
        "//:expect_absl_installed_testing",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn/proto:graph_schema_py_proto",
    ],
)

tf_py_test(
    name = "graph_tensor_test",
    srcs = ["graph_tensor_test.py"],
    deps = [
        ":adjacency",
        ":graph_constants",
        ":graph_tensor",
        ":graph_tensor_test_utils",
        "//:expect_absl_installed_testing",
        "//:expect_tensorflow_installed",
        "//:expect_ai_edge_litert_installed",
    ],
)

tf_py_test(
    name = "graph_tensor_ops_test",
    srcs = ["graph_tensor_ops_test.py"],
    deps = [
        ":adjacency",
        ":broadcast_ops",
        ":graph_constants",
        ":graph_tensor",
        ":graph_tensor_ops",
        ":pool_ops",
        ":readout",
        "//:expect_absl_installed_testing",
        "//:expect_tensorflow_installed",
        "//:expect_ai_edge_litert_installed",
    ],
)

tf_py_test(
    name = "graph_tensor_io_test",
    srcs = ["graph_tensor_io_test.py"],
    deps = [
        ":adjacency",
        ":graph_constants",
        ":graph_tensor",
        ":graph_tensor_io",
        ":schema_utils",
        "//:expect_absl_installed_testing",
        "//third_party/py/google/protobuf",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn/proto:graph_schema_py_proto",
    ],
)

tf_py_test(
    name = "graph_tensor_encode_test",
    srcs = ["graph_tensor_encode_test.py"],
    data = ["@tensorflow_gnn//testdata:feature_repr"],
    deps = [
        ":graph_constants",
        ":graph_tensor",
        ":graph_tensor_encode",
        ":graph_tensor_io",
        ":graph_tensor_random",
        ":schema_utils",
        "//:expect_absl_installed_testing",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn/proto:graph_schema_py_proto",
        "//tensorflow_gnn/utils:test_utils",
    ],
)

tf_py_test(
    name = "graph_tensor_random_test",
    srcs = ["graph_tensor_random_test.py"],
    data = ["@tensorflow_gnn//testdata:feature_repr"],
    deps = [
        ":adjacency",
        ":graph_constants",
        ":graph_tensor",
        ":graph_tensor_random",
        ":schema_utils",
        "//:expect_absl_installed_testing",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn/proto:graph_schema_py_proto",
        "//tensorflow_gnn/utils:test_utils",
    ],
)

tf_py_test(
    name = "graph_tensor_pprint_test",
    srcs = ["graph_tensor_pprint_test.py"],
    data = ["@tensorflow_gnn//testdata:feature_repr"],
    deps = [
        ":graph_tensor_pprint",
        ":graph_tensor_random",
        ":schema_utils",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn/proto:graph_schema_py_proto",
        "//tensorflow_gnn/utils:test_utils",
    ],
)

py_library(
    name = "graph_piece",
    srcs = ["graph_piece.py"],
    type_checking = True,
    deps = [
        ":graph_constants",
        ":tensor_utils",
        ":tf_internal",
        "//:expect_tensorflow_installed",
    ],
)

tf_py_test(
    name = "graph_piece_test",
    srcs = ["graph_piece_test.py"],
    deps = [
        ":graph_piece",
        ":tf_internal",
        "//:expect_absl_installed_testing",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn/utils:tf_test_utils",
    ],
)

py_library(
    name = "adjacency",
    srcs = ["adjacency.py"],
    type_checking = True,
    deps = [
        ":graph_constants",
        ":graph_piece",
        ":tensor_utils",
        ":tf_internal",
        "//:expect_tensorflow_installed",
    ],
)

tf_py_test(
    name = "adjacency_test",
    srcs = ["adjacency_test.py"],
    deps = [
        ":adjacency",
        ":graph_constants",
        "//:expect_absl_installed_testing",
        "//:expect_tensorflow_installed",
    ],
)

py_library(
    name = "normalization_ops",
    srcs = ["normalization_ops.py"],
    type_checking = True,
    deps = [
        ":broadcast_ops",
        ":graph_constants",
        ":graph_tensor",
        ":pool_ops",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn/keras:keras_tensors",
    ],
)

tf_py_test(
    name = "normalization_ops_test",
    srcs = ["normalization_ops_test.py"],
    deps = [
        ":adjacency",
        ":broadcast_ops",
        ":graph_constants",
        ":graph_tensor",
        ":normalization_ops",
        "//:expect_absl_installed_testing",
        "//:expect_tensorflow_installed",
    ],
)

py_library(
    name = "dict_utils",
    srcs = ["dict_utils.py"],
    type_checking = True,
    deps = [],
)

py_test(
    name = "dict_utils_test",
    srcs = ["dict_utils_test.py"],
    type_checking = True,
    deps = [":dict_utils"],
)

py_library(
    name = "preprocessing_common",
    srcs = ["preprocessing_common.py"],
    type_checking = True,
    deps = [
        ":graph_constants",
        "//:expect_tensorflow_installed",
    ],
)

tf_py_test(
    name = "preprocessing_common_test",
    srcs = ["preprocessing_common_test.py"],
    deps = [
        ":graph_constants",
        ":graph_tensor",
        ":preprocessing_common",
        "//:expect_absl_installed_testing",
        "//:expect_tensorflow_installed",
    ],
)

py_library(
    name = "padding_ops",
    srcs = ["padding_ops.py"],
    type_checking = True,
    deps = [
        ":adjacency",
        ":graph_constants",
        ":graph_piece",
        ":graph_tensor",
        ":preprocessing_common",
        ":tensor_utils",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn/keras:keras_tensors",
    ],
)

tf_py_test(
    name = "padding_ops_test",
    srcs = ["padding_ops_test.py"],
    deps = [
        ":adjacency",
        ":graph_constants",
        ":graph_tensor",
        ":graph_tensor_test_utils",
        ":padding_ops",
        ":preprocessing_common",
        "//:expect_absl_installed_testing",
        "//:expect_tensorflow_installed",
    ],
)

py_library(
    name = "pool_ops",
    srcs = ["pool_ops.py"],
    type_checking = True,
    deps = [
        ":adjacency",
        ":graph_constants",
        ":graph_tensor",
        ":tag_utils",
        ":tensor_utils",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn/keras:keras_tensors",
    ],
)

tf_py_test(
    name = "pool_ops_test",
    srcs = ["pool_ops_test.py"],
    deps = [
        ":adjacency",
        ":graph_constants",
        ":graph_tensor",
        ":pool_ops",
        "//:expect_absl_installed_testing",
        "//:expect_tensorflow_installed",
    ],
)

py_library(
    name = "batching_utils",
    srcs = ["batching_utils.py"],
    type_checking = True,
    deps = [
        ":adjacency",
        ":graph_constants",
        ":graph_piece",
        ":graph_tensor",
        ":padding_ops",
        ":preprocessing_common",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
    ],
)

py_library(
    name = "tag_utils",
    srcs = ["tag_utils.py"],
    type_checking = True,
    deps = [
        ":graph_constants",
        ":graph_tensor",
    ],
)

py_library(
    name = "broadcast_ops",
    srcs = ["broadcast_ops.py"],
    type_checking = True,
    deps = [
        ":graph_constants",
        ":graph_tensor",
        ":tag_utils",
        ":tensor_utils",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn/keras:keras_tensors",
    ],
)

py_library(
    name = "tf_internal",
    srcs = ["tf_internal.py"],
    # strict_deps = False,
    tags = [
        # Certain deps are not provided on purpose, so this cannot be a python_strict_library.
        "ignore_for_dep=third_party.tensorflow.python.framework.type_spec_registry",
    ],
    type_checking = True,
    deps = [
        "//:expect_tensorflow_installed",
        "//:expect_tf_keras_installed",
    ],
)

py_test(
    name = "tag_utils_test",
    srcs = ["tag_utils_test.py"],
    type_checking = True,
    deps = [
        ":adjacency",
        ":graph_constants",
        ":graph_tensor",
        ":tag_utils",
        "//:expect_absl_installed_testing",
        "//:expect_tensorflow_installed",
    ],
)

tf_py_test(
    name = "batching_utils_test",
    size = "large",
    srcs = ["batching_utils_test.py"],
    deps = [
        ":adjacency",
        ":batching_utils",
        ":graph_constants",
        ":graph_tensor",
        ":graph_tensor_test_utils",
        ":preprocessing_common",
        "//:expect_absl_installed_testing",
        "//:expect_tensorflow_installed",
    ],
)

tf_py_test(
    name = "broadcast_ops_test",
    srcs = ["broadcast_ops_test.py"],
    deps = [
        ":adjacency",
        ":broadcast_ops",
        ":graph_constants",
        ":graph_tensor",
        "//:expect_absl_installed_testing",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
    ],
)
