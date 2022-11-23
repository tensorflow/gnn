load("@tensorflow_gnn//tensorflow_gnn:tensorflow_gnn.bzl", "pytype_strict_contrib_test", "pytype_strict_library")
load("@tensorflow_gnn//tensorflow_gnn:tensorflow_gnn.bzl", "tf_py_test")

licenses(["notice"])

package(default_visibility = [
    "//tensorflow_gnn:__pkg__",
    "//tensorflow_gnn/compat:__subpackages__",
    "//tensorflow_gnn/converters:__subpackages__",
    "//tensorflow_gnn/examples:__subpackages__",
    "//tensorflow_gnn/experimental:__subpackages__",
    "//tensorflow_gnn/graph:__subpackages__",
    "//tensorflow_gnn/keras:__subpackages__",
])

pytype_strict_library(
    name = "graph_constants",
    srcs = ["graph_constants.py"],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
    ],
)

pytype_strict_library(
    name = "graph_tensor",
    srcs = ["graph_tensor.py"],
    srcs_version = "PY3",
    deps = [
        ":adjacency",
        ":graph_constants",
        ":graph_piece",
        ":tensor_utils",
        "//:expect_tensorflow_installed",
    ],
)

pytype_strict_library(
    name = "graph_tensor_ops",
    srcs = ["graph_tensor_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":adjacency",
        ":graph_constants",
        ":graph_tensor",
        ":tensor_utils",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn/keras:keras_tensors",
    ],
)

pytype_strict_library(
    name = "graph_tensor_io",
    srcs = ["graph_tensor_io.py"],
    srcs_version = "PY3",
    deps = [
        ":adjacency",
        ":graph_constants",
        ":graph_piece",
        ":graph_tensor",
        "//:expect_tensorflow_installed",
    ],
)

pytype_strict_library(
    name = "graph_tensor_encode",
    srcs = ["graph_tensor_encode.py"],
    srcs_version = "PY3",
    deps = [
        ":adjacency",
        ":graph_constants",
        ":graph_piece",
        ":graph_tensor",
        "//:expect_tensorflow_installed",
    ],
)

pytype_strict_library(
    name = "graph_tensor_random",
    srcs = ["graph_tensor_random.py"],
    srcs_version = "PY3",
    deps = [
        ":adjacency",
        ":graph_constants",
        ":graph_tensor",
        ":schema_utils",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn/proto:graph_schema_py_proto",
    ],
)

pytype_strict_library(
    name = "graph_tensor_pprint",
    srcs = ["graph_tensor_pprint.py"],
    srcs_version = "PY3",
    deps = [
        ":graph_tensor",
        "//:expect_numpy_installed",
    ],
)

pytype_strict_library(
    name = "tensor_utils",
    srcs = ["tensor_utils.py"],
    srcs_version = "PY3",
    deps = ["//:expect_tensorflow_installed"],
)

tf_py_test(
    name = "tensor_utils_test",
    srcs = ["tensor_utils_test.py"],
    python_version = "PY3",
    deps = [
        ":tensor_utils",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
    ],
)

pytype_strict_library(
    name = "schema_utils",
    srcs = ["schema_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":adjacency",
        ":graph_constants",
        ":graph_tensor",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn/proto:graph_schema_py_proto",
    ],
)

tf_py_test(
    name = "schema_utils_test",
    srcs = ["schema_utils_test.py"],
    data = ["@tensorflow_gnn//testdata/homogeneous"],
    python_version = "PY3",
    deps = [
        ":adjacency",
        ":graph_tensor",
        ":schema_utils",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn/proto:graph_schema_py_proto",
        "//tensorflow_gnn/utils:test_utils",
    ],
)

pytype_strict_library(
    name = "schema_validation",
    srcs = ["schema_validation.py"],
    srcs_version = "PY3",
    deps = [
        ":adjacency",
        ":graph_constants",
        ":graph_tensor",
        ":schema_utils",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn/proto:graph_schema_py_proto",
    ],
)

tf_py_test(
    name = "schema_validation_test",
    srcs = ["schema_validation_test.py"],
    python_version = "PY3",
    deps = [
        ":adjacency",
        ":graph_tensor",
        ":schema_validation",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn/proto:graph_schema_py_proto",
    ],
)

pytype_strict_library(
    name = "graph_tensor_test_utils",
    srcs = ["graph_tensor_test_utils.py"],
    srcs_version = "PY3",
    visibility = [
        "//tensorflow_gnn/compat:__subpackages__",
        "//tensorflow_gnn/tools:__subpackages__",
    ],
    deps = [
        ":graph_constants",
        ":graph_tensor_encode",
        ":graph_tensor_random",
        ":schema_utils",
        "//:expect_absl_installed",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn/proto:graph_schema_py_proto",
    ],
)

tf_py_test(
    name = "graph_tensor_test",
    srcs = ["graph_tensor_test.py"],
    python_version = "PY3",
    deps = [
        ":adjacency",
        ":graph_constants",
        ":graph_tensor",
        ":graph_tensor_test_utils",
        "//:expect_tensorflow_installed",
    ],
)

tf_py_test(
    name = "graph_tensor_ops_test",
    srcs = ["graph_tensor_ops_test.py"],
    python_version = "PY3",
    deps = [
        ":adjacency",
        ":graph_constants",
        ":graph_tensor",
        ":graph_tensor_ops",
        "//:expect_tensorflow_installed",
    ],
)

tf_py_test(
    name = "graph_tensor_io_test",
    srcs = ["graph_tensor_io_test.py"],
    python_version = "PY3",
    deps = [
        ":adjacency",
        ":graph_constants",
        ":graph_tensor",
        ":graph_tensor_io",
        ":schema_utils",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn/proto:graph_schema_py_proto",
    ],
)

tf_py_test(
    name = "graph_tensor_encode_test",
    srcs = ["graph_tensor_encode_test.py"],
    data = ["@tensorflow_gnn//testdata:feature_repr"],
    python_version = "PY3",
    deps = [
        ":graph_constants",
        ":graph_tensor",
        ":graph_tensor_encode",
        ":graph_tensor_io",
        ":graph_tensor_random",
        ":schema_utils",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn/proto:graph_schema_py_proto",
        "//tensorflow_gnn/utils:test_utils",
    ],
)

tf_py_test(
    name = "graph_tensor_random_test",
    srcs = ["graph_tensor_random_test.py"],
    data = ["@tensorflow_gnn//testdata:feature_repr"],
    python_version = "PY3",
    deps = [
        ":graph_tensor",
        ":graph_tensor_random",
        ":schema_utils",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn/proto:graph_schema_py_proto",
        "//tensorflow_gnn/utils:test_utils",
    ],
)

tf_py_test(
    name = "graph_tensor_pprint_test",
    srcs = ["graph_tensor_pprint_test.py"],
    data = ["@tensorflow_gnn//testdata:feature_repr"],
    python_version = "PY3",
    deps = [
        ":graph_tensor_pprint",
        ":graph_tensor_random",
        ":schema_utils",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn/proto:graph_schema_py_proto",
        "//tensorflow_gnn/utils:test_utils",
    ],
)

pytype_strict_library(
    name = "graph_piece",
    srcs = ["graph_piece.py"],
    srcs_version = "PY3",
    deps = [
        ":graph_constants",
        ":tensor_utils",
        "//:expect_tensorflow_installed",
    ],
)

tf_py_test(
    name = "graph_piece_test",
    srcs = ["graph_piece_test.py"],
    python_version = "PY3",
    deps = [
        ":graph_piece",
        "//:expect_tensorflow_installed",
    ],
)

pytype_strict_library(
    name = "adjacency",
    srcs = ["adjacency.py"],
    srcs_version = "PY3",
    deps = [
        ":graph_constants",
        ":graph_piece",
        ":tensor_utils",
        "//:expect_tensorflow_installed",
    ],
)

tf_py_test(
    name = "adjacency_test",
    srcs = ["adjacency_test.py"],
    python_version = "PY3",
    deps = [
        ":adjacency",
        ":graph_constants",
        "//:expect_tensorflow_installed",
    ],
)

pytype_strict_library(
    name = "normalization_ops",
    srcs = ["normalization_ops.py"],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn/graph:graph_constants",
        "//tensorflow_gnn/graph:graph_tensor",
        "//tensorflow_gnn/graph:graph_tensor_ops",
    ],
)

tf_py_test(
    name = "normalization_ops_test",
    srcs = ["normalization_ops_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":adjacency",
        ":graph_constants",
        ":graph_tensor",
        ":graph_tensor_ops",
        ":normalization_ops",
        "//:expect_absl_installed",
        "//:expect_tensorflow_installed",
    ],
)

pytype_strict_library(
    name = "dict_utils",
    srcs = ["dict_utils.py"],
    srcs_version = "PY3",
    deps = [],
)

pytype_strict_contrib_test(
    name = "dict_utils_test",
    srcs = ["dict_utils_test.py"],
    srcs_version = "PY3",
    deps = [":dict_utils"],
)

pytype_strict_library(
    name = "preprocessing_common",
    srcs = ["preprocessing_common.py"],
    srcs_version = "PY3",
    deps = [
        ":graph_constants",
        "//:expect_tensorflow_installed",
    ],
)

tf_py_test(
    name = "preprocessing_common_test",
    srcs = ["preprocessing_common_test.py"],
    python_version = "PY3",
    deps = [
        ":preprocessing_common",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn/graph:graph_tensor",
    ],
)

pytype_strict_library(
    name = "padding_ops",
    srcs = ["padding_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":adjacency",
        ":graph_constants",
        ":graph_piece",
        ":graph_tensor",
        ":preprocessing_common",
        ":tensor_utils",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
    ],
)

tf_py_test(
    name = "padding_ops_test",
    srcs = ["padding_ops_test.py"],
    python_version = "PY3",
    deps = [
        ":adjacency",
        ":graph_constants",
        ":graph_piece",
        ":graph_tensor",
        ":graph_tensor_test_utils",
        ":padding_ops",
        ":preprocessing_common",
        "//:expect_tensorflow_installed",
    ],
)

pytype_strict_library(
    name = "batching_utils",
    srcs = ["batching_utils.py"],
    srcs_version = "PY3",
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

pytype_strict_library(
    name = "tag_utils",
    srcs = ["tag_utils.py"],
    deps = [":graph_constants"],
)

pytype_strict_contrib_test(
    name = "tag_utils_test",
    srcs = ["tag_utils_test.py"],
    deps = [
        ":graph_constants",
        ":tag_utils",
        "//:expect_absl_installed",
    ],
)

tf_py_test(
    name = "batching_utils_test",
    size = "large",
    srcs = ["batching_utils_test.py"],
    python_version = "PY3",
    deps = [
        ":adjacency",
        ":batching_utils",
        ":graph_constants",
        ":graph_tensor",
        ":graph_tensor_test_utils",
        ":padding_ops",
        ":preprocessing_common",
        "//:expect_tensorflow_installed",
    ],
)
