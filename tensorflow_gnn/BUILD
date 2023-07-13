load("@tensorflow_gnn//tensorflow_gnn:tensorflow_gnn.bzl", "pytype_strict_library")
load("@rules_license//rules:license.bzl", "license")

package(
    default_applicable_licenses = [":license"],
    default_visibility = ["//visibility:public"],
)

license(
    name = "license",
    package_name = "tensorflow_gnn",
    license_kinds = [
        "@rules_license//licenses/spdx:Apache-2.0",
    ],
)

licenses(["notice"])

exports_files(["LICENSE"])

config_setting(
    name = "no_tfgnn_py_deps",
    define_values = {"no_tfgnn_py_deps": "true"},
    visibility = ["//visibility:public"],
)

pytype_strict_library(
    name = "tensorflow_gnn",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":version",
        "//tensorflow_gnn/experimental",
        "//tensorflow_gnn/graph:adjacency",
        "//tensorflow_gnn/graph:batching_utils",
        "//tensorflow_gnn/graph:broadcast_ops",
        "//tensorflow_gnn/graph:graph_constants",
        "//tensorflow_gnn/graph:graph_tensor",
        "//tensorflow_gnn/graph:graph_tensor_encode",
        "//tensorflow_gnn/graph:graph_tensor_io",
        "//tensorflow_gnn/graph:graph_tensor_ops",
        "//tensorflow_gnn/graph:graph_tensor_pprint",
        "//tensorflow_gnn/graph:graph_tensor_random",
        "//tensorflow_gnn/graph:normalization_ops",
        "//tensorflow_gnn/graph:padding_ops",
        "//tensorflow_gnn/graph:pool_ops",
        "//tensorflow_gnn/graph:preprocessing_common",
        "//tensorflow_gnn/graph:readout",
        "//tensorflow_gnn/graph:schema_utils",
        "//tensorflow_gnn/graph:schema_validation",
        "//tensorflow_gnn/graph:tag_utils",
        "//tensorflow_gnn/graph:tensor_utils",
        "//tensorflow_gnn/keras",
        "//tensorflow_gnn/proto:graph_schema",
        "//tensorflow_gnn/sampler",
    ],
)

pytype_strict_library(
    name = "version",
    srcs = ["version.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:private"],
)

filegroup(
    name = "files",
    srcs = glob(["**"]),
)
