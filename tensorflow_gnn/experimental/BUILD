load("@tensorflow_gnn//tensorflow_gnn:tensorflow_gnn.bzl", "pytype_strict_library")

licenses(["notice"])

package(
    default_applicable_licenses = ["//tensorflow_gnn:license"],
    default_visibility = [
        "//tensorflow_gnn:__subpackages__",
    ],
)

pytype_strict_library(
    name = "experimental",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_gnn/graph:readout",
        "//tensorflow_gnn/graph:tensor_utils",
    ],
)
