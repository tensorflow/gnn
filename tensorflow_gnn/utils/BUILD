load("@tensorflow_gnn//tensorflow_gnn:tensorflow_gnn.bzl", "pytype_strict_library")

licenses(["notice"])

package(
    default_applicable_licenses = ["//tensorflow_gnn:license"],
    default_visibility = ["//visibility:public"],
)

pytype_strict_library(
    name = "test_utils",
    srcs = ["test_utils.py"],
    srcs_version = "PY3ONLY",
)

pytype_strict_library(
    name = "api_utils",
    srcs = ["api_utils.py"],
)

pytype_strict_library(
    name = "tf_test_utils",
    srcs = ["tf_test_utils.py"],
    deps = [
        "//:expect_tensorflow_installed",
    ],
)
