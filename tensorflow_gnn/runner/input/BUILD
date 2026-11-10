load("//third_party/bazel_rules/rules_python/python:py_library.bzl", "py_library")
load("@tensorflow_gnn//tensorflow_gnn:tensorflow_gnn.bzl", "tf_py_test")

# Copybara rewrites load() statements back and forth; do not reformat.
# buildifier: disable=out-of-order-load, disable=same-origin-load

# buildifier: disable=out-of-order-load, disable=same-origin-load

licenses(["notice"])

package(
    default_applicable_licenses = ["//tensorflow_gnn:license"],
    default_visibility = ["//visibility:public"],
)

py_library(
    name = "datasets",
    srcs = ["datasets.py"],
    type_checking = True,
    visibility = ["//tensorflow_gnn/runner:__pkg__"],
    deps = [
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn/runner:interfaces",
    ],
)

tf_py_test(
    name = "datasets_test",
    srcs = ["datasets_test.py"],
    deps = [
        ":datasets",
        "//:expect_absl_installed_testing",
        "//third_party/py/google/protobuf:use_fast_cpp_protos",  # Automatically added go/proto_python_upb_flip
        "//:expect_tensorflow_installed",
    ],
)
