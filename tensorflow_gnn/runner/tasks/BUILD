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
    name = "classification",
    srcs = ["classification.py"],
    type_checking = True,
    visibility = ["//tensorflow_gnn/runner:__pkg__"],
    deps = [
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
        "//tensorflow_gnn/runner:interfaces",
    ],
)

tf_py_test(
    name = "classification_test",
    srcs = ["classification_test.py"],
    deps = [
        ":classification",
        "//:expect_absl_installed_testing",
        "//third_party/py/google/protobuf:use_fast_cpp_protos",  # Automatically added go/proto_python_upb_flip
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
        "//tensorflow_gnn/runner:interfaces",
    ],
)

py_library(
    name = "link_prediction",
    srcs = ["link_prediction.py"],
    type_checking = True,
    visibility = ["//tensorflow_gnn/runner:__pkg__"],
    deps = [
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
        "//tensorflow_gnn/runner:interfaces",
    ],
)

tf_py_test(
    name = "link_prediction_test",
    srcs = ["link_prediction_test.py"],
    deps = [
        ":link_prediction",
        "//:expect_absl_installed_testing",
        "//third_party/py/google/protobuf:use_fast_cpp_protos",  # Automatically added go/proto_python_upb_flip
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
    ],
)

py_library(
    name = "regression",
    srcs = ["regression.py"],
    type_checking = True,
    visibility = ["//tensorflow_gnn/runner:__pkg__"],
    deps = [
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
        "//tensorflow_gnn/runner:interfaces",
    ],
)

tf_py_test(
    name = "regression_test",
    srcs = ["regression_test.py"],
    deps = [
        ":regression",
        "//:expect_absl_installed_testing",
        "//third_party/py/google/protobuf:use_fast_cpp_protos",  # Automatically added go/proto_python_upb_flip
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
        "//tensorflow_gnn/runner:interfaces",
    ],
)
