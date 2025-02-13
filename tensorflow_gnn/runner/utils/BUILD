# Copybara rewrites load() statements back and forth; do not reformat.
# buildifier: disable=out-of-order-load, disable=same-origin-load
load("@tensorflow_gnn//tensorflow_gnn:tensorflow_gnn.bzl", "pytype_strict_binary")

# buildifier: disable=out-of-order-load, disable=same-origin-load
load("@tensorflow_gnn//tensorflow_gnn:tensorflow_gnn.bzl", "pytype_strict_library")

# buildifier: disable=out-of-order-load, disable=same-origin-load
load("@tensorflow_gnn//tensorflow_gnn:tensorflow_gnn.bzl", "py_strict_test")

licenses(["notice"])

package(
    default_applicable_licenses = ["//tensorflow_gnn:license"],
    default_visibility = ["//visibility:public"],
)

pytype_strict_library(
    name = "attribution",
    srcs = ["attribution.py"],
    visibility = ["//tensorflow_gnn/runner:__pkg__"],
    deps = [
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
        "//tensorflow_gnn/runner:interfaces",
    ],
)

pytype_strict_library(
    name = "label_fns",
    srcs = ["label_fns.py"],
    visibility = ["//tensorflow_gnn/runner:__pkg__"],
    deps = [
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
    ],
)

pytype_strict_library(
    name = "model_dir",
    srcs = ["model_dir.py"],
    visibility = ["//tensorflow_gnn/runner:__pkg__"],
    deps = [
        "//:expect_tensorflow_installed",
    ],
)

pytype_strict_library(
    name = "model_export",
    srcs = ["model_export.py"],
    visibility = ["//tensorflow_gnn/runner:__pkg__"],
    deps = [
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn/runner:interfaces",
    ],
)

pytype_strict_library(
    name = "padding",
    srcs = ["padding.py"],
    visibility = ["//tensorflow_gnn/runner:__pkg__"],
    deps = [
        ":parsing",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
        "//tensorflow_gnn/runner:interfaces",
    ],
)

pytype_strict_library(
    name = "parsing",
    srcs = ["parsing.py"],
    visibility = ["//tensorflow_gnn/runner:__pkg__"],
    deps = [
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
    ],
)

pytype_strict_library(
    name = "strategies",
    srcs = ["strategies.py"],
    visibility = ["//tensorflow_gnn/runner:__pkg__"],
    deps = [
        "//:expect_tensorflow_installed",
    ],
)

pytype_strict_library(
    name = "test_utils",
    srcs = ["test_utils.py"],
    deps = [
        "//tensorflow_gnn/runner:interfaces",
    ],
)

py_strict_test(
    name = "attribution_test",
    srcs = ["attribution_test.py"],
    deps = [
        ":attribution",
        ":test_utils",
        "//third_party/py/google/protobuf:use_fast_cpp_protos",  # Automatically added go/proto_python_upb_flip
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
    ],
)

py_strict_test(
    name = "model_export_test",
    srcs = ["model_export_test.py"],
    deps = [
        ":model_export",
        ":test_utils",
        "//:expect_absl_installed_testing",
        "//third_party/py/google/protobuf:use_fast_cpp_protos",  # Automatically added go/proto_python_upb_flip
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
    ],
)

py_strict_test(
    name = "parsing_test",
    srcs = ["parsing_test.py"],
    deps = [
        ":parsing",
        "//:expect_absl_installed_testing",
        "//third_party/py/google/protobuf:use_fast_cpp_protos",  # Automatically added go/proto_python_upb_flip
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
    ],
)

pytype_strict_binary(
    name = "saved_model_gen_testdata",
    srcs = ["saved_model_gen_testdata.py"],
    deps = [
        "//:expect_absl_installed_app",
        "//:expect_absl_installed_flags",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
        "//tensorflow_gnn/runner",
    ],
)

pytype_strict_binary(
    name = "saved_model_load_testdata",
    srcs = ["saved_model_load_testdata.py"],
    deps = [
        "//:expect_absl_installed_app",
        "//:expect_absl_installed_flags",
        "//:expect_tensorflow_installed",
    ],
)

sh_test(
    name = "saved_model_test",
    size = "small",
    srcs = ["saved_model_test.sh"],
    args = [
        "$(location :saved_model_gen_testdata)",
        "$(location :saved_model_load_testdata)",
        "0",  # use_legacy_model_save=False
    ],
    data = [
        ":saved_model_gen_testdata",
        ":saved_model_load_testdata",
    ],
    tags = ["tf_at_least_2_13"],
)

sh_test(
    name = "legacy_saved_model_test",
    size = "small",
    srcs = ["saved_model_test.sh"],
    args = [
        "$(location :saved_model_gen_testdata)",
        "$(location :saved_model_load_testdata)",
        "1",  # use_legacy_model_save=True
    ],
    data = [
        ":saved_model_gen_testdata",
        ":saved_model_load_testdata",
    ],
)
