load("//third_party/bazel_rules/rules_python/python:py_library.bzl", "py_library")
# Placeholder: load py_binary
load("@tensorflow_gnn//tensorflow_gnn:tensorflow_gnn.bzl", "tf_py_test")

# Copybara rewrites load() statements back and forth; do not reformat.
# buildifier: disable=out-of-order-load, disable=same-origin-load

# buildifier: disable=out-of-order-load, disable=same-origin-load

# buildifier: disable=out-of-order-load, disable=same-origin-load

licenses(["notice"])

package(
    default_applicable_licenses = ["//tensorflow_gnn:license"],
    default_visibility = ["//visibility:public"],
)

py_library(
    name = "attribution",
    srcs = ["attribution.py"],
    type_checking = True,
    visibility = ["//tensorflow_gnn/runner:__pkg__"],
    deps = [
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
        "//tensorflow_gnn/runner:interfaces",
    ],
)

py_library(
    name = "label_fns",
    srcs = ["label_fns.py"],
    type_checking = True,
    visibility = ["//tensorflow_gnn/runner:__pkg__"],
    deps = [
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
    ],
)

py_library(
    name = "model_dir",
    srcs = ["model_dir.py"],
    type_checking = True,
    visibility = ["//tensorflow_gnn/runner:__pkg__"],
    deps = [
        "//:expect_tensorflow_installed",
    ],
)

py_library(
    name = "model_export",
    srcs = ["model_export.py"],
    type_checking = True,
    visibility = ["//tensorflow_gnn/runner:__pkg__"],
    deps = [
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn/runner:interfaces",
    ],
)

py_library(
    name = "padding",
    srcs = ["padding.py"],
    type_checking = True,
    visibility = ["//tensorflow_gnn/runner:__pkg__"],
    deps = [
        ":parsing",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
        "//tensorflow_gnn/runner:interfaces",
    ],
)

py_library(
    name = "parsing",
    srcs = ["parsing.py"],
    type_checking = True,
    visibility = ["//tensorflow_gnn/runner:__pkg__"],
    deps = [
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
    ],
)

py_library(
    name = "strategies",
    srcs = ["strategies.py"],
    type_checking = True,
    visibility = ["//tensorflow_gnn/runner:__pkg__"],
    deps = [
        "//:expect_tensorflow_installed",
    ],
)

py_library(
    name = "test_utils",
    srcs = ["test_utils.py"],
    type_checking = True,
    deps = [
        "//tensorflow_gnn/runner:interfaces",
    ],
)

tf_py_test(
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

tf_py_test(
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

tf_py_test(
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

py_binary(
    name = "saved_model_gen_testdata",
    srcs = ["saved_model_gen_testdata.py"],
    type_checking = True,
    deps = [
        "//:expect_absl_installed_app",
        "//:expect_absl_installed_flags",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
        "//tensorflow_gnn/runner",
    ],
)

py_binary(
    name = "saved_model_load_testdata",
    srcs = ["saved_model_load_testdata.py"],
    type_checking = True,
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
