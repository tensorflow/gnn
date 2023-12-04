load("@tensorflow_gnn//tensorflow_gnn:tensorflow_gnn.bzl", "pytype_strict_binary", "pytype_strict_library")
load("@tensorflow_gnn//tensorflow_gnn:tensorflow_gnn.bzl", "py_strict_test")

licenses(["notice"])

package(
    default_applicable_licenses = ["//tensorflow_gnn:license"],
    default_visibility = ["//visibility:public"],
)

genrule(
    name = "saved_model_testdata",
    outs = [
        "saved_model_testdata/fingerprint.pb",
        "saved_model_testdata/keras_metadata.pb",
        "saved_model_testdata/saved_model.pb",
        "saved_model_testdata/variables/variables.data-00000-of-00001",
        "saved_model_testdata/variables/variables.index",
    ],
    cmd = "$(location saved_model_gen_testdata) --filepath=$(RULEDIR)/saved_model_testdata",
    local = True,
    tools = [":saved_model_gen_testdata"],
)

pytype_strict_binary(
    name = "saved_model_gen_testdata",
    srcs = ["saved_model_gen_testdata.py"],
    python_version = "PY3",
    srcs_version = "PY3ONLY",
    deps = [
        ":model_export",
        "//:expect_absl_installed_app",
        "//:expect_absl_installed_flags",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn/graph:graph_constants",
        "//tensorflow_gnn/graph:graph_tensor",
        "//tensorflow_gnn/keras/layers",
    ],
)

pytype_strict_library(
    name = "attribution",
    srcs = ["attribution.py"],
    srcs_version = "PY3",
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
    srcs_version = "PY3",
    visibility = ["//tensorflow_gnn/runner:__pkg__"],
    deps = [
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
    ],
)

pytype_strict_library(
    name = "model_dir",
    srcs = ["model_dir.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_gnn/runner:__pkg__"],
    deps = [
        "//:expect_tensorflow_installed",
    ],
)

pytype_strict_library(
    name = "model_export",
    srcs = ["model_export.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_gnn/runner:__pkg__"],
    deps = [
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn/runner:interfaces",
    ],
)

pytype_strict_library(
    name = "padding",
    srcs = ["padding.py"],
    srcs_version = "PY3",
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
    srcs_version = "PY3",
    visibility = ["//tensorflow_gnn/runner:__pkg__"],
    deps = [
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
    ],
)

pytype_strict_library(
    name = "strategies",
    srcs = ["strategies.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_gnn/runner:__pkg__"],
    deps = [
        "//:expect_tensorflow_installed",
    ],
)

py_strict_test(
    name = "attribution_test",
    srcs = ["attribution_test.py"],
    srcs_version = "PY3",
    deps = [
        ":attribution",
        "//third_party/py/google/protobuf:use_fast_cpp_protos",  # Automatically added go/proto_python_upb_flip
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
        "//tensorflow_gnn/runner:interfaces",
    ],
)

py_strict_test(
    name = "model_export_test",
    srcs = ["model_export_test.py"],
    python_version = "PY3",
    deps = [
        ":model_export",
        "//:expect_absl_installed_testing",
        "//third_party/py/google/protobuf:use_fast_cpp_protos",  # Automatically added go/proto_python_upb_flip
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
        "//tensorflow_gnn/runner:interfaces",
    ],
)

py_strict_test(
    name = "parsing_test",
    srcs = ["parsing_test.py"],
    srcs_version = "PY3",
    deps = [
        ":parsing",
        "//:expect_absl_installed_testing",
        "//third_party/py/google/protobuf:use_fast_cpp_protos",  # Automatically added go/proto_python_upb_flip
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
    ],
)

py_strict_test(
    name = "saved_model_test",
    srcs = ["saved_model_test.py"],
    data = [":saved_model_testdata"],
    python_version = "PY3",
    srcs_version = "PY3ONLY",
    deps = [
        ":model_export",
        "//:expect_absl_installed_app",
        "//:expect_absl_installed_flags",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
    ],
)
