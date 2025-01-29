# Copybara rewrites load() statements back and forth; do not reformat.
# buildifier: disable=out-of-order-load, disable=same-origin-load
# Placeholder: load py_binary

# buildifier: disable=out-of-order-load, disable=same-origin-load
load("@tensorflow_gnn//tensorflow_gnn:tensorflow_gnn.bzl", "pytype_strict_contrib_test")

# buildifier: disable=out-of-order-load, disable=same-origin-load
load("@tensorflow_gnn//tensorflow_gnn:tensorflow_gnn.bzl", "pytype_strict_library")

licenses(["notice"])

package(
    default_applicable_licenses = ["//tensorflow_gnn:license"],
    default_visibility = ["//visibility:public"],
)

pytype_strict_library(
    name = "executor_lib",
    srcs = ["executor_lib.py"],
    deps = [
        ":utils",
        "//third_party/py/apache_beam",
        "//third_party/py/apache_beam/utils",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn/experimental/sampler",
        "//tensorflow_gnn/experimental/sampler/proto",
    ],
)

pytype_strict_contrib_test(
    name = "executor_lib_test",
    srcs = ["executor_lib_test.py"],
    deps = [
        ":executor_lib",
        "//third_party/py/absl/testing:absltest",
        "//:expect_absl_installed_testing",
        "//third_party/py/apache_beam",
        "//third_party/py/google/protobuf:use_fast_cpp_protos",  # Automatically added go/proto_python_upb_flip
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn/experimental/sampler",
    ],
)

pytype_strict_contrib_test(
    name = "subgraph_pipeline_test",
    srcs = ["subgraph_pipeline_test.py"],
    deps = [
        ":accessors",
        ":edge_samplers",
        ":executor_lib",
        "//third_party/py/absl/testing:absltest",
        "//:expect_absl_installed_testing",
        "//third_party/py/apache_beam",
        "//third_party/py/apache_beam/utils",
        "//third_party/py/google/protobuf:use_fast_cpp_protos",  # Automatically added go/proto_python_upb_flip
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
        "//tensorflow_gnn/experimental/sampler",
    ],
)

pytype_strict_library(
    name = "accessors",
    srcs = ["accessors.py"],
    deps = [
        ":executor_lib",
        ":utils",
        "//third_party/py/apache_beam",
        "//third_party/py/apache_beam/utils",
        "//:expect_numpy_installed",
        "//tensorflow_gnn/experimental/sampler/proto",
    ],
)

pytype_strict_contrib_test(
    name = "accessors_test",
    srcs = ["accessors_test.py"],
    deps = [
        ":accessors",
        ":executor_lib",
        "//third_party/py/absl/testing:absltest",
        "//:expect_absl_installed_testing",
        "//third_party/py/apache_beam",
        "//third_party/py/google/protobuf:use_fast_cpp_protos",  # Automatically added go/proto_python_upb_flip
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn/experimental/sampler",
    ],
)

pytype_strict_library(
    name = "edge_samplers",
    srcs = ["edge_samplers.py"],
    deps = [
        ":executor_lib",
        ":utils",
        "//third_party/py/apache_beam",
        "//third_party/py/apache_beam/utils",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
        "//tensorflow_gnn/experimental/sampler/proto",
    ],
)

pytype_strict_contrib_test(
    name = "edge_samplers_test",
    srcs = ["edge_samplers_test.py"],
    deps = [
        ":accessors",
        ":edge_samplers",
        ":executor_lib",
        "//third_party/py/absl/testing:absltest",
        "//:expect_absl_installed_testing",
        "//third_party/py/apache_beam",
        "//third_party/py/google/protobuf:use_fast_cpp_protos",  # Automatically added go/proto_python_upb_flip
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn/experimental/sampler",
    ],
)

pytype_strict_library(
    name = "utils",
    srcs = ["utils.py"],
    deps = [
        "//third_party/py/apache_beam",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn/experimental/sampler/proto",
    ],
)

pytype_strict_contrib_test(
    name = "utils_test",
    srcs = ["utils_test.py"],
    deps = [
        ":utils",
        "//third_party/py/absl/testing:absltest",
        "//:expect_absl_installed_testing",
        "//third_party/py/apache_beam",
        "//third_party/py/google/protobuf:use_fast_cpp_protos",  # Automatically added go/proto_python_upb_flip
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn/experimental/sampler/proto",
    ],
)

pytype_strict_library(
    name = "unigraph_utils",
    srcs = ["unigraph_utils.py"],
    deps = [
        ":executor_lib",
        "//third_party/py/apache_beam",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
        "//tensorflow_gnn/data:unigraph",
        "//tensorflow_gnn/sampler",
    ],
)

pytype_strict_library(
    name = "sampler-lib",
    srcs = ["sampler.py"],
    deps = [
        ":accessors",
        ":edge_samplers",
        ":executor_lib",
        ":unigraph_utils",
        "//:expect_absl_installed_app",
        "//:expect_absl_installed_flags",
        "//third_party/py/absl/logging",
        "//third_party/py/apache_beam",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
        "//tensorflow_gnn/data:unigraph",
        "//tensorflow_gnn/experimental/sampler",
        "//tensorflow_gnn/experimental/sampler:subgraph_pipeline",
        "//tensorflow_gnn/proto:graph_schema_py_proto",
        "//tensorflow_gnn/sampler",
    ],
)

pytype_strict_contrib_test(
    name = "sampler_test",
    srcs = ["sampler_test.py"],
    deps = [
        ":sampler-lib",
        "//testing/pybase",
        "//:expect_absl_installed_testing",
        "//third_party/py/google/protobuf:use_fast_cpp_protos",  # Automatically added go/proto_python_upb_flip
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
        "//tensorflow_gnn/sampler",
    ],
)

py_binary(
    name = "sampler",
    srcs = ["sampler.py"],
    deps = [
        ":sampler-lib",
        "//third_party/py/google/protobuf:use_fast_cpp_protos",  # Automatically added go/proto_python_upb_flip
    ],
)

pytype_strict_contrib_test(
    name = "unigraph_utils_test",
    srcs = ["unigraph_utils_test.py"],
    data = [
        "@tensorflow_gnn//testdata/heterogeneous",
        "@tensorflow_gnn//testdata/homogeneous",
    ],
    deps = [
        ":unigraph_utils",
        "//testing/pybase",
        "//third_party/py/apache_beam",
        "//third_party/py/google/protobuf:use_fast_cpp_protos",  # Automatically added go/proto_python_upb_flip
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
        "//tensorflow_gnn/data:unigraph",
        "//tensorflow_gnn/utils:test_utils",
    ],
)
