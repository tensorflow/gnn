load("@tensorflow_gnn//tensorflow_gnn:tensorflow_gnn.bzl", "pytype_strict_contrib_test", "pytype_strict_library")

licenses(["notice"])

package(
    default_applicable_licenses = ["//tensorflow_gnn:license"],
    default_visibility = ["//visibility:public"],
)

pytype_strict_library(
    name = "executor_lib",
    srcs = ["executor_lib.py"],
    srcs_version = "PY3ONLY",
    deps = [
        ":utils",
        "//third_party/py/apache_beam",
        "//third_party/py/apache_beam/utils",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn/experimental/sampler",
        "//tensorflow_gnn/experimental/sampler:eval_dag_py_proto",
    ],
)

pytype_strict_contrib_test(
    name = "executor_lib_test",
    srcs = ["executor_lib_test.py"],
    python_version = "PY3",
    srcs_version = "PY3ONLY",
    deps = [
        ":executor_lib",
        "//third_party/py/absl/testing:absltest",
        "//:expect_absl_installed",
        "//third_party/py/apache_beam",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn/experimental/sampler",
    ],
)

pytype_strict_library(
    name = "accessors",
    srcs = ["accessors.py"],
    srcs_version = "PY3ONLY",
    deps = [
        ":executor_lib",
        ":utils",
        "//third_party/py/apache_beam",
        "//third_party/py/apache_beam/utils",
        "//:expect_numpy_installed",
        "//tensorflow_gnn/experimental/sampler:eval_dag_py_proto",
    ],
)

pytype_strict_contrib_test(
    name = "accessors_test",
    srcs = ["accessors_test.py"],
    python_version = "PY3",
    srcs_version = "PY3ONLY",
    deps = [
        ":accessors",
        ":executor_lib",
        "//third_party/py/absl/testing:absltest",
        "//:expect_absl_installed",
        "//third_party/py/apache_beam",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn/experimental/sampler",
    ],
)

pytype_strict_library(
    name = "edge_samplers",
    srcs = ["edge_samplers.py"],
    srcs_version = "PY3ONLY",
    deps = [
        ":executor_lib",
        ":utils",
        "//third_party/py/apache_beam",
        "//third_party/py/apache_beam/utils",
        "//:expect_numpy_installed",
        "//tensorflow_gnn/experimental/sampler:eval_dag_py_proto",
    ],
)

pytype_strict_contrib_test(
    name = "edge_samplers_test",
    srcs = ["edge_samplers_test.py"],
    python_version = "PY3",
    srcs_version = "PY3ONLY",
    deps = [
        ":accessors",
        ":edge_samplers",
        ":executor_lib",
        "//third_party/py/absl/testing:absltest",
        "//:expect_absl_installed",
        "//third_party/py/apache_beam",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn/experimental/sampler",
    ],
)

pytype_strict_library(
    name = "utils",
    srcs = ["utils.py"],
    srcs_version = "PY3ONLY",
    deps = [
        "//third_party/py/apache_beam",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn/experimental/sampler:eval_dag_py_proto",
    ],
)

pytype_strict_contrib_test(
    name = "utils_test",
    srcs = ["utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3ONLY",
    deps = [
        ":utils",
        "//third_party/py/absl/testing:absltest",
        "//:expect_absl_installed",
        "//third_party/py/apache_beam",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
    ],
)

pytype_strict_library(
    name = "unigraph_utils",
    srcs = ["unigraph_utils.py"],
    srcs_version = "PY3ONLY",
    deps = [
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
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
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
        "//:expect_absl_installed",
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
    ],
)

pytype_strict_contrib_test(
    name = "unigraph_utils_test",
    srcs = ["unigraph_utils_test.py"],
    data = ["@tensorflow_gnn//testdata/heterogeneous"],
    python_version = "PY3",
    srcs_version = "PY3ONLY",
    deps = [
        ":unigraph_utils",
        "//testing/pybase",
        "//third_party/py/apache_beam",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
        "//tensorflow_gnn/data:unigraph",
        "//tensorflow_gnn/utils:test_utils",
    ],
)
