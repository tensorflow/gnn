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
