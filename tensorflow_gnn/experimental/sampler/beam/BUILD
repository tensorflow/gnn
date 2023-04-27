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
