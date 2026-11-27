load("@tensorflow_gnn//tensorflow_gnn:tensorflow_gnn.bzl", "pytype_strict_library")
load("@tensorflow_gnn//tensorflow_gnn:tensorflow_gnn.bzl", "tf_py_test")

pytype_strict_library(
    name = "utils",
    srcs = ["utils.py"],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
    ],
)

tf_py_test(
    name = "utils_test",
    srcs = ["utils_test.py"],
    python_version = "PY3",
    deps = [
        ":utils",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
    ],
)
