load("@tensorflow_gnn//tensorflow_gnn:tensorflow_gnn.bzl", "pytype_strict_library")
load("@tensorflow_gnn//tensorflow_gnn:tensorflow_gnn.bzl", "tf_py_test")

licenses(["notice"])

package(
    default_visibility = [":__subpackages__"],
)

package_group(name = "users")

pytype_strict_library(
    name = "hgt",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = [
        ":__subpackages__",
        ":users",
    ],
    deps = [],
)

pytype_strict_library(
    name = "softmax",
    srcs = ["softmax.py"],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
    ],
)

tf_py_test(
    name = "softmax_test",
    srcs = ["softmax_test.py"],
    srcs_version = "PY3",
    deps = [
        ":softmax",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
    ],
)
