load("@tensorflow_gnn//tensorflow_gnn:tensorflow_gnn.bzl", "pytype_strict_library")
load("@tensorflow_gnn//tensorflow_gnn:tensorflow_gnn.bzl", "tf_py_test")

licenses(["notice"])

package(
    default_visibility = [":__subpackages__"],
)

package_group(
    name = "users",
    packages = [
        "//third_party/google_research/google_research/ugsl/...",
    ],
)

pytype_strict_library(
    name = "gcn",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = [
        ":__subpackages__",
        ":users",
    ],
    deps = [
        ":gcn_conv",
    ],
)

pytype_strict_library(
    name = "gcn_conv",
    srcs = ["gcn_conv.py"],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
    ],
)

tf_py_test(
    name = "gcn_conv_test",
    srcs = ["gcn_conv_test.py"],
    srcs_version = "PY3",
    deps = [
        ":gcn_conv",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
    ],
)
