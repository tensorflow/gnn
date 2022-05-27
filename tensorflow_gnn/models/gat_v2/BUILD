load("@tensorflow_gnn//tensorflow_gnn:tensorflow_gnn.bzl", "pytype_strict_library")
load("@tensorflow_gnn//tensorflow_gnn:tensorflow_gnn.bzl", "tf_py_test")

licenses(["notice"])

package(
    default_visibility = [":__subpackages__"],
)

package_group(name = "users")

pytype_strict_library(
    name = "gat_v2",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = [
        ":__subpackages__",
        ":users",
    ],
    deps = [
        ":layers",
    ],
)

# =============================================================================

pytype_strict_library(
    name = "layers",
    srcs = ["layers.py"],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
    ],
)

tf_py_test(
    name = "layers_test",
    srcs = ["layers_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":gat_v2",
        "//:expect_absl_installed",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
    ],
)

pytype_strict_library(
    name = "gat_v2_model",
    srcs = ["gat_v2_model.py"],
    deps = [
        ":gat_v2",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
    ],
)
