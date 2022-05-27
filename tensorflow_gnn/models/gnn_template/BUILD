load("@tensorflow_gnn//tensorflow_gnn:tensorflow_gnn.bzl", "pytype_strict_library")
load("@tensorflow_gnn//tensorflow_gnn:tensorflow_gnn.bzl", "tf_py_test")

licenses(["notice"])

package(
    default_visibility = [":__subpackages__"],
)

package_group(name = "users")

pytype_strict_library(
    name = "gnn_template",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = [
        ":__subpackages__",
        ":users",
    ],
    deps = [
        ":modeling",
    ],
)

pytype_strict_library(
    name = "gin_configurables",
    srcs = ["gin_configurables.py"],
    srcs_version = "PY3",
    visibility = [
        ":__subpackages__",
        ":users",
    ],
    deps = [":gnn_template"],
)

# =============================================================================

pytype_strict_library(
    name = "modeling",
    srcs = ["modeling.py"],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
    ],
)

tf_py_test(
    name = "modeling_test",
    srcs = ["modeling_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":modeling",
        "//:expect_absl_installed",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
    ],
)
