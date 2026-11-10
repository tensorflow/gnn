load("//third_party/bazel_rules/rules_python/python:py_library.bzl", "py_library")

# Copybara rewrites load() statements back and forth; do not reformat.
# buildifier: disable=out-of-order-load, disable=same-origin-load

# buildifier: disable=out-of-order-load, disable=same-origin-load
load("@tensorflow_gnn//tensorflow_gnn:tensorflow_gnn.bzl", "tf_py_test")

# buildifier: disable=out-of-order-load, disable=same-origin-load
load("@tensorflow_gnn//tensorflow_gnn:tensorflow_gnn.bzl", "distribute_py_test")

package(
    default_applicable_licenses = ["//tensorflow_gnn:license"],
    default_visibility = ["//visibility:private"],
)

licenses(["notice"])

package_group(name = "users")

py_library(
    name = "contrastive_losses",
    srcs = ["__init__.py"],
    type_checking = True,
    visibility = [
        ":users",
    ],
    deps = [
        ":layers",
        ":metrics",
        ":tasks",
        "//tensorflow_gnn/utils:api_utils",
    ],
)

py_library(
    name = "layers",
    srcs = ["layers.py"],
    type_checking = True,
    deps = [
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
    ],
)

tf_py_test(
    name = "layers_test",
    srcs = ["layers_test.py"],
    deps = [
        ":layers",
        "//:expect_absl_installed_testing",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
    ],
)

py_library(
    name = "losses",
    srcs = ["losses.py"],
    type_checking = True,
    deps = [
        "//:expect_tensorflow_installed",
    ],
)

tf_py_test(
    name = "losses_test",
    srcs = ["losses_test.py"],
    deps = [
        ":tasks",
        "//:expect_absl_installed_testing",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
    ],
)

py_library(
    name = "metrics",
    srcs = ["metrics.py"],
    type_checking = True,
    deps = [
        "//:expect_tensorflow_installed",
    ],
)

tf_py_test(
    name = "metrics_test",
    srcs = ["metrics_test.py"],
    deps = [
        ":metrics",
        "//:expect_tensorflow_installed",
    ],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    type_checking = True,
    deps = [
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
    ],
)

tf_py_test(
    name = "utils_test",
    srcs = ["utils_test.py"],
    deps = [
        ":utils",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
    ],
)

py_library(
    name = "tasks",
    srcs = ["tasks.py"],
    type_checking = True,
    deps = [
        ":layers",
        ":losses",
        ":metrics",
        ":utils",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
        "//tensorflow_gnn/runner",
    ],
)

tf_py_test(
    name = "tasks_test",
    srcs = ["tasks_test.py"],
    deps = [
        ":tasks",
        "//:expect_absl_installed_testing",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
        "//tensorflow_gnn/models/vanilla_mpnn",
    ],
)

distribute_py_test(
    name = "distribute_test",
    size = "large",
    srcs = ["distribute_test.py"],
    shard_count = 8,
    tags = [
        "no_oss",  # TODO(b/238827505)
        "nomultivm",  # TODO(b/170502145)
    ],
    xla_enable_strict_auto_jit = False,
    deps = [
        ":tasks",
        "//:expect_absl_installed_testing",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
        "//tensorflow_gnn/models/vanilla_mpnn",
        "//tensorflow_gnn/runner",
    ],
)
