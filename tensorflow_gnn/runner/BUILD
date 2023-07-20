load("@tensorflow_gnn//tensorflow_gnn:tensorflow_gnn.bzl", "pytype_strict_library")
load("@tensorflow_gnn//tensorflow_gnn:tensorflow_gnn.bzl", "py_strict_test")
load("@tensorflow_gnn//tensorflow_gnn:tensorflow_gnn.bzl", "distribute_py_test")

licenses(["notice"])

package(
    default_applicable_licenses = ["//tensorflow_gnn:license"],
    default_visibility = ["//visibility:public"],
)

pytype_strict_library(
    name = "runner",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":interfaces",
        ":orchestration",
        "//tensorflow_gnn/runner/input:datasets",
        "//tensorflow_gnn/runner/tasks:classification",
        "//tensorflow_gnn/runner/tasks:link_prediction",
        "//tensorflow_gnn/runner/tasks:regression",
        "//tensorflow_gnn/runner/trainers:keras_fit",
        "//tensorflow_gnn/runner/utils:attribution",
        "//tensorflow_gnn/runner/utils:label_fns",
        "//tensorflow_gnn/runner/utils:model_dir",
        "//tensorflow_gnn/runner/utils:model_export",
        "//tensorflow_gnn/runner/utils:padding",
        "//tensorflow_gnn/runner/utils:strategies",
    ],
)

pytype_strict_library(
    name = "interfaces",
    srcs = ["interfaces.py"],
    srcs_version = "PY3",
    visibility = [":__subpackages__"],
    deps = [
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
    ],
)

pytype_strict_library(
    name = "orchestration",
    srcs = ["orchestration.py"],
    srcs_version = "PY3",
    deps = [
        ":interfaces",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
        "//tensorflow_gnn/runner/utils:model_export",
        "//tensorflow_gnn/runner/utils:parsing",
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
        ":orchestration",
        "//third_party/py/immutabledict",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
        "//tensorflow_gnn/models/vanilla_mpnn",
        "//tensorflow_gnn/runner/tasks:classification",
        "//tensorflow_gnn/runner/tasks:regression",
        "//tensorflow_gnn/runner/trainers:keras_fit",
        "//tensorflow_gnn/runner/utils:label_fns",
        "//tensorflow_gnn/runner/utils:padding",
    ],
)

py_strict_test(
    name = "orchestration_test",
    srcs = ["orchestration_test.py"],
    deps = [
        ":interfaces",
        ":orchestration",
        "//:expect_absl_installed",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
        "//tensorflow_gnn/models/vanilla_mpnn",
        "//tensorflow_gnn/runner/tasks:classification",
        "//tensorflow_gnn/runner/trainers:keras_fit",
        "//tensorflow_gnn/runner/utils:label_fns",
    ],
)
