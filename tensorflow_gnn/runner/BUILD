load("@tensorflow_gnn//tensorflow_gnn:tensorflow_gnn.bzl", "pytype_strict_library")
load("@tensorflow_gnn//tensorflow_gnn:tensorflow_gnn.bzl", "distribute_py_test")

licenses(["notice"])

pytype_strict_library(
    name = "runner",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":orchestration",
        "//tensorflow_gnn/runner/input:datasets",
        "//tensorflow_gnn/runner/tasks:classification",
        "//tensorflow_gnn/runner/tasks:dgi",
        "//tensorflow_gnn/runner/tasks:regression",
        "//tensorflow_gnn/runner/trainers:keras_fit",
        "//tensorflow_gnn/runner/utils:attribution",
        "//tensorflow_gnn/runner/utils:model",
        "//tensorflow_gnn/runner/utils:model_dir",
        "//tensorflow_gnn/runner/utils:model_export",
        "//tensorflow_gnn/runner/utils:model_templates",
        "//tensorflow_gnn/runner/utils:padding",
        "//tensorflow_gnn/runner/utils:strategies",
    ],
)

pytype_strict_library(
    name = "orchestration",
    srcs = ["orchestration.py"],
    srcs_version = "PY3",
    visibility = [":__subpackages__"],
    deps = [
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
        "//tensorflow_gnn/runner/utils:model",
        "//tensorflow_gnn/runner/utils:model_export",
    ],
)

distribute_py_test(
    name = "orchestration_test",
    size = "medium",
    srcs = ["orchestration_test.py"],
    python_version = "PY3",
    shard_count = 4,
    tags = [
        "no_oss",  # TODO(b/238827505)
        "nomultivm",  # TODO(b/170502145)
    ],
    xla_enable_strict_auto_jit = False,
    deps = [
        ":orchestration",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
        "//tensorflow_gnn/models/vanilla_mpnn",
        "//tensorflow_gnn/runner/tasks:classification",
        "//tensorflow_gnn/runner/trainers:keras_fit",
        "//tensorflow_gnn/runner/utils:model_templates",
        "//tensorflow_gnn/runner/utils:padding",
    ],
)
