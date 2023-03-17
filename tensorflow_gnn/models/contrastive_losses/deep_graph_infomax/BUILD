load("@tensorflow_gnn//tensorflow_gnn:tensorflow_gnn.bzl", "pytype_strict_library")
load("@tensorflow_gnn//tensorflow_gnn:tensorflow_gnn.bzl", "tf_py_test")

licenses(["notice"])

pytype_strict_library(
    name = "layers",
    srcs = ["layers.py"],
    srcs_version = "PY3",
    visibility = [
        "//tensorflow_gnn/models/contrastive_losses:__pkg__",
    ],
    deps = [
        "//:expect_tensorflow_installed",
    ],
)

tf_py_test(
    name = "layers_test",
    srcs = ["layers_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":layers",
        "//:expect_absl_installed",
        "//:expect_tensorflow_installed",
    ],
)
