load("@tensorflow_gnn//tensorflow_gnn:tensorflow_gnn.bzl", "pytype_strict_binary", "pytype_strict_library")

licenses(["notice"])

pytype_strict_library(
    name = "train_lib",
    srcs = ["train.py"],
    visibility = [
        ":__subpackages__",
    ],
    deps = [
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/absl/logging",
        "//third_party/py/ml_collections/config_dict",
        "//third_party/py/ml_collections/config_flags",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
        "//tensorflow_gnn/models/multi_head_attention",
        "//tensorflow_gnn/models/vanilla_mpnn",
        "//tensorflow_gnn/runner",
    ],
)

pytype_strict_binary(
    name = "train",
    srcs = ["train.py"],
    python_version = "PY3",
    deps = [":train_lib"],
)
