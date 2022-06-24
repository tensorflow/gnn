load("@tensorflow_gnn//tensorflow_gnn:tensorflow_gnn.bzl", "pytype_strict_binary")

licenses(["notice"])

pytype_strict_binary(
    name = "main",
    srcs = ["main.py"],
    python_version = "PY3",
    deps = [
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/absl/logging",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
        "//tensorflow_gnn/models/vanilla_mpnn",
        "//tensorflow_gnn/runner",
    ],
)
