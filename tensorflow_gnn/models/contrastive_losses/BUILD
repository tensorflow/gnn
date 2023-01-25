load("@tensorflow_gnn//tensorflow_gnn:tensorflow_gnn.bzl", "pytype_strict_library")

licenses(["notice"])

package_group(name = "users")

pytype_strict_library(
    name = "contrastive_losses",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = [
        ":users",
    ],
    deps = [
        ":layers",
        "//tensorflow_gnn/models/contrastive_losses/deep_graph_infomax:layers",
        "//tensorflow_gnn/models/contrastive_losses/deep_graph_infomax:tasks",
    ],
)

pytype_strict_library(
    name = "layers",
    srcs = ["layers.py"],
    srcs_version = "PY3",
    visibility = [
        ":__subpackages__",
    ],
    deps = [
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
    ],
)
