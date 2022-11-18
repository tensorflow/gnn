load("@tensorflow_gnn//tensorflow_gnn:tensorflow_gnn.bzl", "pytype_strict_contrib_test", "pytype_strict_library")
load("@tensorflow_gnn//tensorflow_gnn:tensorflow_gnn.bzl", "tf_py_test")

licenses(["notice"])

package(
    default_visibility = [":__subpackages__"],
)

package_group(name = "users")

# The common BUILD target for the actual modeling code.
pytype_strict_library(
    name = "vanilla_mpnn",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = [
        ":__subpackages__",
        ":users",
    ],
    deps = [
        ":config_dict",
        ":layers",
    ],
)

# A special BUILD target with a declaration of the model's hyperparameter search space.
# Unlike the model itself, this does not depend on TF/TF-GNN, but does depend on Vizier.
pytype_strict_library(
    name = "hparams_vizier",
    srcs = ["hparams_vizier.py"],
    visibility = [
        ":__subpackages__",
        ":users",
    ],
    deps = [
        "//:expect_vizier_service_pyvizier_installed",
    ],
)

exports_files(
    srcs = ["hparams_vizier.py"],
    visibility = [
        ":__subpackages__",
    ],
)

# =============================================================================

pytype_strict_contrib_test(
    name = "hparams_vizier_test",
    srcs = ["hparams_vizier_test.py"],
    python_version = "PY3",
    deps = [
        ":hparams_vizier",
        "//:expect_vizier_service_pyvizier_installed",
        "//third_party/py/absl/testing:absltest",
    ],
)

pytype_strict_library(
    name = "layers",
    srcs = ["layers.py"],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
    ],
)

pytype_strict_library(
    name = "config_dict",
    srcs = ["config_dict.py"],
    deps = [
        ":layers",
        "//third_party/py/ml_collections/config_dict",
        "//:expect_tensorflow_installed",
    ],
)

tf_py_test(
    name = "config_dict_test",
    srcs = ["config_dict_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":config_dict",
        ":layers",
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
        ":vanilla_mpnn",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
    ],
)
