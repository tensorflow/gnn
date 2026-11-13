load("//third_party/bazel_rules/rules_python/python:py_test.bzl", "py_test")
load("//third_party/bazel_rules/rules_python/python:py_library.bzl", "py_library")

# Copybara rewrites load() statements back and forth; do not reformat.
# buildifier: disable=out-of-order-load, disable=same-origin-load

# buildifier: disable=out-of-order-load, disable=same-origin-load

# buildifier: disable=out-of-order-load, disable=same-origin-load
load("@tensorflow_gnn//tensorflow_gnn:tensorflow_gnn.bzl", "tf_py_test")

licenses(["notice"])

package(
    default_applicable_licenses = ["//tensorflow_gnn:license"],  # copybara:strip
    default_visibility = [":__subpackages__"],
)

# This is the public BUILD target for users of this model.
py_library(
    name = "mt_albis",
    srcs = ["__init__.py"],
    type_checking = True,
    visibility = ["//visibility:public"],
    deps = [
        ":config_dict",
        ":layers",
        "//tensorflow_gnn/utils:api_utils",
    ],
)

# A special BUILD target with a declaration of the model's hyperparameter search space.
# Unlike the model itself, this does not depend on TF/TF-GNN, but does depend on Vizier.
py_library(
    name = "hparams_vizier",
    srcs = ["hparams_vizier.py"],
    type_checking = True,
    visibility = ["//visibility:public"],
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

py_test(
    name = "hparams_vizier_test",
    srcs = ["hparams_vizier_test.py"],
    tags = ["vizier"],
    type_checking = True,
    deps = [
        ":hparams_vizier",
        "//:expect_vizier_service_pyvizier_installed",
        "//third_party/py/absl/testing:absltest",
    ],
)

py_library(
    name = "layers",
    srcs = ["layers.py"],
    type_checking = True,
    deps = [
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
        "//tensorflow_gnn/models/gat_v2",
        "//tensorflow_gnn/models/multi_head_attention",
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
        "//tensorflow_gnn/utils:tf_test_utils",
        "//:expect_ai_edge_litert_installed",
    ],
)

py_library(
    name = "config_dict",
    srcs = ["config_dict.py"],
    type_checking = True,
    deps = [
        ":layers",
        "//third_party/py/ml_collections/config_dict",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
    ],
)

tf_py_test(
    name = "config_dict_test",
    srcs = ["config_dict_test.py"],
    deps = [
        ":config_dict",
        ":layers",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
    ],
)
