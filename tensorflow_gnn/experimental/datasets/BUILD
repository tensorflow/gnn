load("@tensorflow_gnn//tensorflow_gnn:tensorflow_gnn.bzl", "pytype_strict_library")
load("@tensorflow_gnn//tensorflow_gnn:tensorflow_gnn.bzl", "py_strict_test")

pytype_strict_library(
    name = "pyg_adapter",
    srcs = ["pyg_adapter.py"],
    srcs_version = "PY3ONLY",
    deps = [
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
    ],
)

py_strict_test(
    name = "pyg_adapter_test",
    srcs = ["pyg_adapter_test.py"],
    python_version = "PY3",
    srcs_version = "PY3ONLY",
    deps = [
        ":pyg_adapter",
        "//testing/pybase",
        "//third_party/py/absl/testing:absltest",
        "//:expect_absl_installed_testing",
        "//third_party/py/google/protobuf:use_fast_cpp_protos",  # Automatically added go/proto_python_upb_flip
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
    ],
)
