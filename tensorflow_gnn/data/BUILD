load("@tensorflow_gnn//tensorflow_gnn:tensorflow_gnn.bzl", "pytype_strict_contrib_test", "pytype_strict_library")

licenses(["notice"])

package(
    default_applicable_licenses = ["//tensorflow_gnn:license"],
    default_visibility = ["//visibility:public"],
)

GOOGLE_INTERNAL_UNIGRAPH_DEPENDENCIES = []

pytype_strict_library(
    name = "unigraph",
    srcs = ["unigraph.py"],
    srcs_version = "PY3ONLY",
    deps = [
        "//third_party/py/apache_beam",
        "//third_party/py/pyarrow",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
        "//tensorflow_gnn/proto:graph_schema_py_proto",
    ] + GOOGLE_INTERNAL_UNIGRAPH_DEPENDENCIES,
)

pytype_strict_contrib_test(
    name = "unigraph_test",
    srcs = ["unigraph_test.py"],
    data = [
        "@tensorflow_gnn//testdata/heterogeneous",
        "@tensorflow_gnn//testdata/homogeneous",
    ],
    python_version = "PY3",
    srcs_version = "PY3ONLY",
    deps = [
        ":unigraph",
        "//third_party/py/apache_beam",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
        "//tensorflow_gnn/utils:test_utils",
    ],
)
