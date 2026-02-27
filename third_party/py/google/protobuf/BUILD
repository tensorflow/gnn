# OSS stub for third_party/py/google/protobuf.
# These are Google-internal Bazel targets referenced in BUILD files.

load("@tensorflow_gnn//tensorflow_gnn:tensorflow_gnn.bzl", "pytype_strict_library")

package(default_applicable_licenses = ["//tensorflow_gnn:license"])

pytype_strict_library(
    name = "protobuf",
    srcs = [],
    visibility = ["//visibility:public"],
    deps = [],
)

# No-op target. In Google's internal build, this selects fast C++ protos.
# In OSS builds, this is not needed.
pytype_strict_library(
    name = "use_fast_cpp_protos",
    srcs = [],
    visibility = ["//visibility:public"],
    deps = [],
)
