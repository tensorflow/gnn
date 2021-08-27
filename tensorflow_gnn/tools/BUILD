package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

load("@com_google_protobuf//:protobuf.bzl", "py_proto_library")

py_proto_library(
    name = "sampled_stats_py_proto",
    srcs = ["sampled_stats.proto"],
    default_runtime = "@com_google_protobuf//:protobuf_python",
    protoc = "@com_google_protobuf//:protoc",
    srcs_version = "PY3",
    deps = [
        "@com_google_protobuf//:protobuf_python",
    ],
)
