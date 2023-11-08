package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

load("@rules_proto//proto:defs.bzl", "proto_library")
load("@rules_python//python:proto.bzl", "py_proto_library")
load("@tensorflow_gnn//tensorflow_gnn:tensorflow_gnn.bzl", "pytype_strict_contrib_test")

proto_library(
    name = "sampled_stats_proto",
    srcs = ["sampled_stats.proto"],
    deps = ["@org_tensorflow//tensorflow/core:protos_all"],
)

py_proto_library(
    name = "sampled_stats_py_proto",
    deps = [
        ":sampled_stats_proto",
    ],
)
