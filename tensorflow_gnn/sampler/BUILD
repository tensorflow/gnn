package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

load("@rules_proto//proto:defs.bzl", "proto_library")
load("@rules_python//python:proto.bzl", "py_proto_library")

proto_library(
    name = "sampling_spec_proto",
    srcs = ["sampling_spec.proto"],
    deps = ["@org_tensorflow//tensorflow/core:protos_all"],
)

py_proto_library(
    name = "sampling_spec_py_proto",
    deps = [
        ":sampling_spec_proto",
    ],
)