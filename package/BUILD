licenses(["notice"])  # Apache 2.0

config_setting(
    name = "windows",
    constraint_values = [
        "@bazel_tools//platforms:windows",
    ],
)

sh_binary(
    name = "move_generated_files",
    srcs = ["move_generated_files.sh"],
    data = [
        "//tensorflow_gnn/proto:examples_py_proto",
        "//tensorflow_gnn/proto:graph_schema_py_proto",
        "//tensorflow_gnn/sampler:subgraph_py_proto",
        # "//tensorflow_gnn/sampler:sampling_spec_py_proto",
        "//tensorflow_gnn/tools:sampled_stats_py_proto",
    ],
)
