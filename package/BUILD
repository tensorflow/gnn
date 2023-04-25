licenses(["notice"])  # Apache 2.0

sh_binary(
    name = "move_generated_files",
    srcs = ["move_generated_files.sh"],
    data = [
        "//tensorflow_gnn/experimental/sampler:eval_dag_pb2.py",
        "//tensorflow_gnn/proto:examples_pb2.py",
        "//tensorflow_gnn/proto:graph_schema_pb2.py",
        "//tensorflow_gnn/sampler:sampling_spec_pb2.py",
        "//tensorflow_gnn/sampler:subgraph_pb2.py",
        "//tensorflow_gnn/tools:sampled_stats_pb2.py",
    ],
)
