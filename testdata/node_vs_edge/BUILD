licenses(["notice"])

package(
    default_applicable_licenses = ["//tensorflow_gnn:license"],
    default_visibility = ["//visibility:public"],
)

filegroup(
    name = "node_vs_edge",
    srcs = [
        "edge_set_one_to_two.csv",
        "edge_set_two_to_two.csv",
        ":node_set_one.csv",
        ":node_set_two.csv",
    ],
    data = glob(["*.pbtxt"]),
)
