# <p align="center"> Graph Neural Networks in TensorFlow: KDD'23 Tutorial </p>

## <p align="center">**KDD'23 Tutorial**</p>
### <p align="center">Sunday, August 6th, 2023</p>
<p align="center">9:00 a.m. - noon PST</p>

### Motivation

Graphs are general data structures that can represent information from a variety
of domains (social, biomedical, online transactions, and many more). Graph
Neural Networks (GNNs) are an exciting way to use graph structured data inside
neural network models that have recently exploded in popularity. However,
implementing GNNs and running GNNs on large (and complex) datasets still raises
a number of challenges for machine learning platforms.

#### Goals

Thanks for your interest in our tutorial! The main goal of this tutorial is to
help practitioners and researchers to implement GNNs in a TensorFlow setting.
Specifically, the tutorial will be mostly hands-on, and will walk the audience
through a process of running existing GNNs on heterogeneous graph data, and a
tour of how to implement new GNN models. The hands-on portion of the tutorial
will be based on [TF-GNN](https://github.com/tensorflow/gnn), a library for
working with graph structured data in TensorFlow.

### Schedule

| Time (PST)  | Speaker            | Style    | Topic  |
| ----------: |:---------------:   | :-----:  | :----- |
| 9:00 AM       | Bryan Perozzi      | slides   | **Introduction:** Welcome and overview of team's work |
| 9:07 PM     | Sami Abu-El-Haija  | slides   | **Background:** Recap of Graph Neural Networks (GNNs), problems they solve, and a brief intro to TF-GNN  |
| 9:25 AM     | Sami Abu-El-Haija  | hands-on | **Tutorial 1:** TF-GNN in a Nutshell. |
| 9:45 AM     | Break     | -   | **10 minutes break** |
| 9:55 AM     | Brandon Mayer     | slides   | **TF-GNN Internals**: TF-GNN pipelines, GraphTensor, Runner framework |
| 10:20 AM     | Sami Abu-El-Haija     | hands-on | **Tutorial 2:** Graph Tensor Programming (features, Algorithms, grad wrt A) -- optional: datasets. |
| 10:45 AM     | Break     | -   | **15 minutes break** |
| 11:00 AM     | Anton Tsitsulin     | slides | **Advanced Topics**: Advanced Capabilities and low code solutions |
| 11:20 AM     | Variety     | slides | **Research Spotlights** |


### Tutorial Material

1.  Hands-on Tutorial Links
  1.  Tutorial 1: [TF-GNN in a Nutshell](https://github.com/tensorflow/gnn/blob/main/examples/tutorials/kdd_2023/code_tutorial_1.ipynb)
  2.  Tutorial 2: [Graph Tensor Programming](https://github.com/tensorflow/gnn/blob/main/examples/tutorials/kdd_2023/code_tutorial_2.ipynb)
  3.  Tutorial 3: [In-memory Datasets](https://github.com/tensorflow/gnn/blob/main/examples/tutorials/kdd_2023/code_tutorial_datasets.ipynb)


### Additional Resources

If you're interested in learning more about GNNs or TF-GNN, we recommend the
following resources:

-   Our paper
    [*TF-GNN: Graph Neural Networks in TensorFlow*](https://arxiv.org/pdf/2207.03522.pdf),
    details the API design and background of the library.
-   The in-depth notebook
    [*OGBN-MAG end-to-end with TF-GNN*](https://github.com/tensorflow/gnn/blob/main/examples/notebooks/ogbn_mag_e2e.ipynb)
    offers a deep dive on building heterogeneous graph models using TF-GNN.

If you're interested in other work done by the
[Graph Mining team](https://research.google/teams/graph-mining/) at Google, we
have two 'background' workshops with recorded video that cover our work:

1.  NeurIPS'20 Workshop -- https://gm-neurips-2020.github.io/
2.  ICML'22 Workshop -- https://icml.cc/Expo/Conferences/2022/talkpanel/20446

### Contact us

If you have any questions about using TF-GNN, please feel free to drop a
question on the [github repo](https://github.com/tensorflow/gnn/).

