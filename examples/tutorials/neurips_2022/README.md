# <p align="center"> Graph Neural Networks in TensorFlow: A Practical Guide </p>

## <p align="center">**NeurIPS'22 Tutorial**</p>
### <p align="center">Monday, November 28th</p>
<p align="center">9:30 a.m. - 12:25 p.m. CST</p>
<h3 align="center"> <a href=https://neurips.cc/Expo/Conferences/2022/workshop/63090>🎥 Schedule and Livestream Link 🎥</a></h3>

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

| Time (CST)  | Speaker            | Style    | Topic  |
| ----------: |:---------------:   | :-----:  | :----- |
| 9:30        | Bryan Perozzi      | slides   | **Introduction:** Welcome and introduction. |
| 9:40        | Sami Abu-El-Haija  | slides   | **Background:** Recap of Graph Neural Networks (GNNs) and problems they solve.  |
| 10:00       | Sami Abu-El-Haija  | hands-on | **Tutorial 1:** Running TF-GNN on one machine. |
| 10:20       | ---  | break    | *Break*  |
| 10:30       | Neslihan Bulut     | slides   | **Modeling Guide**: Building blocks and modeling guidelines for crafting advanced custom GNN architectures. |
| 10:50       | Neslihan Bulut     | hands-on | **Tutorial 2:** Implementing custom models, using TF-GNN’s modeling building blocks. |
| 11:20       | --- | break    | *Break*  |
| 11:30       | Brandon Mayer      | slides   | **Run at scale:** Challenges and solutions for running-at-scale |
| 11:50       | Brandon Mayer      | hands-on | **Tutorial 3:** Running TF-GNN at industry scale over large graphs. |

### Tutorial Material

Here's links to the material used in the presentation:

1.  [Presentation Slides (PDF)](https://drive.google.com/file/d/1ECcnRgJqjmj7hlegYuPdscLCUw7YJc7G/view?usp=sharing)
2.  Hands-on Tutorial Links
  1.  Tutorial 1: [TF-GNN Basics (Node Classification)](https://github.com/tensorflow/gnn/blob/main/examples/tutorials/neurips_2022/code_tutorial_1_tfgnn_single_machine.ipynb)
  2.  Tutorial 2: [TF-GNN Modeling (Graph Classification), Student version](https://github.com/tensorflow/gnn/blob/main/examples/tutorials/neurips_2022/neurips_student_tfgnn_graph_classification_mutag.ipynb)
      1. [Teacher version](https://github.com/tensorflow/gnn/blob/main/examples/tutorials/neurips_2022/neurips_teacher_tfgnn_graph_classification_mutag.ipynb)
  3.  Tutorial 3: Running TF-GNN at Scale

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
question on the [github repo](), or email us directly at `tensorflow-gnn AT
googlegroups DOT com`.
