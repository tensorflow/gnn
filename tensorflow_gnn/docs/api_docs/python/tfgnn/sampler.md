# Module: tfgnn.sampler

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/sampler/__init__.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Public interface for GNN Sampler.

## Classes

[`class SamplingOp`](../tfgnn/sampler/SamplingOp.md): A ProtocolMessage

[`class SamplingSpec`](../tfgnn/sampler/SamplingSpec.md): A ProtocolMessage

[`class SamplingSpecBuilder`](../tfgnn/sampler/SamplingSpecBuilder.md): Mimics
builder pattern that eases creation of `tfgnn.SamplingSpec`.

## Functions

[`make_sampling_spec_tree(...)`](../tfgnn/sampler/make_sampling_spec_tree.md):
Automatically creates `SamplingSpec` by starting from seed node set.
