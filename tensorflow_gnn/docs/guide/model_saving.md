# Model Saving Guide

## Introduction

TF-GNN lets you express your GNN model in TensorFlow and Keras. The
[Input guide](./input_pipeline.md) has already introduced the basic workflow of
exporting a model for inference that combines feature preprocessing (such as
vocab lookups) with the trained GNN model, so that the complete serving models
maps a Tensor with a batch of strings (serialized tf.Example protos, as
recommended for TF Serving) to one or more output Tensors. For most users, that
workflow – or its implementation by the TF-GNN runner – should be sufficient to
"just make it work".

This in-depth guide for advanced users examines the variety of model saving
methods offered by Keras and TensorFlow in greater detail, and explains which
ones are recommended for TF-GNN.

Before we go any deeper, recall that TF-GNN requires Keras v2 (the one that has
traditionally been included with TensorFlow 2). Multi-backend Keras v3 does not
work with TF-GNN.

## Overview

Model saving is an umbrella term that covers two different use-cases:

1.  **[Fully supported] Model export for inference.**

    After a Python script has defined and trained the model, it gets exported to
    a downstream inference environment like TF Serving, TFLite or custom C++
    code. Virtually every real-world application of GNNs needs this for
    deployment. Below we show how to export a TF-GNN model as a TensorFlow
    [SavedModel](https://www.tensorflow.org/guide/saved_model) with a
    serving signature. This is **recommended and fully supported**.

2.  **[Not recommended] Model saving for reuse.**

    The trained model is saved from one and loaded back into another
    Python/TF/Keras program, as part of more complex model-building workflows.
    Below we explain how TF-GNN relates to the various methods offered by
    TF/Keras. In a nutshell:

    1.  Saving as a
        [Reusable SavedModel](https://www.tensorflow.org/hub/reusable_saved_models)
        and restoring as traced tf.functions:
        **experimental use only (no guarantees)**.
    2.  [Keras model saving with
        `save_format="tf"`](https://github.com/keras-team/keras-io/blob/3f25251089ceda61f6aec77d3c278d72d4eb1597/guides/ipynb/serialization_and_saving.ipynb)
        (the traditional default for Keras v2) and restoring as a Keras model
         from the saved layer configs:
        **partial support** on a case-by-case basis.
    3.  [Keras model saving with
        `save_format="keras"`](https://keras.io/guides/serialization_and_saving/)
        (a.k.a. `"keras_v3"`):
        **unsupported and broken**, due to API changes from `save_format="tf"`.
    
    As an alternative, there is always the option to re-build the model from the
    original code and restore its weights from a training checkpoint.

## Export to SavedModel for inference

### Basics

Exporting a `tf.keras.Model` for inference requires that the model accepts
batched inputs and returns batched outputs **represented by one or more
`tf.Tensor` values**, because the calling signature of the SavedModel defines
both the inputs and the outputs in a language-independent way as a mapping from
`str` keys to `tf.Tensor` values. Python-only abstractions like
`tfgnn.GraphTensor` are not available at this level, even though they can be
used liberally to construct the model in between inputs and outputs.

The usual input format for models deployed to TF Serving or bulk inference is a
single `tf.Tensor` of shape `[batch_size]` and dtype `tf.string` that contains a
batch of serialized `tf.Example` protos. These are readily converted to
`tfgnn.GraphTensor` using `tfgnn.keras.layers.ParseExample`. TFLite and custom
C++ environments can instead use a collection of numeric `tf.Tensor` values and
hand-construct a `tfgnn.GraphTensor` from them.

The model’s output format depends on the problem it solves. For example, root
node classification on a batch of sampled subgraphs typically returns a logits
tensor of shape `[batch_size, num_classes]`.

### Using `Model.export()` (requires TF >=2.13)

Saving:

```
inputs = tf.keras.layers.Input(shape=[],  # The batch dim is implied.
                               dtype=tf.string, name="examples")
graph = tfgnn.keras.layers.ParseExample(...)(inputs)
logits = ...  # Computed from `graph` by feature processing and a GNN.
outputs = {"logits": logits}
model = tf.keras.Model(inputs, outputs)
model.export("/tmp/my_saved_model")  # Requires TF2.13+.
```

Loading (in Python, for demonstration):

```
# This program does **not** import tensorflow_gnn.
restored_model = tf.saved_model.load("/tmp/my_saved_model")
signature_fn = restored_model.signatures[
    tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
input_dict = {"examples": ...}
output_dict = signature_fn(**input_dict)
logits = output_dict["logits"]
```

Notice how the keys of `input_dict` are defined by the names of the respective
`tf.keras.Input` objects (or, more precisely, the `tf.TypeSpec`s they contain).
For the keys of `output_dict`, `Model.export()` uses the same keys as the model
output, provided it is a dict. (If it were a list, you’d get meaningless
synthetic names, so it’s better to avoid that.) The name of the signature is
fixed. If you need multiple signatures or more control over other aspects,
consider moving from `Model.export()` to Keras’ underlying `ExportArchive` API.

Exporting a TF-GNN model like this maps all Python-level abstractions to
TensorFlow ops. TF-GNN does not define custom ops for GNN modeling. Thus, the
resulting SavedModel can be loaded into any Python TensorFlow program, without
having to import the TF-GNN library. (This matters for deployment pipelines that
include Python programs downstream from the trainer script.) The SavedModel is
covered by TensorFlow’s
[stability guarantees](https://www.tensorflow.org/guide/versions#compatibility_of_savedmodels_graphs_and_checkpoints).

This independence from TF-GNN is not trivial: The older `Model.save()` API can
introduce traced tf.functions for individual layers and thereby expose TF-GNN
types, as discussed in the next section. These are usually ignored when loading
into a non-Python environment but cause `tf.saved_model.load()` to fail if
TF-GNN was not imported beforehand. In TF 2.12, an early implementation of
`Model.export()` had the same issues.

### Using `runner.export_model()` (required for TF 2.12)

Instead of calling `Model.export()` directly, you can do

```
from tensorflow_gnn import runner
runner.export_model(model, "/tmp/my_saved_model")
```

The two main differences are:

*   `runner.export_model()` does not require TF2.13+.
*   `runner.export_model()` emulates the naming convention for signature outputs
    of the older `Model.save()` API: The exported model is free to output any
    nest of `tf.Tensor`s. By default, each output gets its name from the final
    Layer that produced it (irrespective of any dict keys in the nest!).
    Explicit naming overrides are possible. This helps with collaborative naming
    in the runner’s multi-task training.

### Never: `tf.saved_model.save()` on a Keras model

Do not use `tf.saved_model.save()` on a Keras model built with TF-GNN: It
behaves much like the legacy `Model.save()` API but does not accept the extra
options required to make the resulting SavedModel independent of TF-GNN.

## Model saving for reuse in Python

### Reusable SavedModels with tf.functions

**Status: experimental use only (no guarantees), not recommended.**

TF-GNN’s composite tensor types, `tfgnn.GraphTensor` and its pieces, cannot be
represented in the language-independent SignatureDef of a SavedModel. However,
when a Python/TensorFlow program restores a SavedModel with

```
restored_model = tf.saved_model.load(filepath)
```

...this not only restores the full model and its signatures but also the
underlying graph of trackable objects, including any `tf.function`s that may
have been attached to them at saving time (which is beyond the scope of this
doc).

A `tf.function` can perfectly well use a `tfgnn.GraphTensor` or any of its
pieces as an input or output. However, for (de)serialization, their TypeSpec
classes must be registered. If loading a SavedModel fails with an error like

```
ValueError: No TypeSpec has been registered with name 'tensorflow_gnn.EdgeSetSpec'
```

...you need to `import tensorflow_gnn` beforehand to register the TypeSpecs.

Already at saving time, TensorFlow warns about this with a message like

```
.../tensorflow/python/saved_model/nested_structure_coder.py:458: UserWarning: Encoding a StructuredValue with type tensorflow_gnn.GraphTensorSpec; loading this StructuredValue will require that this type be imported and registered.
```

Even if you do `import tensorflow_gnn` before loading, subtle issues can arise
from a mismatch between the TF-GNN version used to save the model and the
version used to import it. As we have seen extremely low adoption rates for
workflows around reusing SavedModels, TF-GNN provides **no guarantees** for this
way of reloading a model and does **not recommend** it.

### Keras save\_format="tf"

**Status: partially supported, not recommended.**

In Keras v2, `model.save(save_format="tf", ...)` not only creates a SavedModel
for TensorFlow but also includes a Keras model configuration (essentially, a
JSON string with a textual representation of the nested Layer objects that make
up the model).

Typical code it looks like this:

```
model.save("/tmp/my_keras_model",  save_format="tf", save_traces=False,
           include_optimizer=False, ...)
restored_model = tf.keras.models.load_model("/tmp/my_keras_model")
```

Here, the `restored_model` is a fully-fledged Keras model, recreated Layer by
Layer, using the stored model config to supply the necessary arguments to Layer
initialization. For this to work, every Layer class used in the model must
support serialization through `Layer.get_config()` and be registered with Keras
for deserialization. (Usually, that happens automatically by importing the
Python module that defines it.)

Some TF-GNN Layer types support this, some do not. Please check the
documentation of the Layer types you use. Unless documented, consider it
unsupported. Even if supported, the serialization format may have changed
between versions of the TF-GNN library.

If deserialization of a Layer from its Keras model config fails, Keras attempts
to fall back to restoring the Layer’s call behavior from a tf.function traced
specifically for that Layer, using the machinery discussed in the previous
section, which requires extra care to get right. We recommend to to pass
`save_traces=False` at saving time and avoid these additional considerations.

In summary, we consider this an experimental feature with partial support from
some Layer types, but we do not recommend it and we do not make promises for
stability between TF-GNN versions.

If you need to restore pre-trained models for further model-building, a crude
but often effective alternative is to save a checkpoint, rebuild the model from
the original code, and restore the checkpoint.

### Keras save\_format="keras" (a.k.a. "keras\_v3")

**Status: unsupported and broken.**

Available as of Keras/TensorFlow 2.12, this way of saving and reloading Keras
models follows the same goals as the previous section, but in a pure Keras
format (disentangled from SavedModel) that also works with Keras v3.

Unfortunately, this complete reimplementation has changed the interface contract
for composite tensor types and for `Layer.get_config/from_config`, so, for the
most part, it currently does not work with TF-GNN.
