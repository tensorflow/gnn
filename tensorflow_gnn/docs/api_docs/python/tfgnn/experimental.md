# Module: tfgnn.experimental

<!-- Insert buttons and diff -->

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/experimental/__init__.py">
<img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source
on GitHub </a>

Experimental (unstable) parts of the public interface of TensorFlow GNN.

A symbol `foo` exposed here is available to library users as

```
import tensorflow_gnn as tfgnn

tfgnn.experimental.foo()
```

This is the preferred way to expose individual functions on track to inclusion
into the stable public interface of TensorFlow GNN.

Beyond these symbols, there are also experimental sub-libraries that need to be
imported separately (`from tensorflow_gnn.experimental import foo`). That is for
special cases only.
