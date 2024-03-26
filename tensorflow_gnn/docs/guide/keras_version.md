# Keras Version Configuration for TF-GNN

The TensorFlow GNN library requires `tf.keras` to be Keras v2, because Keras v3
does not support composite tensor types like `tfgnn.GraphTensor`. Up to TF 2.15,
Keras v2 was the default for `tf.keras`. For TF 2.16+, you need to make special
arrangements, as described in this guide.

<!-- PLACEHOLDER FOR KERAS VERSION GOOGLE EXTRAS -->

## Installation and program execution

### For TensorFlow 2.16 and up

TensorFlow as of release 2.16 depends on `keras>=3` but no longer on a package
for Keras v2. Install TF-GNN together with the package `tf-keras` that continues
to supply an implementation of Keras v2.

```
pip install tensorflow-gnn tf-keras
```

Running a TF-GNN program under TF 2.16+ requires to set the
[environment variable](https://en.wikipedia.org/wiki/Environment_variable)
`TF_USE_LEGACY_KERAS` to `1` one way or another, for example:

  * with the Unix shell command

    ```
    export TF_USE_LEGACY_KERAS=1
    ```

    or your system's equivalent;

  * at the top of the main Python program or Colab notebook by

    ```python
    import os
    os.environ["TF_USE_LEGACY_KERAS"] = "1"
    ```

    **before** any part of the program gets to `import tensorflow`.


### For TensorFlow 2.15

Nothing special is required: TF 2.15 depends on `keras==2.15.*` and, by default,
defines `tf.keras` with it.

In case `TF_USE_LEGACY_KERAS` is set to `1` and `tf-keras` is installed, then
`tf.keras` is defined in terms of that package. While it offers the same API,
it is a separate package with separate static registries and class hierarchies,
so it does not mix well with user code that uses `import keras` and objects
from `keras.*`. Hence we recommend to not use `keras.*` in user code.

### For TensorFlow 2.12, 2.13 and 2.14

Nothing special is required: These TF 2.x versions depend on the matching
version of Keras 2.x and define `tf.keras` with it. The environment
variable `TF_USE_LEGACY_KERAS` is ignored.


## Writing compatible code

### For all supported TensorFlow versions

For use of TF-GNN under any supported version of TensorFlow, we recommend that
user code does `import tensorflow as tf` and uses the Keras API at `tf.keras.*`.
The installation instructions above make sure this is Keras v2, as required
by the TF-GNN library itself.

Do not import and use `keras.*` directly: it will break for TF 2.16 and above.

### For TF 2.16+ only

As of TF 2.16+, `import tensorflow_gnn as tfgnn` checks that `tf.keras` is
Keras v2, which implies that it comes from the `tf_keras` package.

Users who wish to emphasize the use of Keras v2 at the expense of breaking
compatibility with older versions of TensorFlow can use `tf_keras` as a
syntactic alternative to `tf.keras`.


## Other libraries from the TensorFlow ecosystem

Keras does not support mixing different versions or packages of it.
Using TF-GNN in combination with other TensorFlow add-on libraries requires
all of them to work with the same Keras package, and it must be Keras v2.