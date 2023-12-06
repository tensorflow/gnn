# runner.DatasetProvider

<!-- Insert buttons and diff -->

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/runner/interfaces.py#L76-L81">
<img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source
on GitHub </a>

Helper class that provides a standard way to create an ABC using inheritance.

<!-- Placeholder for "Used in" -->

## Methods

<h3 id="get_dataset"><code>get_dataset</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/runner/interfaces.py#L78-L81">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
<code>get_dataset(
    context: tf.distribute.InputContext
) -> tf.data.Dataset
</code></pre>

Get a `tf.data.Dataset` by `context` per replica.
