<!-- lint-g3mark -->

# runner.GraphTensorPadding

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/runner/interfaces.py#L84-L95">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Collects `GraphtTensor` padding helpers.

<!-- Placeholder for "Used in" -->

## Methods

<h3 id="get_filter_fn"><code>get_filter_fn</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/runner/interfaces.py#L87-L91">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
<code>get_filter_fn(
    size_constraints: SizeConstraints
) -> Callable[..., bool]
</code></pre>

<h3 id="get_size_constraints"><code>get_size_constraints</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/runner/interfaces.py#L93-L95">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
<code>get_size_constraints(
    target_batch_size: int
) -> SizeConstraints
</code></pre>
