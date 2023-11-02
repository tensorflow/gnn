<!-- lint-g3mark -->

# runner.TFDataServiceConfig

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/runner/orchestration.py#L68-L79">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Provides tf.data service related configuration options.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>runner.TFDataServiceConfig(
    tf_data_service_address: str,
    tf_data_service_job_name: str,
    tf_data_service_mode: Union[str, tf.data.experimental.service.ShardingPolicy]
)
</code></pre>

<!-- Placeholder for "Used in" -->

tf.data service has data flexible visitation guarantees, its impact over your
training pipelines will be empirical. Check out the tf.data service internals
and operation details from
<https://www.tensorflow.org/api_docs/python/tf/data/experimental/service>.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`tf_data_service_address`<a id="tf_data_service_address"></a>
</td>
<td>
Dataclass field
</td>
</tr><tr>
<td>
`tf_data_service_job_name`<a id="tf_data_service_job_name"></a>
</td>
<td>
Dataclass field
</td>
</tr><tr>
<td>
`tf_data_service_mode`<a id="tf_data_service_mode"></a>
</td>
<td>
Dataclass field
</td>
</tr>
</table>

## Methods

<h3 id="__eq__"><code>__eq__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__eq__(
    other
)
</code></pre>

Return self==value.
