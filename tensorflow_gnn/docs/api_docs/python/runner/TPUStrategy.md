<!-- lint-g3mark -->

# runner.TPUStrategy

[TOC]

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/runner/utils/strategies.py#L36-L43">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

A `TPUStrategy` convenience wrapper.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>runner.TPUStrategy(
    tpu: str = &#x27;&#x27;
)
</code></pre>

<!-- Placeholder for "Used in" -->

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`tpu_cluster_resolver`<a id="tpu_cluster_resolver"></a>
</td>
<td>
A
`tf.distribute.cluster_resolver.TPUClusterResolver` instance, which
provides information about the TPU cluster. If None, it will assume
running on a local TPU worker.
</td>
</tr><tr>
<td>
`experimental_device_assignment`<a id="experimental_device_assignment"></a>
</td>
<td>
Optional
`tf.tpu.experimental.DeviceAssignment` to specify the placement of
replicas on the TPU cluster.
</td>
</tr><tr>
<td>
`experimental_spmd_xla_partitioning`<a id="experimental_spmd_xla_partitioning"></a>
</td>
<td>
If True, enable the SPMD (Single
Program Multiple Data) mode in XLA compiler. This flag only affects the
performance of XLA compilation and the HBM requirement of the compiled
TPU program. Ceveat: if this flag is True, calling
`tf.distribute.TPUStrategy.experimental_assign_to_logical_device` will
result in a ValueError.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`cluster_resolver`<a id="cluster_resolver"></a>
</td>
<td>
Returns the cluster resolver associated with this strategy.

`tf.distribute.TPUStrategy` provides the associated
`tf.distribute.cluster_resolver.ClusterResolver`. If the user provides one in
`__init__`, that instance is returned; if the user does not, a default
`tf.distribute.cluster_resolver.TPUClusterResolver` is provided.

</td>
</tr><tr>
<td>
`extended`<a id="extended"></a>
</td>
<td>
`tf.distribute.StrategyExtended` with additional methods.
</td>
</tr><tr>
<td>
`num_replicas_in_sync`<a id="num_replicas_in_sync"></a>
</td>
<td>
Returns number of replicas over which gradients are aggregated.
</td>
</tr>
</table>

## Methods

<h3 id="distribute_datasets_from_function"><code>distribute_datasets_from_function</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>distribute_datasets_from_function(
    dataset_fn, options=None
)
</code></pre>

Distributes `tf.data.Dataset` instances created by calls to `dataset_fn`.

The argument `dataset_fn` that users pass in is an input function that has a
`tf.distribute.InputContext` argument and returns a `tf.data.Dataset` instance.
It is expected that the returned dataset from `dataset_fn` is already batched by
per-replica batch size (i.e. global batch size divided by the number of replicas
in sync) and sharded. `tf.distribute.Strategy.distribute_datasets_from_function`
does not batch or shard the `tf.data.Dataset` instance returned from the input
function. `dataset_fn` will be called on the CPU device of each of the workers
and each generates a dataset where every replica on that worker will dequeue one
batch of inputs (i.e. if a worker has two replicas, two batches will be dequeued
from the `Dataset` every step).

This method can be used for several purposes. First, it allows you to specify
your own batching and sharding logic. (In contrast,
`tf.distribute.experimental_distribute_dataset` does batching and sharding for
you.) For example, where `experimental_distribute_dataset` is unable to shard
the input files, this method might be used to manually shard the dataset
(avoiding the slow fallback behavior in `experimental_distribute_dataset`). In
cases where the dataset is infinite, this sharding can be done by creating
dataset replicas that differ only in their random seed.

The `dataset_fn` should take an `tf.distribute.InputContext` instance where
information about batching and input replication can be accessed.

You can use `element_spec` property of the `tf.distribute.DistributedDataset`
returned by this API to query the `tf.TypeSpec` of the elements returned by the
iterator. This can be used to set the `input_signature` property of a
`tf.function`. Follow `tf.distribute.DistributedDataset.element_spec` to see an
example.

IMPORTANT: The `tf.data.Dataset` returned by `dataset_fn` should have a
per-replica batch size, unlike `experimental_distribute_dataset`, which uses the
global batch size. This may be computed using
`input_context.get_per_replica_batch_size`.

Note: If you are using TPUStrategy, the order in which the data is processed by
the workers when using `tf.distribute.Strategy.experimental_distribute_dataset`
or `tf.distribute.Strategy.distribute_datasets_from_function` is not guaranteed.
This is typically required if you are using `tf.distribute` to scale prediction.
You can however insert an index for each element in the batch and order outputs
accordingly. Refer to [this
snippet](https://www.tensorflow.org/tutorials/distribute/input#caveats) for an
example of how to order outputs.

Note: Stateful dataset transformations are currently not supported with
`tf.distribute.experimental_distribute_dataset` or
`tf.distribute.distribute_datasets_from_function`. Any stateful ops that the
dataset may have are currently ignored. For example, if your dataset has a
`map_fn` that uses `tf.random.uniform` to rotate an image, then you have a
dataset graph that depends on state (i.e the random seed) on the local machine
where the python process is being executed.

For a tutorial on more usage and properties of this method, refer to the
[tutorial on distributed
input](https://www.tensorflow.org/tutorials/distribute/input#tfdistributestrategyexperimental_distribute_datasets_from_function)).
If you are interested in last partial batch handling, read [this
section](https://www.tensorflow.org/tutorials/distribute/input#partial_batches).

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`dataset_fn`
</td>
<td>
A function taking a `tf.distribute.InputContext` instance and
returning a `tf.data.Dataset`.
</td>
</tr><tr>
<td>
`options`
</td>
<td>
`tf.distribute.InputOptions` used to control options on how this
dataset is distributed.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A `tf.distribute.DistributedDataset`.
</td>
</tr>

</table>

<h3 id="experimental_assign_to_logical_device"><code>experimental_assign_to_logical_device</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>experimental_assign_to_logical_device(
    tensor, logical_device_id
)
</code></pre>

Adds annotation that `tensor` will be assigned to a logical device.

This adds an annotation to `tensor` specifying that operations on `tensor` will
be invoked on logical core device id `logical_device_id`. When model parallelism
is used, the default behavior is that all ops are placed on zero-th logical
device.

``` python

# Initializing TPU system with 2 logical devices and 4 replicas.
resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
tf.config.experimental_connect_to_cluster(resolver)
topology = tf.tpu.experimental.initialize_tpu_system(resolver)
device_assignment = tf.tpu.experimental.DeviceAssignment.build(
    topology,
    computation_shape=[1, 1, 1, 2],
    num_replicas=4)
strategy = tf.distribute.TPUStrategy(
    resolver, experimental_device_assignment=device_assignment)
iterator = iter(inputs)

@tf.function()
def step_fn(inputs):
  output = tf.add(inputs, inputs)

  # Add operation will be executed on logical device 0.
  output = strategy.experimental_assign_to_logical_device(output, 0)
  return output

strategy.run(step_fn, args=(next(iterator),))
```

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`tensor`
</td>
<td>
Input tensor to annotate.
</td>
</tr><tr>
<td>
`logical_device_id`
</td>
<td>
Id of the logical core to which the tensor will be
assigned.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
The logical device id presented is not consistent with total
number of partitions specified by the device assignment or the TPUStrategy
is constructed with `experimental_spmd_xla_partitioning=True`.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
Annotated tensor with identical value as `tensor`.
</td>
</tr>

</table>

<h3 id="experimental_distribute_dataset"><code>experimental_distribute_dataset</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>experimental_distribute_dataset(
    dataset, options=None
)
</code></pre>

Creates `tf.distribute.DistributedDataset` from `tf.data.Dataset`.

The returned `tf.distribute.DistributedDataset` can be iterated over similar to
regular datasets. NOTE: The user cannot add any more transformations to a
`tf.distribute.DistributedDataset`. You can only create an iterator or examine
the `tf.TypeSpec` of the data generated by it. See API docs of
`tf.distribute.DistributedDataset` to learn more.

The following is an example:

    >>> global_batch_size = 2
    >>> # Passing the devices is optional.
    ... strategy = tf.distribute.MirroredStrategy(devices=["GPU:0", "GPU:1"])
    >>> # Create a dataset
    ... dataset = tf.data.Dataset.range(4).batch(global_batch_size)
    >>> # Distribute that dataset
    ... dist_dataset = strategy.experimental_distribute_dataset(dataset)
    >>> @tf.function
    ... def replica_fn(input):
    ...   return input*2
    >>> result = []
    >>> # Iterate over the `tf.distribute.DistributedDataset`
    ... for x in dist_dataset:
    ...   # process dataset elements
    ...   result.append(strategy.run(replica_fn, args=(x,)))
    >>> print(result)
    [PerReplica:{
      0: <tf.Tensor: shape=(1,), dtype=int64, numpy=array([0])>,
      1: <tf.Tensor: shape=(1,), dtype=int64, numpy=array([2])>
    }, PerReplica:{
      0: <tf.Tensor: shape=(1,), dtype=int64, numpy=array([4])>,
      1: <tf.Tensor: shape=(1,), dtype=int64, numpy=array([6])>
    }]

Three key actions happening under the hood of this method are batching,
sharding, and prefetching.

In the code snippet above, `dataset` is batched by `global_batch_size`, and
calling `experimental_distribute_dataset` on it rebatches `dataset` to a new
batch size that is equal to the global batch size divided by the number of
replicas in sync. We iterate through it using a Pythonic for loop. `x` is a
`tf.distribute.DistributedValues` containing data for all replicas, and each
replica gets data of the new batch size. `tf.distribute.Strategy.run` will take
care of feeding the right per-replica data in `x` to the right `replica_fn`
executed on each replica.

Sharding contains autosharding across multiple workers and within every worker.
First, in multi-worker distributed training (i.e. when you use
`tf.distribute.experimental.MultiWorkerMirroredStrategy` or
`tf.distribute.TPUStrategy`), autosharding a dataset over a set of workers means
that each worker is assigned a subset of the entire dataset (if the right
`tf.data.experimental.AutoShardPolicy` is set). This is to ensure that at each
step, a global batch size of non-overlapping dataset elements will be processed
by each worker. Autosharding has a couple of different options that can be
specified using `tf.data.experimental.DistributeOptions`. Then, sharding within
each worker means the method will split the data among all the worker devices
(if more than one a present). This will happen regardless of multi-worker
autosharding.

Note: for autosharding across multiple workers, the default mode is
`tf.data.experimental.AutoShardPolicy.AUTO`. This mode will attempt to shard the
input dataset by files if the dataset is being created out of reader datasets
(e.g. `tf.data.TFRecordDataset`, `tf.data.TextLineDataset`, etc.) or otherwise
shard the dataset by data, where each of the workers will read the entire
dataset and only process the shard assigned to it. However, if you have less
than one input file per worker, we suggest that you disable dataset autosharding
across workers by setting the
`tf.data.experimental.DistributeOptions.auto_shard_policy` to be
`tf.data.experimental.AutoShardPolicy.OFF`.

By default, this method adds a prefetch transformation at the end of the user
provided `tf.data.Dataset` instance. The argument to the prefetch transformation
which is `buffer_size` is equal to the number of replicas in sync.

If the above batch splitting and dataset sharding logic is undesirable, please
use `tf.distribute.Strategy.distribute_datasets_from_function` instead, which
does not do any automatic batching or sharding for you.

Note: If you are using TPUStrategy, the order in which the data is processed by
the workers when using `tf.distribute.Strategy.experimental_distribute_dataset`
or `tf.distribute.Strategy.distribute_datasets_from_function` is not guaranteed.
This is typically required if you are using `tf.distribute` to scale prediction.
You can however insert an index for each element in the batch and order outputs
accordingly. Refer to [this
snippet](https://www.tensorflow.org/tutorials/distribute/input#caveats) for an
example of how to order outputs.

Note: Stateful dataset transformations are currently not supported with
`tf.distribute.experimental_distribute_dataset` or
`tf.distribute.distribute_datasets_from_function`. Any stateful ops that the
dataset may have are currently ignored. For example, if your dataset has a
`map_fn` that uses `tf.random.uniform` to rotate an image, then you have a
dataset graph that depends on state (i.e the random seed) on the local machine
where the python process is being executed.

For a tutorial on more usage and properties of this method, refer to the
[tutorial on distributed
input](https://www.tensorflow.org/tutorials/distribute/input#tfdistributestrategyexperimental_distribute_dataset).
If you are interested in last partial batch handling, read [this
section](https://www.tensorflow.org/tutorials/distribute/input#partial_batches).

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`dataset`
</td>
<td>
`tf.data.Dataset` that will be sharded across all replicas using
the rules stated above.
</td>
</tr><tr>
<td>
`options`
</td>
<td>
`tf.distribute.InputOptions` used to control options on how this
dataset is distributed.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A `tf.distribute.DistributedDataset`.
</td>
</tr>

</table>

<h3 id="experimental_distribute_values_from_function"><code>experimental_distribute_values_from_function</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>experimental_distribute_values_from_function(
    value_fn
)
</code></pre>

Generates `tf.distribute.DistributedValues` from `value_fn`.

This function is to generate `tf.distribute.DistributedValues` to pass into
`run`, `reduce`, or other methods that take distributed values when not using
datasets.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`value_fn`
</td>
<td>
The function to run to generate values. It is called for
each replica with `tf.distribute.ValueContext` as the sole argument. It
must return a Tensor or a type that can be converted to a Tensor.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A `tf.distribute.DistributedValues` containing a value for each replica.
</td>
</tr>

</table>

#### Example usage:

1.  Return constant value per replica:

        >>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
        >>> def value_fn(ctx):
        ...   return tf.constant(1.)
        >>> distributed_values = (
        ...     strategy.experimental_distribute_values_from_function(
        ...        value_fn))
        >>> local_result = strategy.experimental_local_results(
        ...     distributed_values)
        >>> local_result
        (<tf.Tensor: shape=(), dtype=float32, numpy=1.0>,
        <tf.Tensor: shape=(), dtype=float32, numpy=1.0>)

2.  Distribute values in array based on replica_id: {: value=2}

        >>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
        >>> array_value = np.array([3., 2., 1.])
        >>> def value_fn(ctx):
        ...   return array_value[ctx.replica_id_in_sync_group]
        >>> distributed_values = (
        ...     strategy.experimental_distribute_values_from_function(
        ...         value_fn))
        >>> local_result = strategy.experimental_local_results(
        ...     distributed_values)
        >>> local_result
        (3.0, 2.0)

3.  Specify values using num_replicas_in_sync: {: value=3}

        >>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
        >>> def value_fn(ctx):
        ...   return ctx.num_replicas_in_sync
        >>> distributed_values = (
        ...     strategy.experimental_distribute_values_from_function(
        ...         value_fn))
        >>> local_result = strategy.experimental_local_results(
        ...     distributed_values)
        >>> local_result
        (2, 2)

4.  Place values on devices and distribute: {: value=4}

        strategy = tf.distribute.TPUStrategy()
        worker_devices = strategy.extended.worker_devices
        multiple_values = []
        for i in range(strategy.num_replicas_in_sync):
          with tf.device(worker_devices[i]):
            multiple_values.append(tf.constant(1.0))

        def value_fn(ctx):
          return multiple_values[ctx.replica_id_in_sync_group]

        distributed_values = strategy.
          experimental_distribute_values_from_function(
          value_fn)

<h3 id="experimental_local_results"><code>experimental_local_results</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>experimental_local_results(
    value
)
</code></pre>

Returns the list of all local per-replica values contained in `value`.

Note: This only returns values on the worker initiated by this client. When
using a `tf.distribute.Strategy` like
`tf.distribute.experimental.MultiWorkerMirroredStrategy`, each worker will be
its own client, and this function will only return values computed on that
worker.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`value`
</td>
<td>
A value returned by `experimental_run()`, `run(), or a variable
created in `scope`.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A tuple of values contained in `value` where ith element corresponds to
ith replica. If `value` represents a single value, this returns
`(value,).`
</td>
</tr>

</table>

<h3 id="experimental_replicate_to_logical_devices"><code>experimental_replicate_to_logical_devices</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>experimental_replicate_to_logical_devices(
    tensor
)
</code></pre>

Adds annotation that `tensor` will be replicated to all logical devices.

This adds an annotation to tensor `tensor` specifying that operations on
`tensor` will be invoked on all logical devices.

``` python
# Initializing TPU system with 2 logical devices and 4 replicas.
resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
tf.config.experimental_connect_to_cluster(resolver)
topology = tf.tpu.experimental.initialize_tpu_system(resolver)
device_assignment = tf.tpu.experimental.DeviceAssignment.build(
    topology,
    computation_shape=[1, 1, 1, 2],
    num_replicas=4)
strategy = tf.distribute.TPUStrategy(
    resolver, experimental_device_assignment=device_assignment)

iterator = iter(inputs)

@tf.function()
def step_fn(inputs):
  images, labels = inputs
  images = strategy.experimental_split_to_logical_devices(
    inputs, [1, 2, 4, 1])

  # model() function will be executed on 8 logical devices with `inputs`
  # split 2 * 4  ways.
  output = model(inputs)

  # For loss calculation, all logical devices share the same logits
  # and labels.
  labels = strategy.experimental_replicate_to_logical_devices(labels)
  output = strategy.experimental_replicate_to_logical_devices(output)
  loss = loss_fn(labels, output)

  return loss

strategy.run(step_fn, args=(next(iterator),))
```

Args: tensor: Input tensor to annotate.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
Annotated tensor with identical value as `tensor`.
</td>
</tr>

</table>

<h3 id="experimental_split_to_logical_devices"><code>experimental_split_to_logical_devices</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>experimental_split_to_logical_devices(
    tensor, partition_dimensions
)
</code></pre>

Adds annotation that `tensor` will be split across logical devices.

This adds an annotation to tensor `tensor` specifying that operations on
`tensor` will be split among multiple logical devices. Tensor `tensor` will be
split across dimensions specified by `partition_dimensions`. The dimensions of
`tensor` must be divisible by corresponding value in `partition_dimensions`.

For example, for system with 8 logical devices, if `tensor` is an image tensor
with shape (batch_size, width, height, channel) and `partition_dimensions` is
\[1, 2, 4, 1\], then `tensor` will be split 2 in width dimension and 4 way in
height dimension and the split tensor values will be fed into 8 logical devices.

``` python
# Initializing TPU system with 8 logical devices and 1 replica.
resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
tf.config.experimental_connect_to_cluster(resolver)
topology = tf.tpu.experimental.initialize_tpu_system(resolver)
device_assignment = tf.tpu.experimental.DeviceAssignment.build(
    topology,
    computation_shape=[1, 2, 2, 2],
    num_replicas=1)
# Construct the TPUStrategy. Since we are going to split the image across
# logical devices, here we set `experimental_spmd_xla_partitioning=True`
# so that the partitioning can be compiled in SPMD mode, which usually
# results in faster compilation and smaller HBM requirement if the size of
# input and activation tensors are much bigger than that of the model
# parameters. Note that this flag is suggested but not a hard requirement
# for `experimental_split_to_logical_devices`.
strategy = tf.distribute.TPUStrategy(
    resolver, experimental_device_assignment=device_assignment,
    experimental_spmd_xla_partitioning=True)

iterator = iter(inputs)

@tf.function()
def step_fn(inputs):
  inputs = strategy.experimental_split_to_logical_devices(
    inputs, [1, 2, 4, 1])

  # model() function will be executed on 8 logical devices with `inputs`
  # split 2 * 4  ways.
  output = model(inputs)
  return output

strategy.run(step_fn, args=(next(iterator),))
```

Args: tensor: Input tensor to annotate. partition_dimensions: An unnested list
of integers with the size equal to rank of `tensor` specifying how `tensor` will
be partitioned. The product of all elements in `partition_dimensions` must be
equal to the total number of logical devices per replica.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
1) If the size of partition_dimensions does not equal to rank
of `tensor` or 2) if product of elements of `partition_dimensions` does
not match the number of logical devices per replica defined by the
implementing DistributionStrategy's device specification or
3) if a known size of `tensor` is not divisible by corresponding
value in `partition_dimensions`.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
Annotated tensor with identical value as `tensor`.
</td>
</tr>

</table>

<h3 id="gather"><code>gather</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>gather(
    value, axis
)
</code></pre>

Gather `value` across replicas along `axis` to the current device.

Given a `tf.distribute.DistributedValues` or `tf.Tensor`-like object `value`,
this API gathers and concatenates `value` across replicas along the `axis`-th
dimension. The result is copied to the "current" device, which would typically
be the CPU of the worker on which the program is running. For
`tf.distribute.TPUStrategy`, it is the first TPU host. For multi-client
`tf.distribute.MultiWorkerMirroredStrategy`, this is the CPU of each worker.

This API can only be called in the cross-replica context. For a counterpart in
the replica context, see `tf.distribute.ReplicaContext.all_gather`.

Note: For all strategies except `tf.distribute.TPUStrategy`, the input `value`
on different replicas must have the same rank, and their shapes must be the same
in all dimensions except the `axis`-th dimension. In other words, their shapes
cannot be different in a dimension `d` where `d` does not equal to the `axis`
argument. For example, given a `tf.distribute.DistributedValues` with component
tensors of shape `(1, 2, 3)` and `(1, 3, 3)` on two replicas, you can call
`gather(..., axis=1, ...)` on it, but not `gather(..., axis=0, ...)` or
`gather(..., axis=2, ...)`. However, for `tf.distribute.TPUStrategy.gather`, all
tensors must have exactly the same rank and same shape.

Note: Given a `tf.distribute.DistributedValues` `value`, its component tensors
must have a non-zero rank. Otherwise, consider using `tf.expand_dims` before
gathering them.

    >>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
    >>> # A DistributedValues with component tensor of shape (2, 1) on each replica
    ... distributed_values = strategy.experimental_distribute_values_from_function(lambda _: tf.identity(tf.constant([[1], [2]])))
    >>> @tf.function
    ... def run():
    ...   return strategy.gather(distributed_values, axis=0)
    >>> run()
    <tf.Tensor: shape=(4, 1), dtype=int32, numpy=
    array([[1],
           [2],
           [1],
           [2]], dtype=int32)>

Consider the following example for more combinations:

    >>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1", "GPU:2", "GPU:3"])
    >>> single_tensor = tf.reshape(tf.range(6), shape=(1,2,3))
    >>> distributed_values = strategy.experimental_distribute_values_from_function(lambda _: tf.identity(single_tensor))
    >>> @tf.function
    ... def run(axis):
    ...   return strategy.gather(distributed_values, axis=axis)
    >>> axis=0
    >>> run(axis)
    <tf.Tensor: shape=(4, 2, 3), dtype=int32, numpy=
    array([[[0, 1, 2],
            [3, 4, 5]],
           [[0, 1, 2],
            [3, 4, 5]],
           [[0, 1, 2],
            [3, 4, 5]],
           [[0, 1, 2],
            [3, 4, 5]]], dtype=int32)>
    >>> axis=1
    >>> run(axis)
    <tf.Tensor: shape=(1, 8, 3), dtype=int32, numpy=
    array([[[0, 1, 2],
            [3, 4, 5],
            [0, 1, 2],
            [3, 4, 5],
            [0, 1, 2],
            [3, 4, 5],
            [0, 1, 2],
            [3, 4, 5]]], dtype=int32)>
    >>> axis=2
    >>> run(axis)
    <tf.Tensor: shape=(1, 2, 12), dtype=int32, numpy=
    array([[[0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
            [3, 4, 5, 3, 4, 5, 3, 4, 5, 3, 4, 5]]], dtype=int32)>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`value`
</td>
<td>
a `tf.distribute.DistributedValues` instance, e.g. returned by
`Strategy.run`, to be combined into a single tensor. It can also be a
regular tensor when used with `tf.distribute.OneDeviceStrategy` or the
default strategy. The tensors that constitute the DistributedValues
can only be dense tensors with non-zero rank, NOT a `tf.IndexedSlices`.
</td>
</tr><tr>
<td>
`axis`
</td>
<td>
0-D int32 Tensor. Dimension along which to gather. Must be in the
range [0, rank(value)).
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A `Tensor` that's the concatenation of `value` across replicas along
`axis` dimension.
</td>
</tr>

</table>

<h3 id="reduce"><code>reduce</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>reduce(
    reduce_op, value, axis
)
</code></pre>

Reduce `value` across replicas and return result on current device.

    >>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
    >>> def step_fn():
    ...   i = tf.distribute.get_replica_context().replica_id_in_sync_group
    ...   return tf.identity(i)
    >>>
    >>> per_replica_result = strategy.run(step_fn)
    >>> total = strategy.reduce("SUM", per_replica_result, axis=None)
    >>> total
    <tf.Tensor: shape=(), dtype=int32, numpy=1>

To see how this would look with multiple replicas, consider the same example
with MirroredStrategy with 2 GPUs:

``` python
strategy = tf.distribute.MirroredStrategy(devices=["GPU:0", "GPU:1"])
def step_fn():
  i = tf.distribute.get_replica_context().replica_id_in_sync_group
  return tf.identity(i)

per_replica_result = strategy.run(step_fn)
# Check devices on which per replica result is:
strategy.experimental_local_results(per_replica_result)[0].device
# /job:localhost/replica:0/task:0/device:GPU:0
strategy.experimental_local_results(per_replica_result)[1].device
# /job:localhost/replica:0/task:0/device:GPU:1

total = strategy.reduce("SUM", per_replica_result, axis=None)
# Check device on which reduced result is:
total.device
# /job:localhost/replica:0/task:0/device:CPU:0

```

This API is typically used for aggregating the results returned from different
replicas, for reporting etc. For example, loss computed from different replicas
can be averaged using this API before printing.

Note: The result is copied to the "current" device - which would typically be
the CPU of the worker on which the program is running. For `TPUStrategy`, it is
the first TPU host. For multi client `MultiWorkerMirroredStrategy`, this is CPU
of each worker.

There are a number of different tf.distribute APIs for reducing values across
replicas:

  - `tf.distribute.ReplicaContext.all_reduce`: This differs from
    `Strategy.reduce` in that it is for replica context and does not copy the
    results to the host device. `all_reduce` should be typically used for
    reductions inside the training step such as gradients.
  - `tf.distribute.StrategyExtended.reduce_to` and
    `tf.distribute.StrategyExtended.batch_reduce_to`: These APIs are more
    advanced versions of `Strategy.reduce` as they allow customizing the
    destination of the result. They are also called in cross replica context.

*What should axis be?*

Given a per-replica value returned by `run`, say a per-example loss, the batch
will be divided across all the replicas. This function allows you to aggregate
across replicas and optionally also across batch elements by specifying the axis
parameter accordingly.

For example, if you have a global batch size of 8 and 2 replicas, values for
examples `[0, 1, 2, 3]` will be on replica 0 and `[4, 5, 6, 7]` will be on
replica 1. With `axis=None`, `reduce` will aggregate only across replicas,
returning `[0+4, 1+5, 2+6, 3+7]`. This is useful when each replica is computing
a scalar or some other value that doesn't have a "batch" dimension (like a
gradient or loss).

    strategy.reduce("sum", per_replica_result, axis=None)

Sometimes, you will want to aggregate across both the global batch *and* all
replicas. You can get this behavior by specifying the batch dimension as the
`axis`, typically `axis=0`. In this case it would return a scalar
`0+1+2+3+4+5+6+7`.

    strategy.reduce("sum", per_replica_result, axis=0)

If there is a last partial batch, you will need to specify an axis so that the
resulting shape is consistent across replicas. So if the last batch has size 6
and it is divided into \[0, 1, 2, 3\] and \[4, 5\], you would get a shape
mismatch unless you specify `axis=0`. If you specify
`tf.distribute.ReduceOp.MEAN`, using `axis=0` will use the correct denominator
of 6. Contrast this with computing `reduce_mean` to get a scalar value on each
replica and this function to average those means, which will weigh some values
`1/8` and others `1/4`.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`reduce_op`
</td>
<td>
a `tf.distribute.ReduceOp` value specifying how values should
be combined. Allows using string representation of the enum such as
"SUM", "MEAN".
</td>
</tr><tr>
<td>
`value`
</td>
<td>
a `tf.distribute.DistributedValues` instance, e.g. returned by
`Strategy.run`, to be combined into a single tensor. It can also be a
regular tensor when used with `OneDeviceStrategy` or default strategy.
</td>
</tr><tr>
<td>
`axis`
</td>
<td>
specifies the dimension to reduce along within each
replica's tensor. Should typically be set to the batch dimension, or
`None` to only reduce across replicas (e.g. if the tensor has no batch
dimension).
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A `Tensor`.
</td>
</tr>

</table>

<h3 id="run"><code>run</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>run(
    fn, args=(), kwargs=None, options=None
)
</code></pre>

Run the computation defined by `fn` on each TPU replica.

Executes ops specified by `fn` on each replica. If `args` or `kwargs` have
`tf.distribute.DistributedValues`, such as those produced by a
`tf.distribute.DistributedDataset` from
`tf.distribute.Strategy.experimental_distribute_dataset` or
`tf.distribute.Strategy.distribute_datasets_from_function`, when `fn` is
executed on a particular replica, it will be executed with the component of
`tf.distribute.DistributedValues` that correspond to that replica.

`fn` may call `tf.distribute.get_replica_context()` to access members such as
`all_reduce`.

All arguments in `args` or `kwargs` should either be nest of tensors or
`tf.distribute.DistributedValues` containing tensors or composite tensors.

#### Example usage:

    >>> resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
    >>> tf.config.experimental_connect_to_cluster(resolver)
    >>> tf.tpu.experimental.initialize_tpu_system(resolver)
    >>> strategy = tf.distribute.TPUStrategy(resolver)
    >>> @tf.function
    ... def run():
    ...   def value_fn(value_context):
    ...     return value_context.num_replicas_in_sync
    ...   distributed_values = (
    ...       strategy.experimental_distribute_values_from_function(value_fn))
    ...   def replica_fn(input):
    ...     return input * 2
    ...   return strategy.run(replica_fn, args=(distributed_values,))
    >>> result = run()

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`fn`
</td>
<td>
The function to run. The output must be a `tf.nest` of `Tensor`s.
</td>
</tr><tr>
<td>
`args`
</td>
<td>
(Optional) Positional arguments to `fn`.
</td>
</tr><tr>
<td>
`kwargs`
</td>
<td>
(Optional) Keyword arguments to `fn`.
</td>
</tr><tr>
<td>
`options`
</td>
<td>
(Optional) An instance of `tf.distribute.RunOptions` specifying
the options to run `fn`.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
Merged return value of `fn` across replicas. The structure of the return
value is the same as the return value from `fn`. Each element in the
structure can either be `tf.distribute.DistributedValues`, `Tensor`
objects, or `Tensor`s (for example, if running on a single replica).
</td>
</tr>

</table>

<h3 id="scope"><code>scope</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>scope()
</code></pre>

Context manager to make the strategy current and distribute variables.

This method returns a context manager, and is used as follows:

    >>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
    >>> # Variable created inside scope:
    >>> with strategy.scope():
    ...   mirrored_variable = tf.Variable(1.)
    >>> mirrored_variable
    MirroredVariable:{
      0: <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=1.0>,
      1: <tf.Variable 'Variable/replica_1:0' shape=() dtype=float32, numpy=1.0>
    }
    >>> # Variable created outside scope:
    >>> regular_variable = tf.Variable(1.)
    >>> regular_variable
    <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=1.0>

*What happens when Strategy.scope is entered?*

  - `strategy` is installed in the global context as the "current" strategy.
    Inside this scope, `tf.distribute.get_strategy()` will now return this
    strategy. Outside this scope, it returns the default no-op strategy.
  - Entering the scope also enters the "cross-replica context". See
    `tf.distribute.StrategyExtended` for an explanation on cross-replica and
    replica contexts.
  - Variable creation inside `scope` is intercepted by the strategy. Each
    strategy defines how it wants to affect the variable creation. Sync
    strategies like `MirroredStrategy`, `TPUStrategy` and
    `MultiWorkerMiroredStrategy` create variables replicated on each replica,
    whereas `ParameterServerStrategy` creates variables on the parameter
    servers. This is done using a custom `tf.variable_creator_scope`.
  - In some strategies, a default device scope may also be entered: in
    `MultiWorkerMiroredStrategy`, a default device scope of "/CPU:0" is entered
    on each worker.

Note: Entering a scope does not automatically distribute a computation, except
in the case of high level training framework like keras `model.fit`. If you're
not using `model.fit`, you need to use `strategy.run` API to explicitly
distribute that computation. See an example in the [custom training loop
tutorial](https://www.tensorflow.org/tutorials/distribute/custom_training).

*What should be in scope and what should be outside?*

There are a number of requirements on what needs to happen inside the scope.
However, in places where we have information about which strategy is in use, we
often enter the scope for the user, so they don't have to do it explicitly (i.e.
calling those either inside or outside the scope is OK).

  - Anything that creates variables that should be distributed variables must be
    called in a `strategy.scope`. This can be accomplished either by directly
    calling the variable creating function within the scope context, or by
    relying on another API like `strategy.run` or `keras.Model.fit` to
    automatically enter it for you. Any variable that is created outside scope
    will not be distributed and may have performance implications. Some common
    objects that create variables in TF are Models, Optimizers, Metrics. Such
    objects should always be initialized in the scope, and any functions that
    may lazily create variables (e.g., `Model.__call__()`, tracing a
    `tf.function`, etc.) should similarly be called within scope. Another source
    of variable creation can be a checkpoint restore - when variables are
    created lazily. Note that any variable created inside a strategy captures
    the strategy information. So reading and writing to these variables outside
    the `strategy.scope` can also work seamlessly, without the user having to
    enter the scope.
  - Some strategy APIs (such as `strategy.run` and `strategy.reduce`) which
    require to be in a strategy's scope, enter the scope automatically, which
    means when using those APIs you don't need to explicitly enter the scope
    yourself.
  - When a `tf.keras.Model` is created inside a `strategy.scope`, the Model
    object captures the scope information. When high level training framework
    methods such as `model.compile`, `model.fit`, etc. are then called, the
    captured scope will be automatically entered, and the associated strategy
    will be used to distribute the training etc. See a detailed example in
    [distributed keras
    tutorial](https://www.tensorflow.org/tutorials/distribute/keras). WARNING:
    Simply calling `model(..)` does not automatically enter the captured scope
    -- only high level training framework APIs support this behavior:
    `model.compile`, `model.fit`, `model.evaluate`, `model.predict` and
    `model.save` can all be called inside or outside the scope.
  - The following can be either inside or outside the scope:
      - Creating the input datasets
      - Defining `tf.function`s that represent your training step
      - Saving APIs such as `tf.saved_model.save`. Loading creates variables, so
        that should go inside the scope if you want to train the model in a
        distributed way.
      - Checkpoint saving. As mentioned above - `checkpoint.restore` may
        sometimes need to be inside scope if it creates variables.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A context manager.
</td>
</tr>

</table>
