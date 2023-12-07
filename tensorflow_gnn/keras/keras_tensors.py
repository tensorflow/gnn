# Copyright 2021 The TensorFlow GNN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""KerasTensor specializations for GraphTensor pieces.

IMPORTANT: Some utilities below rely on TF's fallback dispatch system to
specialize callables for KerasTensor inputs. This system assumes that a
`ValueError` or `TypeError` is raised when the wrapped callables are called
with an unexpected argument type.
"""
import functools
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import tensorflow as tf
from tensorflow_gnn.graph import adjacency as adj
from tensorflow_gnn.graph import graph_constants as const
from tensorflow_gnn.graph import graph_piece as gp
from tensorflow_gnn.graph import graph_tensor as gt
from tensorflow_gnn.graph import tf_internal


class GraphPieceKerasTensor(tf_internal.KerasTensor):
  """Base class for graph pieces Keras tensors.

  Each graph piece (e.g. `tfgnn.Context`, `tfgnn.NodeSet`, etc.) must define
  corresponding Keras tensor and register them with the
  `register_keras_tensor_specialization`.

  Keras tensors specialization is done according to the rules:
  1. Graph piece methods and properties that return static values (e.g. python
     scalars) must be explicitly mirrored in the corresponding Keras tensors
     See, for example, `rank` and  `indices_dtype` below.

  2. Methods and properties that return dynamic values (e.g. tf.Tensor) must be
     delegated with the `_delegate_method` and `_delegate_property`.

  3. Class methods (e.g. class factory methods) of the original pieces must be
     decorated with `@tf.__internal__.dispatch.add_dispatch_support` and then
     delegated for Keras arguments using _GraphPieceClassMethodDispatcher (see
     class docstring for more details). NOTE: for delegation to work, class
     methods are assumed to raise `ValueError` or `TypeError` on unsupported
     arguments.
  """

  @property
  def rank(self):
    return self.shape.rank

  @property
  def indices_dtype(self):
    return self.spec.indices_dtype

  @property
  def spec(self):
    return self._type_spec


class GraphKerasTensor(GraphPieceKerasTensor):
  pass


class ContextKerasTensor(GraphPieceKerasTensor):
  pass


class NodeSetKerasTensor(GraphPieceKerasTensor):
  pass


class EdgeSetKerasTensor(GraphPieceKerasTensor):
  pass


class HyperAdjacencyKerasTensor(GraphPieceKerasTensor):

  def node_set_name(
      self, node_set_tag: const.IncidentNodeTag
  ) -> const.NodeSetName:
    return self.spec.node_set_name(node_set_tag)


class AdjacencyKerasTensor(HyperAdjacencyKerasTensor):

  @property
  def source_name(self) -> const.NodeSetName:
    return self.spec.source_name

  @property
  def target_name(self) -> const.NodeSetName:
    return self.spec.target_name


class _GraphPieceClassMethodDispatcher(tf_internal.OpDispatcher):
  """Dispatcher for `GraphPiece` class methods.

  Wraps a class method call as a `_GraphPieceClassMethod` Keras layer if any
  of the call inputs are Keras tensors. In particular, this allows to use graph
  piece factory methods, such as `GraphTensor.from_pieces(...)`, with the Keras
  Functional API.

  For this dispatcher to work, the target class methods of graph pieces must
  be decorated with `@tf.__internal__.dispatch.add_dispatch_support` and then
  registered with `register()` class method of this class.

  Wrapped class methods may not follow Keras Layer `call()` method convention
  (https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer#call), e.g.,
  positional arguments can be Python scalars which is not supported by Keras.
  To support generic cases, this class rearranges arguments of the dispatched
  `classmethod(*args, **kwargs)` into arguments for invoking
  `_GraphPieceClassMethod(args_index)(layer_args, **layer_kwargs)` as follows:

   1. Arguments of a tensor type (namely Tensor, composite Tensor, or
      KerasTensor wrappers of those) are put into the list `layer_args`,
      which is passed as a single positional argument.
   2. Arguments of all other types are put into the dict `layer_kwargs`
      under str keys `f"arg{index}"` with index 0, 1, 2, ..., which is
      expanded to keyword arguments for the layer call.

  The `args_index` is a nest with the same structure as the original
  `(*args, **kwargs)`, except that each value has been replaced by its
  int index into `layer_args` or its str key into `layer_kwargs`, respectively.
  """

  def __init__(self, cls, cls_method_name):
    # pylint: disable=protected-access
    if not issubclass(cls, gp.GraphPieceBase):
      raise ValueError(
          f'Expected that {cls.__name__} is subclass of `GraphPieceBase`'
      )

    self._cls = cls
    self._cls_method_name = cls_method_name

  def handle(self, args, kwargs):
    """Handle the specified operation with the specified arguments."""
    if not has_keras_tensors([args, kwargs]):
      return self.NOT_SUPPORTED

    # NOTE: dispatcher could be called with the subclass of the dispatched
    # `self._cls` and we must use the caller's cls for dispatching.
    cls, args = args[0], args[1:]
    # pylint: disable=protected-access
    if not issubclass(cls, self._cls):
      raise ValueError(
          f'Expected that {cls.__name__} is subclass of {self._cls.__name__}'
      )

    layer_args, layer_kwargs, args_index = _pack_args(args, kwargs)
    return _GraphPieceClassMethod(
        cls._type_spec_cls(), self._cls_method_name, args_index
    )(layer_args, **layer_kwargs)


@tf.keras.utils.register_keras_serializable(package='GNN')
class _GraphPieceClassMethod(tf.keras.layers.Layer):
  """Keras wrapper for class methods, see _GraphPieceClassMethodDispatcher."""

  @tf.__internal__.tracking.no_automatic_dependency_tracking
  def __init__(
      self,
      type_spec_cls: gp.GraphPieceSpecBase,
      cls_method_name: str,
      args_index,
      **kwargs,
  ):
    """Constructor.

    Args:
      type_spec_cls: The GraphPiece type spec class.
      cls_method_name: The wrapped class method name.
      args_index: A nest (*args, **kwargs) with the same structure as the
        arguments of the wrapped class method, indicating their position in the
        arguments of `call(layer_args, **layer_kwargs)`: an int is an index into
        the list `layer_args` and a str is a key into `layer_kwargs`.
      **kwargs: Extra arguments to pass to the base class.
    """
    super().__init__(**kwargs)
    assert issubclass(type_spec_cls, gp.GraphPieceSpecBase)
    self._type_spec_cls = type_spec_cls
    self._cls_method_name = cls_method_name
    self._args_index = args_index

    piece_cls = type_spec_cls._value_type()
    self._cls_method = getattr(piece_cls, cls_method_name)

  def get_config(self):
    return dict(
        cls_method_name=self._cls_method_name,
        type_spec_name=tf_internal.type_spec_get_name(self._type_spec_cls),
        args_index=self._args_index,
        **super().get_config(),
    )

  @classmethod
  def from_config(cls, config, custom_objects=None):
    config = config.copy()
    cls_method_name = config.pop('cls_method_name')
    type_spec_name = config.pop('type_spec_name')
    args_index = config.pop('args_index')

    return cls(
        type_spec_cls=tf_internal.type_spec_lookup(type_spec_name),
        cls_method_name=cls_method_name,
        args_index=args_index,
        **config,
    )

  def call(self, args, **kwargs):
    args = tf.nest.flatten(args)

    args, kwargs = _unpack_args(args, kwargs, self._args_index)
    return self._cls_method(*args, **kwargs)


# pylint: disable=protected-access
ContextKerasTensor._overload_operator(gt.Context, '__getitem__')
NodeSetKerasTensor._overload_operator(gt.NodeSet, '__getitem__')
EdgeSetKerasTensor._overload_operator(gt.EdgeSet, '__getitem__')
HyperAdjacencyKerasTensor._overload_operator(adj.HyperAdjacency, '__getitem__')

GRAPH_PIECE_PROPERTIES = ('_data',)
GRAPH_PIECE_WITH_FEATURES_PROPERTIES = GRAPH_PIECE_PROPERTIES + (
    'features',
    '_get_features_ref',
    'sizes',
    'total_size',
    'num_components',
    'total_num_components',
)

for _cls, _properties in [
    (EdgeSetKerasTensor, GRAPH_PIECE_WITH_FEATURES_PROPERTIES + ('adjacency',)),
    (NodeSetKerasTensor, GRAPH_PIECE_WITH_FEATURES_PROPERTIES),
    (ContextKerasTensor, GRAPH_PIECE_WITH_FEATURES_PROPERTIES),
    (
        GraphKerasTensor,
        (
            'node_sets',
            'edge_sets',
            'context',
            'num_components',
            'total_num_components',
        )
        + GRAPH_PIECE_PROPERTIES,
    ),
    (HyperAdjacencyKerasTensor, GRAPH_PIECE_PROPERTIES),
    (AdjacencyKerasTensor, GRAPH_PIECE_PROPERTIES + ('source', 'target')),
]:
  for _property in _properties:
    tf_internal.delegate_property(_cls, _property)

GRAPH_PIECE_METHODS = ('with_indices_dtype', 'with_row_splits_dtype')
GRAPH_PIECE_WITH_FEATURES_METHODS = (
    'get_features_dict',
    'replace_features',
) + GRAPH_PIECE_METHODS
for _cls, _methods in [
    (EdgeSetKerasTensor, GRAPH_PIECE_WITH_FEATURES_METHODS),
    (NodeSetKerasTensor, GRAPH_PIECE_WITH_FEATURES_METHODS),
    (ContextKerasTensor, GRAPH_PIECE_WITH_FEATURES_METHODS),
    (
        GraphKerasTensor,
        ('remove_features', 'replace_features', 'merge_batch_to_components')
        + GRAPH_PIECE_METHODS,
    ),
    (HyperAdjacencyKerasTensor, ('get_indices_dict',) + GRAPH_PIECE_METHODS),
]:
  for _method in _methods:
    tf_internal.delegate_method(_cls, _method)

for _cls, _class_methods in [
    (adj.HyperAdjacency, ('from_indices',)),
    (adj.Adjacency, ('from_indices',)),
    (gt.Context, ('from_fields',)),
    (gt.NodeSet, ('from_fields',)),
    (gt.EdgeSet, ('from_fields',)),
    (gt.GraphTensor, ('from_pieces',)),
]:
  for _class_method in _class_methods:
    _GraphPieceClassMethodDispatcher(_cls, _class_method).register(
        getattr(_cls, _class_method)
    )

tf_internal.register_keras_tensor_specialization(
    adj.Adjacency, AdjacencyKerasTensor
)
tf_internal.register_keras_tensor_specialization(
    adj.HyperAdjacency, HyperAdjacencyKerasTensor
)
tf_internal.register_keras_tensor_specialization(
    gt.GraphTensor, GraphKerasTensor
)
tf_internal.register_keras_tensor_specialization(gt.Context, ContextKerasTensor)
tf_internal.register_keras_tensor_specialization(gt.NodeSet, NodeSetKerasTensor)
tf_internal.register_keras_tensor_specialization(gt.EdgeSet, EdgeSetKerasTensor)


_OPS_REGISTRY = {}


class _TFGNNOpDispatcher:
  """Dispatches function calls for KerasTensor inputs.

  Wraps a TF-GNN library function as a TFGNNOpLambda Keras layer if any
  of the call inputs are Keras tensors. In particular, this allows to use TFGNN
  functions, such as `tf.broadcast(...)`, with the Keras Functional API.

  See `_GraphPieceClassMethodDispatcher` for details on how function arguments
  are translated into the Keras Layer inputs.
  """

  def __init__(self, op, name: Optional[str] = None):
    registry_key = name if name else op.__qualname__
    if registry_key in _OPS_REGISTRY:
      raise ValueError(f'Function {registry_key} is already registered.')

    self._op = op
    self._resistry_key = registry_key
    _OPS_REGISTRY[self._resistry_key] = op
    self.__doc__ = op.__doc__

  def __call__(self, *args, **kwargs):
    """Handle the specified operation with the specified arguments."""
    if not has_keras_tensors([args, kwargs]):
      return self._op(*args, **kwargs)

    layer_args, layer_kwargs, args_index = _pack_args(args, kwargs)
    return TFGNNOpLambda(self._resistry_key, args_index)(
        layer_args, **layer_kwargs
    )


@tf.keras.utils.register_keras_serializable(package='GNN')
class TFGNNOpLambda(tf.keras.layers.Layer):
  """Keras wrapper for any function, see _TFGNNOpDispatcher."""

  @tf.__internal__.tracking.no_automatic_dependency_tracking
  def __init__(
      self,
      registry_key: str,
      args_index,
      **kwargs,
  ):
    if registry_key not in _OPS_REGISTRY:
      raise ValueError(
          f'The function {registry_key} is not registered in TFGNN function'
          ' registry. The reason could be that the funciton was annotated with'
          ' @kt.delegate_keras_tensors, used with Keras saved model API and'
          ' then removed from the library,'
      )

    self._registry_key = registry_key
    self._args_index = args_index
    self._fn = _OPS_REGISTRY[registry_key]
    name = kwargs.pop('name', None)
    if name is None:
      name = tf_internal.unique_keras_object_name(
          registry_key, zero_based=True, avoid_observed_names=True
      )
    super().__init__(name=name, **kwargs)

  def get_config(self):
    return dict(
        registry_key=self._registry_key,
        args_index=self._args_index,
        **super().get_config(),
    )

  def call(self, args, **kwargs):
    args = tf.nest.flatten(args)
    args, kwargs = _unpack_args(args, kwargs, self._args_index)
    return self._fn(*args, **kwargs)


class _NotSupportedDispatcher:
  """Dispatcher for functions that disallow Keras Tensor arguments."""

  def __init__(
      self, op, name: Optional[str] = None, alternative: Optional[str] = None
  ):
    self._name = name if name else op.__qualname__
    self._op = op
    self._alternative = alternative
    self.__doc__ = op.__doc__

  def __call__(self, *args, **kwargs):
    """Handle the specified operation with the specified arguments."""
    if has_keras_tensors([args, kwargs]):
      msg = (
          f'Calling `{self._name}()` using the Keras Functional API is not'
          ' supported. Consider calling this function from the `call()`'
          ' method of a custom Keras Layer'
      )
      if self._alternative:
        msg += f' or {self._alternative} as an alternative.'

      raise ValueError(msg)

    return self._op(*args, **kwargs)


def delegate_keras_tensors(target=None, name: Optional[str] = None):
  """Annotation that wraps function as a Keras Layer in Keras Funcional API.

  Wraps function as `TFGNNOPLambda` layer when any of its call arguments is a
  `KerasTensor`.

  Although Keras, internally, wraps all TF Core functions as `TFOpLambda` layer,
  the implementation is tightly coupled with TF internals and does not allow
  generic use.

  This is a workaround that allows the library users to call TFGNN
  functions directly using the Keras Functional API without having to wrap each
  function call in its own Keras layer.

  NOTE: Do NOT expose to public API. The delegator is designed and tested for
  TFGNN functionality only and is not intended for general use. Each wrapped
  function must be carefully tested with `KerasTensor`s inputs.

  Args:
    target: The target function.
    name: The name to use for the dynamically created Keras layer.

  Returns:
    _TFGNNOpDispatcher if target is not None (when annotation is used without
    parameters) or callable that constructs _TFGNNOpDispatcher for the input
    target.
  """

  def decorator(target=None):
    impl = _TFGNNOpDispatcher(target, name)

    @functools.wraps(target)
    def fn(*argw, **kwargs):
      return impl(*argw, **kwargs)

    return fn

  if target is None:
    return decorator
  else:
    return decorator(target)


def disallow_keras_tensors(
    target=None, name: Optional[str] = None, alternative: Optional[str] = None
):
  """Annotation that disallows the use of the function in Keras Funcional API.

  It decorated function will raise an error if any of the inputs is a
  KerasTensor.

  NOTE: Internal use only.

  Args:
    target: The target function.
    name: The name to use for the dynamically created Keras layer.
    alternative: An optional message with an alternative that can be used
      instead of the attempted function (e.g. what `tfgnn.keras.layers` should
      be considered).

  Returns:
    _NotSupportedDispatcher if target is not None (when annotation is used
    without parameters) or callable that constructs _TFGNNOpDispatcher for the
    input target.
  """

  def decorator(target=None):
    impl = _NotSupportedDispatcher(target, name=name, alternative=alternative)

    @functools.wraps(target)
    def fn(*argw, **kwargs):
      return impl(*argw, **kwargs)

    return fn

  if target is None:
    return decorator
  else:
    return decorator(target)


def _pack_args(
    args, kwargs
) -> Tuple[List[tf_internal.KerasTensor], Dict[str, Any], Any]:
  """Packs positional and keyword arguments into Keras-compatible form.

  Args:
    args: positional arguments.
    kwargs: keyword arguments.

  Returns:
    A nest with the same strucutre as (*args, **kwargs) and indices.
  """
  inputs = (args, kwargs)
  flat_inputs = tf.nest.flatten(inputs)

  layer_args, layer_kwargs = [], {}
  args_index = []
  for value in flat_inputs:
    if isinstance(
        value, (tf_internal.KerasTensor, gp.GraphPieceBase)
    ) or tf.is_tensor(value):
      key = len(layer_args)
      layer_args.append(value)
      args_index.append(key)
    else:
      key = f'arg{len(layer_kwargs)}'
      layer_kwargs[key] = value
      args_index.append(key)
  assert inputs
  args_index = tf.nest.pack_sequence_as(inputs, args_index)
  return layer_args, layer_kwargs, args_index


def _unpack_args(
    args: List[tf_internal.KerasTensor], kwargs: Dict[str, Any], args_index: Any
) -> Tuple[Sequence[Any], Dict[str, Any]]:
  """Reverse operation to the `_pack_args()`."""
  args = tf.nest.flatten(args)

  def map_fn(key: Union[int, str]):
    return args[key] if isinstance(key, int) else kwargs[key]

  return tf.nest.map_structure(map_fn, args_index)


def has_keras_tensors(nest) -> bool:
  return any(
      isinstance(x, tf_internal.KerasTensor) for x in tf.nest.flatten(nest)
  )
