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
"""Sampling-related utils."""

import random
from typing import Any, List, Optional, Iterable, Iterator, Tuple
import apache_beam as beam

Key = Any
QueryData = Any
ValueData = Any
ShardedKey = Tuple[Key, int]
PCollection = beam.pvalue.PCollection


def unique_values_combiner(values: Iterable[List[Any]],
                           max_result_size: Optional[int] = None) -> List[Any]:
  """Extracts unique values from multiple input list.

  Args:
    values: Input lists.
    max_result_size: The maximum result caridatinality, if applied. Used to
      validate result.

  Returns:
    Unique values aggregated from input.

  Raises:
    ValueError: If result size if larger then `max_result_size`.
  """
  result = set()
  for ids in values:
    result.update(ids)
  result = list(result)
  if max_result_size is not None and len(result) > max_result_size:
    raise ValueError(
        f"Result size {len(result)} is larger than {max_result_size}.")

  return result


def balanced_inner_lookup_join(
    stage_name: str,
    queries: PCollection[Tuple[Key, QueryData]],
    lookup_table: PCollection[Tuple[Key, ValueData]],
    num_shards: int = 1024
) -> PCollection[Tuple[Key, Tuple[QueryData, ValueData]]]:
  """Matches keys from queries with unique values.

  Results are returned only for matched keys.

  Compared to the `CoGroupByKey` the implementation avoids materialization of
  all queries having the same key on a single worker memory. This is achieved by
  sharding the query keys into the `num_shards` buckets, duplicating values for
  each key's shard and joining sharded queries with sharded values using
  `(key, shard)` tuple. The approach allows to better load-balance hot keys
  with the cost of introducing more stages.

  Args:
    stage_name: The unique Beam stage name.
    queries: Pairs of query keys and associated query data.
    lookup_table: Values keyed by unique keys.
    num_shards: The total number of buckets used to shard input keys. This
      parameter controls how hot keys are sharded.

  Returns:
    Pairs of query data and value for matched keys.

  Raises:
    ValueError: If value keys are not unique or number of shards is not
      positive.
  """
  if num_shards <= 0:
    raise ValueError("The number of shards should be positive,"
                     f" got {num_shards}.")

  def shard_fn(key: Key, data: QueryData) -> Tuple[ShardedKey, QueryData]:
    return (key, random.randint(0, num_shards - 1)), data

  def shard_values(key: Key, group) -> Iterator[Tuple[ShardedKey, ValueData]]:
    for value_index, value in enumerate(group["value"]):
      if value_index != 0:
        raise ValueError(f"Values are not unique for key={key}.")

      for shard_index, shards in enumerate(group["shards"]):
        assert shard_index == 0, f"Shards are not unique for key={key}."
        for shard in shards:
          yield (key, shard), value

  sharded_queries: PCollection[Tuple[ShardedKey, QueryData]] = (
      queries | f"{stage_name}/ShardQueries" >> beam.MapTuple(shard_fn))

  def as_list(key: Key, shard: int)-> Tuple[Key, List[int]]:
    return key, [shard]

  unique_shards: PCollection[Tuple[Key, List[int]]] = (
      sharded_queries
      | f"{stage_name}/ExtractShardedKeys" >> beam.Keys()
      | f"{stage_name}/ConvertShardToList" >> beam.MapTuple(as_list)
      | f"{stage_name}/CreateUniqueShards" >> beam.CombinePerKey(
          unique_values_combiner, max_result_size=num_shards))
  sharded_values: PCollection[Tuple[ShardedKey, ValueData]] = (
      {
          "shards": unique_shards,
          "value": lookup_table
      }
      | f"{stage_name}/JoinValuesAndShards" >> beam.CoGroupByKey()
      | f"{stage_name}/ShardValues" >> beam.FlatMapTuple(shard_values))

  def extract_result(
      sharded_key: ShardedKey,
      group) -> Iterator[Tuple[Key, Tuple[QueryData, ValueData]]]:
    for value_data in group["value"]:
      for query_data in group["queries"]:
        yield sharded_key[0], (query_data, value_data)

  return ({
      "queries": sharded_queries,
      "value": sharded_values
  }
          | f"{stage_name}/JoinShardedQueriesAndValues" >> beam.CoGroupByKey()
          | f"{stage_name}/ExtractResult" >> beam.FlatMapTuple(extract_result))
