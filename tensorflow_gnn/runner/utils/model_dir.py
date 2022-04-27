"""Model directory methods."""
import os

import tensorflow as tf


def incrementing_model_dir(dirname: str, start: int = 0) -> str:
  """Create, given some `dirname`, an incrementing model directory.

  Args:
    dirname: The base directory name.
    start: The starting integer.

  Returns:
    A model directory `dirname/n` where 'n' is the maximum integer in `dirname`.
  """
  if not tf.io.gfile.isdir(dirname):
    return os.path.join(dirname, str(start))
  files = tf.io.gfile.listdir(dirname)
  integers = [int(f) for f in files if f.isdigit()]
  return os.path.join(dirname, str(max(integers) + 1 if integers else start))
