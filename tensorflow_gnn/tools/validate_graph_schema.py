"""Validate a schema's features and shapes.

This script ensures that a schema is valid, has correct shapes, and isn't
clobbering over reserve feature names.
"""

import sys

from absl import app
from absl import flags
from absl import logging
import tensorflow_gnn as tfgnn


FLAGS = flags.FLAGS


def define_flags():
  """Define program flags."""

  flags.DEFINE_string("graph_schema", None,
                      ("A filename to a text-formatted schema proto describing "
                       "the available graph features."))

  flags.mark_flag_as_required("graph_schema")


def app_main(unused_argv):
  """App runner main function."""
  schema = tfgnn.read_schema(FLAGS.graph_schema)
  try:
    warnings = tfgnn.validate_schema(schema)
    for warning in warnings:
      logging.warning(warning)
    logging.info("Schema validated correctly.")
  except tfgnn.ValidationError as exc:
    logging.error("Schema validation error: %s", exc)
    sys.exit(1)


def main():
  define_flags()
  app.run(app_main)


if __name__ == "__main__":
  main()
