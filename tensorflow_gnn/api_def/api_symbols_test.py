# Copyright 2023 The TensorFlow GNN Authors. All Rights Reserved.
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
r"""Tests which symbols are in the public API.

# HOW-TO FOR TF-GNN DEVELOPERS

## How to fix a test failure after adding (removing?!) public symbols.

```
# Check out a writeable workspace from version control.
bazel build -c opt \  # Or Google's equivalent.
    tensorflow_gnn/api_def:api_symbols_test
bazel-bin/tensorflow_gnn/api_def/api_symbols_test \
    --refresh_golden_files
# Inspect diffs of *-symbols.txt in your IDE.
# Edit code and reiterate until diffs look good.
bazel-bin/tensorflow_gnn/api_def/api_symbols_test  # Check that it passes.
```

## How to add a new public sub-package of TF-GNN for testing

 0. Familiarize yourself with the multiple steps of the test code below.
 1. In step 1, add `from tensorflow_gnn.foo import bar` and
    `_ALL_PACKAGES = [ ..., "bar"]`
 2. In step 2, check and remove bar from foo and foo from tfgnn
    (see comments for an explanation of the technical Python background).
 3. Step 3 should "just work" based on `_ALL_PACKAGES`.
 4. For step 4, create `./bar-symbols.txt` **before** re-building and
    running as in "how to fix a test failure"
"""


## NOTE: This test is about the behavior of import statements at module scope,
## so it is structured differently from the usual Python unit tests: there are
## multiple steps of importing and capturing symbols before the usual
## test fixture runs.

##
## STEP 1: Import the public Python packages of TF-GNN before any other code
##

from __future__ import annotations

# pylint: disable=unused-import
import tensorflow_gnn as tfgnn
from tensorflow_gnn import runner
from tensorflow_gnn.experimental import sampler
from tensorflow_gnn.models import contrastive_losses
from tensorflow_gnn.models import gat_v2
from tensorflow_gnn.models import gcn
from tensorflow_gnn.models import graph_sage
from tensorflow_gnn.models import hgt
from tensorflow_gnn.models import mt_albis
from tensorflow_gnn.models import multi_head_attention
from tensorflow_gnn.models import vanilla_mpnn
# pylint: enable=unused-import


# Remember the names of all imported packages for the actual testing below.
_ALL_PACKAGES = [
    "tfgnn",
    "runner",
    "sampler",
    "contrastive_losses",
    "gat_v2",
    "gcn",
    "graph_sage",
    "hgt",
    "mt_albis",
    "multi_head_attention",
    "vanilla_mpnn",
]

##
## STEP 2: Deal with subpackages added after __init__.py
##

#
# BACKGROUND INFO ON PYTHON IMPORTS
#
# Recall that `from foo.bar import baz` does three things:
#
#  1. If not already present in the sys.modules dict, a module object is
#     created and initialized by its __init__.py for each of foo, bar and baz.
#  2. After creating the module bar, the module object of foo is updated
#     (after its __init__.py had already completed!) by setting its attribute
#     foo.bar = bar. Likewise, after creating baz, it sets bar.baz = baz.
#  3. Finally, `baz` is set in the current scope.

#
# HOW TF-GNN AND THIS TEST HANDLE IT
#
# This test needs to address three different effects of item (2):
#
#  * Library users are meant to `import tensorflow_gnn as tfgnn` to obtain
#    `tfgnn` itself and certain automatically imported submodules such as
#    `tfgnn.keras.layers`. This test checks all of them and their expected
#    symbols.
#  * Library users are meant to `from tensorflow_gnn import runner` and then
#    access it through the `runner` symbol. Importing redundantly adds the
#    symbol `tfgnn.runner`; the following code removes the redundant alias
#    before collecting the actual content of `tfgnn` and `runner` separately.
#  * Any import statement for a public package causes imports of private
#    implementation-only modules in __init__.py files and thus sets
#    attributes for them in public module objects. This test verifies that
#    these are all removed again before control is returned back to the user
#    (e.g., by calling _api_utils.remove_submodules_except(__name__, [...])).

assert tfgnn.runner is runner
del tfgnn.runner

assert tfgnn.experimental.sampler is sampler
del tfgnn.experimental.sampler

assert tfgnn.models.contrastive_losses is contrastive_losses
del tfgnn.models.contrastive_losses

assert tfgnn.models.gat_v2 is gat_v2
del tfgnn.models.gat_v2

assert tfgnn.models.gcn is gcn
del tfgnn.models.gcn

assert tfgnn.models.graph_sage is graph_sage
del tfgnn.models.graph_sage

assert tfgnn.models.hgt is hgt
del tfgnn.models.hgt

assert tfgnn.models.mt_albis is mt_albis
del tfgnn.models.mt_albis

assert tfgnn.models.multi_head_attention is multi_head_attention
del tfgnn.models.multi_head_attention

assert tfgnn.models.vanilla_mpnn is vanilla_mpnn
del tfgnn.models.vanilla_mpnn
del tfgnn.models

# TODO(b/316135889): remove once fixed.
del tfgnn.proto.graph_schema_pb2

##
## STEP 3: Recursively collect all module attributes exposed by the public API.
##

# pylint: disable=g-import-not-at-top,g-bad-import-order
import types
# pylint: enable=g-import-not-at-top,g-bad-import-order


def get_symbols(
    module_name: str, module: types.ModuleType, depth_limit=6) -> list[str]:
  """Returns all non-dunder symbols from module, recursively."""
  result = []
  for symbol in dir(module):
    if symbol.startswith("__"):
      continue  # Ignore Python internals like __file__ etc.
    prefixed_symbol = f"{module_name}.{symbol}"
    obj = getattr(module, symbol)
    if isinstance(obj, types.ModuleType):
      if depth_limit > 0:
        result.extend(get_symbols(prefixed_symbol, obj, depth_limit - 1))
      else:
        raise RecursionError(
            f"Recursion depth limit exceeded for {prefixed_symbol}")
    else:
      result.append(prefixed_symbol)
  return result


_ACTUAL_SYMBOLS = {
    k: get_symbols(k, globals()[k]) for k in _ALL_PACKAGES
}


##
## STEP 4: Test against golden files (with an option to refresh them).
##


# pylint: disable=g-import-not-at-top,g-bad-import-order
from absl import flags
from absl.testing import absltest
from absl.testing import parameterized

from tensorflow_gnn.utils import test_utils
# pylint: enable=g-import-not-at-top,g-bad-import-order


_REFRESH_GOLDEN_FILES = flags.DEFINE_bool(
    "refresh_golden_files", False,
    "If set, overwrites the files with the expected symbols by the actual "
    "symbols observe in the test. This only works when running localy in a "
    "Piper client and having all files writeable (by `g4 edit` or "
    "`p4 client --set_option=allwrite`)")


class ApiSymbolsTest(parameterized.TestCase):

  @parameterized.named_parameters((f"_{m}", m) for m in _ALL_PACKAGES)
  def test(self, module_name):
    actual = set(_ACTUAL_SYMBOLS[module_name])
    golden_file_name = test_utils.get_resource(
        f"api_def/{module_name}-symbols.txt")
    if _REFRESH_GOLDEN_FILES.value:
      print(f"Refreshing {golden_file_name} with {len(actual)} symbols")
      with open(golden_file_name, "w") as f:
        for s in sorted(actual):
          print(s, file=f)
    with open(golden_file_name, "r") as f:
      expected = set(l.strip() for l in f.readlines())
    extraneous = actual - expected
    self.assertEmpty(extraneous)
    missing = expected - actual
    self.assertEmpty(missing)


if __name__ == "__main__":
  absltest.main()
