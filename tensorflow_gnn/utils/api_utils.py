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
"""Utilities for defining the library API in _init__.py files.

The TF-GNN library consists of several public Python packages that each must be
imported individually, for example:

```
# Typical user code.
import tensorflow_gnn as tfgnn
from tensorflow_gnn import runner
from tensorflow_gnn.models import mt_albis
```

Each package `tensorflow_gnn.foo` is defined by a file `foo/__init__.py`
that

 1. imports private subpackages with the actual implementations,
    such as `from tensorflow_gnn.foo import bar_lib`;
 2. assigns public names for some of the imported objects, for example,
   `Bar = bar_lib.bar`;
 3. finally, removes all names that are not meant to be public, especially
    those of imported subpackages: see `remove_submodules_except()`.

This does not preclude automatically importing and exposing submodules.
For example, `tfgnn.keras` and `tfgnn.keras.layers` are imported as part of
`tfgnn`. That's what the "except" part of `remove_submodules_except()` is for.

Package authors should remember two technical constraints:

  * The __init__.py file should not contain function or class definitions,
    because function bodies need rely on the presence of imported identifiers
    at module scope at the time they are executed (not at the time they are
    parsed), but by then step 3 has already removed those identifiers.

  * If foo/bar.py imports foo/qux.py, this defines the attribute foo.qux even if
    foo/__init__.py never mentions the name qux, but only the first time such an
    import happens. Afterwards, sys.modules[] caches qux, and this kind of
    initialization is not repeated. Therefore:
    ANY PRIVATE SUB-MODULE OF tensorflow_gnn.foo MUST BE IMPORTED (directly or
    indirectly) BY foo/__init__.py, so that `remove_submodules_except()` can
    get rid of it before user code can see the module object foo. Importing
    tensorflow_gnn.foo.qux_lib from tensorflow_gnn.eeek would add foo.qux_lib
    after foo/__init__.py has finished executing.

Note that, at the Python level, there is no safeguard against direct imports of
private modules such as `from tensorflow_gnn.foo import bar_lib` in user code.
That has to be addressed separately (API doc, lint checks or the like).

We could have avoided some hassle by putting all implementation files into a
`_src` subdirectory that, even if found in the containing module object, has
a clearly private name.
"""

from __future__ import annotations

import sys
import types
from typing import Iterable


# No unit test, tested end-to-end via ../api_def/api_symbols_test.py.
def remove_submodules_except(module_name: str, allowed: Iterable[str]):
  """From sys.modules[module_name], deletes all submodules not allowed here.

  This is meant to be called as follows as part of defining the public
  interface of package tensorflow_gnn.foo:

  ```
  # tensorflow_gnn/foo/__init__.py:
  from tensorflow_gnn.foo import bar_lib  # Private: removed by default.
  from tensorflow_gnn.foo import util  # Public: explicitly excepted below.
  from tensorflow_gnn.utils import api_utils  # Private: removed by default.

  Bar = bar_lib.Bar  # Export a symbol from bar_lib.

  # Remove `bar_lib` and all side effects of transitive imports.
  api_utils.remove_submodules_except(__name__, ["util"])
  ```

  Args:
    module_name: A key to `sys.modules[]`, usually set to `__name__`.
      Upon return, that module object has all attributes deleted whose name
      did not start with "__" and whose value was a module object.
    allowed: Attribute names to keep, despite being bound to module objects.
  """
  module = sys.modules[module_name]
  allowed_set = set(allowed)
  for symbol in dir(module):
    if symbol.startswith("__") or symbol in allowed_set:
      continue  # Keep Python-internal and allowed symbols regardless of type.
    obj = getattr(module, symbol)
    if not isinstance(obj, types.ModuleType):
      continue  # Keep symbols other than submodules.
    delattr(module, symbol)
