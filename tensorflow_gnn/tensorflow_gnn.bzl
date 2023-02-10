"""GNN common starlark macros."""

def clean_dep(target):
    """Returns string to 'target' in this repository.
    """

    # A repo-relative label is resolved relative to the file in which the
    # Label() call appears, i.e. @org_tensorflow.
    return str(Label(target))

# Placeholder to use until bazel supports py_strict_binary.
def py_strict_binary(name, **kwargs):
    native.py_binary(name = name, **kwargs)

# Placeholder to use until bazel supports py_strict_library.
def py_strict_library(name, **kwargs):
    native.py_library(name = name, **kwargs)

# Placeholder to use until bazel supports pytype_strict_binary.
def pytype_strict_binary(name, **kwargs):
    native.py_binary(name = name, **kwargs)

# Placeholder to use until bazel supports pytype_strict_library.
def pytype_strict_library(name, **kwargs):
    native.py_library(name = name, **kwargs)

# Placeholder to use until bazel supports pytype_library.
def pytype_library(name, **kwargs):
    native.py_library(name = name, **kwargs)

# Deletes deps with a no_tfgnn_py_deps flag so that tests
# can run from the pip wheel instead of bazel runfiles
def py_test(name, deps = [], **kwargs):
    native.py_test(
        name = name,
        deps = select({
            "//conditions:default": deps,
            clean_dep("//tensorflow_gnn:no_tfgnn_py_deps"): [],
        }),
        **kwargs
    )

# Placeholder to use until bazel supports py_strict_test.
def py_strict_test(name, deps = [], **kwargs):
    py_test(
        name = name,
        deps = deps,
        **kwargs
    )

# Placeholder to use until bazel supports pytype_strict_contrib_test
def pytype_strict_contrib_test(name, deps = [], **kwargs):
    py_test(
        name = name,
        deps = deps,
        **kwargs
    )

# This is a trimmed down version of tf_py_test since a lot of internal
# features are just not available to OSS build, and also not applicable to TFGNN.
# So far xla, grpc and tfrt are ignored.
def tf_py_test(name, deps = [], **kwargs):
    py_test(
        name = name,
        deps = deps,
        **kwargs
    )

# This is a trimmed down version of distribute_py_test since a lot of internal
# features are just not available to OSS build, and also not applicable to TFGNN.
# Especially the TPU tests branches are removed.
def distribute_py_test(
        name,
        deps = [],
        main = None,
        xla_enable_strict_auto_jit = False,
        **kwargs):
    # Default to PY3 since multi worker tests require PY3.
    kwargs.setdefault("python_version", "PY3")
    main = main if main else "%s.py" % name
    xla_enable_strict_auto_jit = xla_enable_strict_auto_jit  # Dummy to suppress warning
    tf_py_test(
        name = name,
        deps = deps,
        main = main,
        **kwargs
    )
