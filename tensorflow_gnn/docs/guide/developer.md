# GitHub Developer Guide

## Environment Setup

We recommend setting up a local environment with the needed dev tools.

1.  [Bazel](https://bazel.build/) is the tool to build and test TF-GNN. See the
    [installation guide](https://bazel.build/install)
    for how to install and config bazel for your local environment.
2.  [git](https://github.com/) for code repository management.
3.  [python](https://www.python.org/) to build and code in TF-GNN.

NOTE: TF-GNN 1.0 has been tested with bazel 7 but not bazel 8+.
If you use [bazelisk](https://bazel.build/install/bazelisk), you can select a
version by setting an environment variable like this:
`export USE_BAZEL_VERSION=7.4.1`

The following commands check the above tools are successfully installed. Note
that tensorflow_gnn requires at least the version specified in [setup.py](https://github.com/tensorflow/gnn/setup.py).

```shell
bazel --version
git --version
python --version
```

A [Python virtual environment](https://docs.python.org/3/tutorial/venv.html)
(venv) is a powerful tool to create a self-contained environment that isolates
any changes from system level configuration. It is highly recommended to avoid
any unexpected dependency or version issue.

With the following commands, you create a new venv, named `venv_dir`.

```shell
mkdir venv_dir
python3 -m venv venv_dir
```

You can activate the venv with the following command. You should always run the
tests with the venv activated. You need to activate the venv every time you open
a new shell.

```shell
source venv_dir/bin/activate  # for linux or MacOS
```

Clone your forked repo to your local machine. Go to the cloned directory to
install the dependencies into the venv.

```shell
git clone https://github.com/YOUR_GITHUB_USERNAME/gnn.git
cd gnn
pip install -r requirements-dev.txt
pip install tensorflow
```

For TF2.16+, you will additionally need to follow the instructions from the
[Keras version](./keras_version.md) guide.

## Building the package and running tests

We use [Bazel](https://bazel.build/) to build and run tests. Since we use
[Protocol Buffers](https://developers.google.com/protocol-buffers) for
serialization and schemas, there is an intermediate build step before running
tests. To avoid needing to build all of TensorFlow in this build step, we build
only the protobuf components and run tests with the pip package in a separate
directory.

### Build the wheel

First, we need to build the Python package and install it in the venv.

```shell
python3 setup.py bdist_wheel
pip install dist/tensorflow_gnn-*.whl
```

### Create a clean test directory

Next, we need to set up a test directory with all the build and test targets. We
do this by sym-linking to the current tensorflow_gnn directory and using
specific Bazel flags as in the next section.

```shell
mkdir -p bazel_pip
ln -s "$(pwd)"/tensorflow_gnn "$(pwd)"/bazel_pip/tensorflow_gnn
```

### Run all tests

After setting up the test directory, you can run all tests locally by running
the following command in the repo root directory.

```
bazel test --build_tag_filters=-no_oss,-oss_excluded --test_tag_filters=-no_oss,-oss_excluded --test_output=errors --verbose_failures=true --build_tests_only --define=no_tfgnn_py_deps=true --keep_going --experimental_repo_remote_exec --test_env="TF_USE_LEGACY_KERAS=1" //bazel_pip/tensorflow_gnn/...
```

The `--define=no_tfgnn_py_deps=true` flag directs bazel to assume that all
dependencies are provided in the environment (where we just installed the .whl)

The flags `--build_tag_filters=-no_oss,-oss_excluded` and
`--test_tag_filters=-no_oss,-oss_excluded` disable tests that pass in the
internal production environment but fail on GitHub.

The flag `--test_env="TF_USE_LEGACY_KERAS=1"` comes from the
[Keras version](./keras_version.md) guide and is required for TF2.16+.

### Run a single test file

It is also possible to run a single test file by specifying its path (for
example)

`bazel test --define=no_tfgnn_py_deps=true --experimental_repo_remote_exec
--test_env="TF_USE_LEGACY_KERAS=1"
//bazel_pip/tensorflow_gnn/models/gcn:gcn_conv_test`

### Run a single test case

To run a single test case, use the `--test_filter` flag.

`bazel test --define=no_tfgnn_py_deps=true --experimental_repo_remote_exec
--test_env="TF_USE_LEGACY_KERAS=1"
//bazel_pip/tensorflow_gnn/models/gcn:gcn_conv_test
--test_filter=*test_gcnconv_activation*`

### Cleanup

To clean all the artifacts produced, run the following commands.

```shell
bazel clean
rm -rf bazel-*
rm -rf build
rm -rf dist
rm -rf *.egg-info
rm -rf bazel_pip
pip uninstall -y tensorflow_gnn
```
