# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Package Setup script for TensorFlow GNN."""

import os
import platform
import shutil
import subprocess
import sys
import setuptools
from setuptools import dist
from setuptools.command import build
from setuptools.command import install


class _BuildCommand(build.build):
  """Build everything needed to install.

  This overrides the original distutils "build" command to to run bazel_build
  command instead, before any sub_commands. This is convenient in order to
  generate protocol buffer files and eventually also build C++ extension
  modules.

  The build command is also invoked from bdist_wheel and install command,
  therefore this implementation covers the following commands:

    - pip install . (which invokes bdist_wheel)
    - python setup.py install (which invokes install command)
    - python setup.py bdist_wheel (which invokes bdist_wheel command)

  """

  def _build_cc_extensions(self):
    return True

  # Add the "bazel_build" command as the first sub-command of "build". Each
  # sub_command of "build" (e.g. "build_py", "build_ext", etc.) is executed
  # sequentially when running a "build" command, if the second item in the tuple
  # (predicate method) is evaluated to true.
  sub_commands = [
      ('bazel_build', _build_cc_extensions)
  ] + build.build.sub_commands


class _BazelBuildCommand(setuptools.Command):
  """Build C++ extensions and public protos with Bazel.

  Running this command will populate the *_pb2.py files next to your *.proto
  files.
  """

  def initialize_options(self):
    pass

  def finalize_options(self):
    self._bazel_cmd = shutil.which('bazel')
    if not self._bazel_cmd:
      raise RuntimeError(
          'Could not find "bazel" binary. Please visit '
          'https://docs.bazel.build/versions/master/install.html for '
          'installation instruction.')
    self._additional_build_options = []
    if platform.system() == 'Darwin':
      self._additional_build_options = ['--macos_minimum_os=10.9']
    elif platform.system() == 'Windows':
      self._additional_build_options = ['--copt=-DWIN32_LEAN_AND_MEAN']

  def run(self):
    subprocess.check_call(
        [self._bazel_cmd,
         'run', '-c', 'opt', '--experimental_repo_remote_exec'] +
        self._additional_build_options +
        ['@tensorflow_gnn//package:move_generated_files'],
        # Bazel should be invoked in a directory containing bazel WORKSPACE
        # file, which is the root directory.
        cwd=os.path.dirname(os.path.realpath(__file__)),
        env=dict(os.environ, PYTHON_BIN_PATH=sys.executable))


# TFDV is not a purelib. However because of the extension module is not built
# by setuptools, it will be incorrectly treated as a purelib. The following
# works around that bug.
class _InstallPlatlibCommand(install.install):

  def finalize_options(self):
    install.install.finalize_options(self)
    self.install_lib = self.install_platlib


class _SourceDistributionWithProtos(dist.Distribution):
  """Proto-only deps don't require OS specific wheels."""

  def is_pure(self):
    return True

  def has_ext_modules(self):
    return False


setuptools.setup(
    distclass=_SourceDistributionWithProtos,
    cmdclass={
        'install': _InstallPlatlibCommand,
        'build': _BuildCommand,
        'bazel_build': _BazelBuildCommand,
    },
)
