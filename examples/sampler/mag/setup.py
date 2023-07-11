# Copyright 2016 Google Inc. All Rights Reserved.
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
"""Package Setup script for tf.Transform."""
from setuptools import find_packages
from setuptools import setup


def _make_required_install_packages():
  # Make sure to sync the versions of common dependencies (absl-py, numpy, and
  # protobuf) with TF and pyarrow version with tfx-bsl.
  return [
      'absl-py>=0.9,<2.0.0',
      'apache-beam[gcp]>=2.41,<3',
      'numpy',
      'tensorflow',
      'tensorflow_gnn',
  ]


# Get the long description from the README file.
setup(
    name='sampler',
    version='0.0.1',
    author='Google Inc.',
    license='Apache 2.0',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    namespace_packages=[],
    install_requires=_make_required_install_packages(),
    python_requires='>=3.8,<4',
    packages=find_packages(),
    include_package_data=True,
    description='A library for graph sampling',
    keywords='tfgnn sampler',
    url='https://github.com/tensorflow/gnn',
    download_url='https://github.com/tensorflow/gnn',
    requires=[],
)
