# Copyright 2022 Google LLC
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
#
# Example of building the docker image locally:
# docker build . -t tfgnn:latest
#
# You can then start an interactive shell with:
# docker run -it --entrypoint /bin/bash tfgnn:latest
# TODO:Change this to tensorflow/tensorflow:nightly-jupyter
FROM ubuntu:focal
# tzdata asks questions.
ENV DEBIAN_FRONTEND="noninteractive"
ENV TZ="America/New_York"
RUN apt-get -y update \
  && apt-get install -y apt-transport-https \
  && apt-get install -y curl \
  && apt-get install -y gnupg \
  && curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor > bazel.gpg \
  && mv bazel.gpg /etc/apt/trusted.gpg.d/ \
  && echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list \
  && apt-get -y update \
  && apt-get install -y --no-install-recommends \
  build-essential \
  pkg-config \
  bazel \
  python3.9 \
  python3.9-dev \
  python3.9-venv \
  graphviz-dev \
  graphviz
COPY . /app
# Set up venv to avoid root installing/running python
ENV VIRTUAL_ENV=/opt/venv
RUN python3.9 -m venv ${VIRTUAL_ENV}
# Put the virtual environment on the path.
ENV PATH="${VIRTUAL_ENV}/bin:/app/tensorflow_gnn:${PATH}"
RUN pip3 install --upgrade pip
# Add `--no-cache-dir` if disk space is an issue.
RUN pip3 install -U apache-beam[gcp] httplib2 notebook ogb
RUN python3 -m pip install /app
# Not sure why this gets downgraded during install process...
# RUN python3 -m pip install -U httplib2
# Install the apache beam sdk for local and dataflow runner support.
COPY --from=apache/beam_python3.9_sdk /opt/apache/beam /opt/apache/beam
# Set the default entry point to the apache beam binary for out-of-the box
# compatability with dataflow. This entrypoint can be overriden to run other
# tf-gnn programs within this environment.
ENTRYPOINT [ "/opt/apache/beam/boot" ]
