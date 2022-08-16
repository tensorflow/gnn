#!/bin/bash
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
#
# Dataflow workers must have access to the tfgnn docker container.
# Clients must have a local copy of the tensorflow_gnn image to launch the
# dataflow job from within the container.
#
# Clients must also have access to the target's GCP project application default
# credentials in their home directory.
#
# To run sampling as a Dataflow job, clients must have run the data generation
# tool in tensorflow_gnn/examples/Makefile and copied the output folder to GCS.
# An end to end example from building docker, pushing to GCR,
# copying data to GCS, and running sampling:
#
# docker build [path-to]/tensorflow_gnn -t tfgnn:latest \
#   -t gcr.io/${GOOGLE_CLOUD_PROJECT}/tfgnn:latest
# docker push gcr.io/${GOOGLE_CLOUD_PROJECT}/tfgnn:latest
#
# docker run -v /tmp:/tmp \
#   -it --entrypoint make tfgnn:latest -C /app/examples/arxiv graph \
#
# gsutil cp /tmp/data/ogbn-arxiv/graph \
#   gs://${GOOGLE_CLOUD_PROJECT}/tfgnn/examples/arxiv
#
# ./[path-to]/tensorflow_gnn/examples/arxiv/sample_dataflow.sh
#
MAX_NUM_WORKERS=1000

# The sampling spec is included in the container, read by the controller and
# sent to the remote Dataflow server.
SAMPLING_SPEC="/app/examples/mag/sampling_spec.pbtxt"

# The values below are just suggestions, feel free to change them.
GOOGLE_CLOUD_PROJECT="[FILL-ME-IN]"
GOOGLE_CLOUD_COMPUTE_REGION="[FILL-ME-In]"
# Make sure you have already made the GCP bucket with something like:
# `gsutil mb gs://${GOOGLE_CLOUD_PROJECT}`.
TIMESTAMP="$(date +"%Y-%m-%d-%H-%M-%S")"
EXAMPLE_ARTIFACT_DIRECTORY="gs://${GOOGLE_CLOUD_PROJECT}/tfgnn/ogbn-mag/${TIMESTAMP}"

# Sampler expects the graph artifacts to be in the same directory as the
# schema.
GRAPH_SCHEMA="${EXAMPLE_ARTIFACT_DIRECTORY}/schema.pbtxt"

# Required by Dataflow
TEMP_LOCATION="${EXAMPLE_ARTIFACT_DIRECTORY}/tmp"

# (Sharded) output sample tfrecord filespec.
OUTPUT_SAMPLES="${EXAMPLE_ARTIFACT_DIRECTORY}/samples@100"

# This should be a path to a docker image with TFGNN installed that pinned to
# the image version that the user is running this script with. A valid example
# using Google Container Registry:
# `gcr.io/${GOOGLE_CLOUD_PROJECT}/tfgnn:latest`.
REMOTE_WORKER_CONTAINER="[FILL-ME-IN]"

# Useful to define a private GCP VPC hat does not allocate external IP addresses
# so worker machines do not impact quota limits.
GCP_VPC_NAME="[FILL-ME-IN]"

EDGE_AGGREGATION_METHOD="edge"

JOB_NAME="tfgnn-mag-sampling-${EDGE_AGGREGATION_METHOD}-${TIMESTAMP}"

# Placeholder for Google-internal script config

docker run -v ~/.config/gcloud:/root/.config/gcloud \
  -e "GOOGLE_CLOUD_PROJECT=${GOOGLE_CLOUD_PROJECT}" \
  -e "GOOGLE_APPLICATION_CREDENTIALS=/root/.config/gcloud/application_default_credentials.json" \
  --entrypoint tfgnn_graph_sampler \
  tfgnn:latest \
  --graph_schema="${GRAPH_SCHEMA}" \
  --sampling_spec="${SAMPLING_SPEC}" \
  --output_samples="${OUTPUT_SAMPLES}" \
  --edge_aggregation_method="${EDGE_AGGREGATION_METHOD}" \
  --runner=DataflowRunner \
  --project=${GOOGLE_CLOUD_PROJECT} \
  --region=${GOOGLE_CLOUD_COMPUTE_REGION} \
  --max_num_workers="${MAX_NUM_WORKERS}" \
  --temp_location="${TEMP_LOCATION}" \
  --job_name="${JOB_NAME}" \
  --no_use_public_ips \
  --network="${GCP_VPC_NAME}" \
  --dataflow_service_options=enable_prime \
  --experiments=use_monitoring_state_manager \
  --experiments=enable_execution_details_collection \
  --experiment=use_runner_v2 \
  --worker_harness_container_image=gcr.io/${GOOGLE_CLOUD_PROJECT}/tfgnn:latest \
  --number_of_worker_harness_threads="${NUMBER_OF_WORKER_HARNESS_THREADS}" \
  --alsologtostderr
