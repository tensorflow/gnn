#!/bin/bash
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

NUM_WORKER_THREADS=2
MACHINE_TYPE="n1-highmem-2"

DATASET="mag"
MAX_NUM_WORKERS=100
MIN_NUM_WORKERS=30

TIMESTAMP="$(date +"%Y-%m-%d-%H-%M-%S")"

GOOGLE_CLOUD_PROJECT="<Your Google Cloud Project>"
EXAMPLE_ARTIFACT_DIRECTORY="gs://${GOOGLE_CLOUD_PROJECT}/sampler/${DATASET}/${TIMESTAMP}"

python3 -m tensorflow_gnn.experimental.sampler.beam.sampler \
  --project ${GOOGLE_CLOUD_PROJECT} \
  --region "us-east1" \
  --save_main_session \
  --setup_file "./setup.py" \
  --graph_schema "gs://<path-to-data-dir>/graph_schema.pbtxt" \
  --sampling_spec "gs://<path-to-data-dir>/sampling_spec.pbtxt" \
  --output_samples "${EXAMPLE_ARTIFACT_DIRECTORY}/outputs/examples.tfrecord" \
  --runner DataflowRunner \
  --temp_location "${EXAMPLE_ARTIFACT_DIRECTORY}/tmp" \
  --machine_type ${MACHINE_TYPE} \
  --experiments "min_num_workers=${MIN_NUM_WORKERS}" \
  --max_num_workers ${MAX_NUM_WORKERS} \
  --num_workers ${MIN_NUM_WORKERS} \
  --number_of_worker_harness_threads ${NUM_WORKER_THREADS} \
  --dataflow_service_options enable_google_cloud_profiler