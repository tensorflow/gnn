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
# Simple script for using TFGNN docker image to download and format OGBN-MAG.
# Dataset download is approximately 0.4 GB and expands on disk to be larger.
# To run the batch sampler using Dataflow on this dataset, the graph artifacts
# must be pushed to GCS. Assuming the user is authenticated to work with a
# GCP project with a bucket named ${BUCKET}, this can be done with a command
# akin to (This will copy approximately 2GB of data to GCS):
#
# gsutil -m cp -r ${OUTPUT_PATH} gs://${BUCKET}/tfgnn/examples/obgn-mag
#
DATASET="ogbn-mag"
DOWNLOAD_PATH="/tmp/ogb-preprocessed"
OUTPUT_PATH="/tmp/data/${DATASET}/graph"

docker run -it --entrypoint tfgnn_convert_ogb_dataset \
  -v /tmp:/tmp \
  tfgnn:latest \
  --dataset="${DATASET}" \
  --ogb_datasets_dir="${DOWNLOAD_PATH}" \
  --output="${OUTPUT_PATH}"

sudo chown -R ${USER} ${OUTPUT_PATH}

# Copy over the extended schema with the "written" relationship.
cp $(dirname $0)/schema.pbtxt ${OUTPUT_PATH}
