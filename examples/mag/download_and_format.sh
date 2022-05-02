#!/bin/bash
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
