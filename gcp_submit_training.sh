#!/bin/bash
# BUCKET_NAME=${PROJECT_ID}-aiplatform1

# Define region
REGION=us-central1

# Define image URI
IMAGE_URI=gcr.io/dtumlops-338009/testing:latest

# # Create model directory
# MODEL_DIR=testing_model_$(date +%Y%m%d_%H%M%S)

# Define unique job name
JOB_NAME=custom_container_job_$(date +%Y%m%d_%H%M%S)

# Submit job to AI- Platform
gcloud ai-platform jobs submit training $JOB_NAME \
  --region $REGION \
  --master-image-uri $IMAGE_URI 
  # \
  # -- \
  # --model-dir=gs://$BUCKET_NAME/$MODEL_DIR