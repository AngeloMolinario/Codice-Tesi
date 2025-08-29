#!/bin/bash

# Script to run the test.py script with specific parameters

# Define default values
MODEL_TYPE="Siglip2SoftCPT"
DATASETS=("AffectNet" "FairFace" "UTKFace" "Raf-DB" "LSW")  # List of datasets
BATCH_SIZE=32
BASE_DATASET_PATH="../processed_datasets/datasets_with_standard_labels"
BASE_OUTPUT_PATH="../TEST/Siglip2/SoftCPT/0001"
NUM_PROMPT=0
CKPT_DIR="../TRAIN/Siglip2/SoftCPT/0001/ckpt"
USE_TQDM="" # set to "--no_tqdm" to disable
PALIGEMMA="" # set to "--paligemma" to enable

# Loop through the datasets
for DATASET in "${DATASETS[@]}"; do
  # Construct dataset and output paths
  DATASET_PATH="${BASE_DATASET_PATH}/${DATASET}"
  OUTPUT_PATH="${BASE_OUTPUT_PATH}/${DATASET}"

  # Check if dataset path exists
  if [ ! -d "$DATASET_PATH" ]; then
    echo "Error: Dataset path '$DATASET_PATH' does not exist."
    continue  # Skip to the next dataset
  fi

  # Construct the command
  COMMAND="python3 test.py \
    --model_type \"${MODEL_TYPE}\" \
    --dataset_path \"${DATASET_PATH}\" \
    --batch_size ${BATCH_SIZE} \
    --output_path \"${OUTPUT_PATH}\" \
    ${USE_TQDM} \
    --num_prompt ${NUM_PROMPT} \
    --ckpt_dir \"${CKPT_DIR}\" \
    ${PALIGEMMA}"

  # Print the command (for debugging)
  echo "Running command for dataset '$DATASET'"

  # Execute the command
  eval $COMMAND
done

exit 0
