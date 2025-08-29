#!/bin/bash

# Script to run the test.py script with specific parameters

# Define default values
MODEL_TYPE="PECoreSoftCPT"
DATASETS=("AffectNet" "FairFace" "UTKFace" "Raf-DB" "CelebA_HQ" "LSW")  # List of datasets
DATASETS=("FairFace" "UTKFace")
BATCH_SIZE=256
BASE_DATASET_PATH="../processed_datasets/datasets_with_standard_labels"
BASE_OUTPUT_PATH="../TEST/Pali_age_PECore/SoftCPT/emd10_WSam_max_1_task_noAffectNet"
NUM_PROMPT=0
MODEL_CKPT_PATH=""
VPT_CKPT_PATH=""
TEXT_CKPT="../TRAIN/PECore/SoftCPT/emd10_WSam_max_1_task_noAffectNet/ckpt/text_features_bacc.pt"
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
    --model_ckpt_path \"${MODEL_CKPT_PATH}\" \
    --vpt_ckpt_path \"${VPT_CKPT_PATH}\" \
    --text_ckpt \"${TEXT_CKPT}\" \
    ${PALIGEMMA}\

  # Print the command (for debugging)
  echo "Running command for dataset '$DATASET': $COMMAND"

  # Execute the command
  eval $COMMAND
done

exit 0