#!/bin/bash

# Define default values
DATASETS=("UTKFace" "FairFace" "RAF-DB" "LFW" "CelebA_HQ" "VggFace2-Test")  # List of datasets to test
BATCH_SIZE=256
BASE_DATASET_PATH="../processed_datasets/datasets_with_standard_labels"

# Build list of valid dataset paths
DATASET_PATHS=()
for DATASET in "${DATASETS[@]}"; do
  DATASET_PATH="${BASE_DATASET_PATH}/${DATASET}"
  if [ -d "$DATASET_PATH" ]; then
    DATASET_PATHS+=("$DATASET_PATH")
  else
    echo "Warning: Dataset path '$DATASET_PATH' does not exist. Skipping."
  fi
done

if [ ${#DATASET_PATHS[@]} -eq 0 ]; then
  echo "Error: no valid dataset paths found. Exiting."
  exit 1
fi

echo "Running a single test invocation over ${#DATASET_PATHS[@]} datasets..."

echo "Testing PECORE VPT_TSCA_augment_batch100"
python3 test.py \
  --model_type "PECoreVPT" \
  --dataset_paths "${DATASET_PATHS[@]}" \
  --batch_size ${BATCH_SIZE} \
  --output_base_path "../TEST/PECore/VPT/text_features_TSCA_batch100" \
  --num_prompt 70 \
  --ckpt_dir "../TRAIN/PECore/VPT/text_features_TSCA_batch100/ckpt" 

echo "Testing Siglip2 VPT_TSCA_augment_batch100"
python3 test.py \
  --model_type "Siglip2VPT" \
  --dataset_paths "${DATASET_PATHS[@]}" \
  --batch_size ${BATCH_SIZE} \
  --output_base_path "../TEST/Siglip2/VPT/text_features_TSCA_batch100" \
  --num_prompt 70 \
  --ckpt_dir "../TRAIN/Siglip2/VPT/text_features_TSCA_batch100/ckpt"

exit $?
