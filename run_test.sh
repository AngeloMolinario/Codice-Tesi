#! /bin/bash

# Define default values
DATASETS=("RAF-DB")  # List of datasets to test
BATCH_SIZE=50
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
echo "Datasets: ${DATASET_PATHS[*]}"
echo "Testing PECore BASE with VPT tokens..."
python3 test.py \
  --model_type "PECoreVPT" \
  --dataset_paths "${DATASET_PATHS[@]}" \
  --batch_size ${BATCH_SIZE} \
  --output_base_path "../TEST/PECore/L14/vpt/Emotion/" \
  --num_prompt 10 \
  --ckpt_dir "../TRAIN/PECore/L14/vpt/Emotion/ckpt" \
  --save_to_load "bval" \
  --pe_vision_config "PE-Core-L14-336" \

exit $?
