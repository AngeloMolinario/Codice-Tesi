#!/bin/bash

# Script to run the test.py script with specific parameters

# Define default values
MODEL_TYPE="Siglip2SoftCPT"
DATASETS=("FairFace" "UTKFace" "RAF-DB" "LFW" "CelebA_HQ")  # List of datasets
BATCH_SIZE=64
BASE_DATASET_PATH="../processed_datasets/datasets_with_standard_labels"
BASE_OUTPUT_PATH="../TEST/Siglip2/SoftCPT/00000001_class_specific"
NUM_PROMPT=0
CKPT_DIR="../TRAIN/Siglip2/SoftCPT/00000001_class_specific/ckpt"
USE_TQDM="" # set to "--no_tqdm" to disable
PALIGEMMA="" # set to "--paligemma" to enable

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

# Execute a single command with multiple datasets; the Python script
# will create one output subfolder per dataset under BASE_OUTPUT_PATH

echo "Testing PECoreVPT"
python3 test.py \
  --model_type "PECoreVPT" \
  --dataset_paths "${DATASET_PATHS[@]}" \
  --batch_size ${BATCH_SIZE} \
  --output_base_path "../TEST/PECore/VPT/text_features_TSCA" \
  --num_prompt 70 \
  --ckpt_dir "../TRAIN/PECore/VPT/text_features_TSCA/ckpt" 

exit $?

echo "Testing iglip2_VPT"
python3 test.py \
  --model_type "Siglip2VPT" \
  --dataset_paths "${DATASET_PATHS[@]}" \
  --batch_size ${BATCH_SIZE} \
  --output_base_path "../TEST/Siglip2/VPT/text_features_TSCA" \
  --num_prompt 70 \
  --ckpt_dir "../TRAIN/Siglip2/VPT/text_features_TSCA/ckpt" 
  
echo "Testing SoftSiglip2_TSCA"
python3 test.py \
  --model_type "Siglip2SoftCPT" \
  --dataset_paths "${DATASET_PATHS[@]}" \
  --batch_size ${BATCH_SIZE} \
  --output_base_path "../TEST/Siglip2/SoftCPT/TSCA" \
  --num_prompt 0 \
  --ckpt_dir "../TRAIN/Siglip2/SoftCPT/TSCA/ckpt" 
  
echo "Testing SoftSiglip2_TSCS"
python3 test.py \
  --model_type "Siglip2SoftCPT" \
  --dataset_paths "${DATASET_PATHS[@]}" \
  --batch_size ${BATCH_SIZE} \
  --output_base_path "../TEST/Siglip2/SoftCPT/TSCS" \
  --num_prompt 0 \
  --ckpt_dir "../TRAIN/Siglip2/SoftCPT/TSCS/ckpt"
 
echo "Testing SoftPECore_TSCS"
python3 test.py \
  --model_type "PECoreSoftCPT" \
  --dataset_paths "${DATASET_PATHS[@]}" \
  --batch_size ${BATCH_SIZE} \
  --output_base_path "../TEST/PECore/SoftCPT/TSCS" \
  --num_prompt 0 \
  --ckpt_dir "../TRAIN/PECore/SoftCPT/TSCS/ckpt" 

echo "Testing SoftPECore_TSCA"
python3 test.py \
  --model_type "PECoreSoftCPT" \
  --dataset_paths "${DATASET_PATHS[@]}" \
  --batch_size ${BATCH_SIZE} \
  --output_base_path "../TEST/PECore/SoftCPT/TSCA" \
  --num_prompt 0 \
  --ckpt_dir "../TRAIN/PECore/SoftCPT/TSCA/ckpt" 



exit $?