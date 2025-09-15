#!/bin/bash

echo "#######################################################################"
echo "##                  TRAINING EMOTION ON COOP 15                      ##"
echo "#######################################################################"
python3 train_multitask.py config/softpe_15.json

echo "#######################################################################"
echo "##                  TRAINING EMOTION ON COOP 20                      ##"
echo "#######################################################################"
python3 train_multitask.py config/softpe_20.json

echo "#######################################################################"
echo "##                  TRAINING EMOTION ON COOP 25                      ##"
echo "#######################################################################"
python3 train_multitask.py config/softpe_25.json

echo "#######################################################################"
echo "##                        TRAINING COMPLETED                         ##"
echo "#######################################################################"

echo "#######################################################################"
echo "##                        START TESTING                             ##"
echo "#######################################################################"
python3 test_new.py --model_type "PECoreSoftCPT" \
                    --num_prompt 0 \
                    --dataset_path "../processed_datasets/datasets_with_standard_labels/RAF-DB" \
                    "../processed_datasets/datasets_with_standard_labels/UTKFace" \
                    "../processed_datasets/datasets_with_standard_labels/FairFace" \
                    "../processed_datasets/datasets_with_standard_labels/CelebA_HQ" \
                    "../processed_datasets/datasets_with_standard_labels/VggFace2-Test" \
                    --ckpt_dir "../TRAIN/PECore/L14/SoftCPT/TSCA_cntx_15/ckpt/" \
                    --batch_size 128 --no_tqdm
echo "#######################################################################"
python3 test_new.py --model_type "PECoreSoftCPT" \
                    --num_prompt 0 \
                    --dataset_path "../processed_datasets/datasets_with_standard_labels/RAF-DB" \
                    "../processed_datasets/datasets_with_standard_labels/UTKFace" \
                    "../processed_datasets/datasets_with_standard_labels/FairFace" \
                    "../processed_datasets/datasets_with_standard_labels/CelebA_HQ" \
                    "../processed_datasets/datasets_with_standard_labels/VggFace2-Test" \
                    --ckpt_dir "../TRAIN/PECore/L14/SoftCPT/TSCA_cntx_20/ckpt/" \
                    --batch_size 128 --no_tqdm
echo "#######################################################################"
python3 test_new.py --model_type "PECoreSoftCPT" \
                    --num_prompt 0 \
                    --dataset_path "../processed_datasets/datasets_with_standard_labels/RAF-DB" \
                    "../processed_datasets/datasets_with_standard_labels/UTKFace" \
                    "../processed_datasets/datasets_with_standard_labels/FairFace" \
                    "../processed_datasets/datasets_with_standard_labels/CelebA_HQ" \
                    "../processed_datasets/datasets_with_standard_labels/VggFace2-Test" \
                    --ckpt_dir "../TRAIN/PECore/L14/SoftCPT/TSCA_cntx_25/ckpt/" \
                    --batch_size 128 --no_tqdm
echo "#######################################################################"