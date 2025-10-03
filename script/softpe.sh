#!/bin/bash

echo "#######################################################################"
echo "##                  TRAINING MLTK ON CONTX 15                        ##"
echo "#######################################################################"
python3 train_multitask.py config/softpe_15.json

echo "#######################################################################"
echo "##                  TRAINING MLTK ON CONTX 20                        ##"
echo "#######################################################################"
python3 train_multitask.py config/softpe_20.json

echo "#######################################################################"
echo "##                  TRAINING MLTK ON CONTX 25                        ##"
echo "#######################################################################"
python3 train_multitask.py config/softpe_25.json

echo "#######################################################################"
echo "##                  TRAINING MLTK ON VPT 10 CN 15                    ##"
echo "#######################################################################"
python3 train_multitask.py config/vpt_10_cn15.json

echo "#######################################################################"
echo "##                  TRAINING MLTK ON VPT 10 CN 20                    ##"
echo "#######################################################################"
python3 train_multitask.py config/vpt_10_cn20.json

echo "#######################################################################"
echo "##                  TRAINING MLTK ON VPT 10 CN 25                    ##"
echo "#######################################################################"
python3 train_multitask.py config/vpt_10_cn25.json

echo "#######################################################################"
echo "##                        TRAINING COMPLETED                         ##"
echo "#######################################################################"

echo "#######################################################################"
echo "##                        START TESTING                             ##"
echo "#######################################################################"
python3 test.py --model_type "PECore" \
                    --num_prompt 0 \
                    --dataset_path "../processed_datasets/datasets_with_standard_labels/RAF-DB" \
                    "../processed_datasets/datasets_with_standard_labels/UTKFace" \
                    "../processed_datasets/datasets_with_standard_labels/FairFace" \
                    "../processed_datasets/datasets_with_standard_labels/CelebA_HQ" \
                    "../processed_datasets/datasets_with_standard_labels/VggFace2-Test" \
                    --ckpt_dir "../TRAIN/PECore/L14/SoftCPT/TSCA_cntx_15/ckpt/" \
                    --batch_size 32 --no_tqdm
echo "#######################################################################"
python3 test.py --model_type "PECore" \
                    --num_prompt 0 \
                    --dataset_path "../processed_datasets/datasets_with_standard_labels/RAF-DB" \
                    "../processed_datasets/datasets_with_standard_labels/UTKFace" \
                    "../processed_datasets/datasets_with_standard_labels/FairFace" \
                    "../processed_datasets/datasets_with_standard_labels/CelebA_HQ" \
                    "../processed_datasets/datasets_with_standard_labels/VggFace2-Test" \
                    --ckpt_dir "../TRAIN/PECore/L14/SoftCPT/TSCA_cntx_20/ckpt/" \
                    --batch_size 32 --no_tqdm
echo "#######################################################################"
python3 test.py --model_type "PECore" \
                    --num_prompt 0 \
                    --dataset_path "../processed_datasets/datasets_with_standard_labels/RAF-DB" \
                    "../processed_datasets/datasets_with_standard_labels/UTKFace" \
                    "../processed_datasets/datasets_with_standard_labels/FairFace" \
                    "../processed_datasets/datasets_with_standard_labels/CelebA_HQ" \
                    "../processed_datasets/datasets_with_standard_labels/VggFace2-Test" \
                    --ckpt_dir "../TRAIN/PECore/L14/SoftCPT/TSCA_cntx_25/ckpt/" \
                    --batch_size 32 --no_tqdm
echo "#######################################################################"
python3 test.py --model_type "PECore" \
                    --num_prompt 10 \
                    --dataset_path "../processed_datasets/datasets_with_standard_labels/RAF-DB" \
                    "../processed_datasets/datasets_with_standard_labels/UTKFace" \
                    "../processed_datasets/datasets_with_standard_labels/FairFace" \
                    "../processed_datasets/datasets_with_standard_labels/CelebA_HQ" \
                    "../processed_datasets/datasets_with_standard_labels/VggFace2-Test" \
                    --ckpt_dir "../TRAIN/PECore/L14/SoftCPT/TSCA_cntx_15_vpt_10/ckpt/" \
                    --batch_size 32 --no_tqdm
echo "#######################################################################"
python3 test.py --model_type "PECore" \
                    --num_prompt 10 \
                    --dataset_path "../processed_datasets/datasets_with_standard_labels/RAF-DB" \
                    "../processed_datasets/datasets_with_standard_labels/UTKFace" \
                    "../processed_datasets/datasets_with_standard_labels/FairFace" \
                    "../processed_datasets/datasets_with_standard_labels/CelebA_HQ" \
                    "../processed_datasets/datasets_with_standard_labels/VggFace2-Test" \
                    --ckpt_dir "../TRAIN/PECore/L14/SoftCPT/TSCA_cntx_20_vpt_10/ckpt/" \
                    --batch_size 32 --no_tqdm
echo "#######################################################################"
python3 test.py --model_type "PECore" \
                    --num_prompt 10 \
                    --dataset_path "../processed_datasets/datasets_with_standard_labels/RAF-DB" \
                    "../processed_datasets/datasets_with_standard_labels/UTKFace" \
                    "../processed_datasets/datasets_with_standard_labels/FairFace" \
                    "../processed_datasets/datasets_with_standard_labels/CelebA_HQ" \
                    "../processed_datasets/datasets_with_standard_labels/VggFace2-Test" \
                    --ckpt_dir "../TRAIN/PECore/L14/SoftCPT/TSCA_cntx_25_vpt_10/ckpt/" \
                    --batch_size 32 --no_tqdm