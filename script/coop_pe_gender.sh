#!/bin/bash

echo "#######################################################################"
echo "##                  TRAINING GENDER  ON COOP 15                      ##"
echo "#######################################################################"
python3 coop_train.py config/coop/gender/PE_coop_15.json

echo "#######################################################################"
echo "##                  TRAINING GENDER  ON COOP 20                      ##"
echo "#######################################################################"
python3 coop_train.py config/coop/gender/PE_coop_20.json

echo "#######################################################################"
echo "##                  TRAINING GENDER  ON COOP 25                      ##"
echo "#######################################################################"
python3 coop_train.py config/coop/gender/PE_coop_25.json


echo "#######################################################################"
echo "##                  TRAINING GENDER ON VPT 10 CNTX 15                ##"
echo "#######################################################################"
python3 coop_train.py config/coop/gender/PE_vpt_10_cn15.json

echo "#######################################################################"
echo "##                  TRAINING GENDER ON VPT 10 CNTX 20                ##"
echo "#######################################################################"
python3 coop_train.py config/coop/gender/PE_vpt_10_cn20.json

echo "#######################################################################"
echo "##                  TRAINING GENDER ON VPT 10 CNTX 25                ##"
echo "#######################################################################"
python3 coop_train.py config/coop/gender/PE_vpt_10_cn25.json

echo "#######################################################################"
echo "##                        TRAINING FINISHED                          ##"
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
                    --ckpt_dir "../TRAIN/PECore/L14/coop/Gender_cntx_15/ckpt/" \
                    --batch_size 32 --no_tqdm
echo "#######################################################################"
python3 test.py --model_type "PECore" \
                    --num_prompt 0 \
                    --dataset_path "../processed_datasets/datasets_with_standard_labels/RAF-DB" \
                    "../processed_datasets/datasets_with_standard_labels/UTKFace" \
                    "../processed_datasets/datasets_with_standard_labels/FairFace" \
                    "../processed_datasets/datasets_with_standard_labels/CelebA_HQ" \
                    "../processed_datasets/datasets_with_standard_labels/VggFace2-Test" \
                    --ckpt_dir "../TRAIN/PECore/L14/coop/Gender_cntx_20/ckpt/" \
                    --batch_size 32 --no_tqdm
echo "#######################################################################"
python3 test.py --model_type "PECore" \
                    --num_prompt 0 \
                    --dataset_path "../processed_datasets/datasets_with_standard_labels/RAF-DB" \
                    "../processed_datasets/datasets_with_standard_labels/UTKFace" \
                    "../processed_datasets/datasets_with_standard_labels/FairFace" \
                    "../processed_datasets/datasets_with_standard_labels/CelebA_HQ" \
                    "../processed_datasets/datasets_with_standard_labels/VggFace2-Test" \
                    --ckpt_dir "../TRAIN/PECore/L14/coop/Gender_cntx_25/ckpt/" \
                    --batch_size 32 --no_tqdm
echo "#######################################################################"
python3 test.py --model_type "PECore" \
                    --num_prompt 10 \
                    --dataset_path "../processed_datasets/datasets_with_standard_labels/RAF-DB" \
                    "../processed_datasets/datasets_with_standard_labels/UTKFace" \
                    "../processed_datasets/datasets_with_standard_labels/FairFace" \
                    "../processed_datasets/datasets_with_standard_labels/CelebA_HQ" \
                    "../processed_datasets/datasets_with_standard_labels/VggFace2-Test" \
                    --ckpt_dir "../TRAIN/PECore/L14/vpt/Gender_cntx_15_vpt_10/ckpt/" \
                    --batch_size 23 --no_tqdm
echo "#######################################################################"
python3 test.py --model_type "PECore" \
                    --num_prompt 10 \
                    --dataset_path "../processed_datasets/datasets_with_standard_labels/RAF-DB" \
                    "../processed_datasets/datasets_with_standard_labels/UTKFace" \
                    "../processed_datasets/datasets_with_standard_labels/FairFace" \
                    "../processed_datasets/datasets_with_standard_labels/CelebA_HQ" \
                    "../processed_datasets/datasets_with_standard_labels/VggFace2-Test" \
                    --ckpt_dir "../TRAIN/PECore/L14/vpt/Gender_cntx_20_vpt_10/ckpt/" \
                    --batch_size 32 --no_tqdm
echo "#######################################################################"
python3 test.py --model_type "PECore" \
                    --num_prompt 10 \
                    --dataset_path "../processed_datasets/datasets_with_standard_labels/RAF-DB" \
                    "../processed_datasets/datasets_with_standard_labels/UTKFace" \
                    "../processed_datasets/datasets_with_standard_labels/FairFace" \
                    "../processed_datasets/datasets_with_standard_labels/CelebA_HQ" \
                    "../processed_datasets/datasets_with_standard_labels/VggFace2-Test" \
                    --ckpt_dir "../TRAIN/PECore/L14/vpt/Gender_cntx_25_vpt_10/ckpt/" \
                    --batch_size 32 --no_tqdm