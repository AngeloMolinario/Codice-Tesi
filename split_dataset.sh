#!/bin/bash

echo "Splitting CelebA_HQ"
python3 dataset/split_csv.py ../processed_datasets/datasets_with_standard_labels/CelebA_HQ/train/labels.csv --train_ratio 0.8 --seed 2025 --rename_original_csv
echo "Splitting FairFace"
python3 dataset/split_csv.py ../processed_datasets/datasets_with_standard_labels/FairFace/train/labels.csv --train_ratio 0.8 --seed 2025 --rename_original_csv
echo "Splitting RAF-DB"
python3 dataset/split_csv.py ../processed_datasets/datasets_with_standard_labels/RAF-DB/train/labels.csv --train_ratio 0.8 --seed 2025 --rename_original_csv
echo "Splitting Lagenda"
python3 dataset/split_csv.py ../processed_datasets/datasets_with_standard_labels/Lagenda/train/labels.csv --train_ratio 0.8 --seed 2025 --rename_original_csv

exit $?