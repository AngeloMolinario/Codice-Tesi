#!/bin/bash

python3 dataset/split_csv.py ../processed_datasets/datasets_with_standard_labels/CelebA_HQ/train/labels.csv --train_ratio 0.8 --seed 2025 --rename_original_csv
python3 dataset/split_csv.py ../processed_datasets/datasets_with_standard_labels/FairFace/train/labels.csv --train_ratio 0.8 --seed 2025 --rename_original_csv
python3 dataset/split_csv.py ../processed_datasets/datasets_with_standard_labels/MiviaGender/train/labels.csv --train_ratio 0.8 --seed 2025 --rename_original_csv
python3 dataset/split_csv.py ../processed_datasets/datasets_with_standard_labels/RAF-DB/train/labels.csv --train_ratio 0.8 --seed 2025 --rename_original_csv
python3 dataset/split_csv.py ../processed_datasets/datasets_with_standard_labels/TestDataset/train/labels.csv --train_ratio 0.8 --seed 2025 --rename_original_csv
python3 dataset/split_csv.py ../processed_datasets/datasets_with_standard_labels/VggFace2/train/labels.csv --train_ratio 0.8 --seed 2025 --rename_original_csv