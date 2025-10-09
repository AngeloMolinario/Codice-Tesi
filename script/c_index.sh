datasets=("RAF-DB" "UTKFace" "FairFace" "VggFace2-Test")

for dataset in "${datasets[@]}"
do
    dataset_path="../processed_datasets/datasets_with_standard_labels/$dataset"

    python3 concordance_index_computation.py \
        --dataset_path "$dataset_path" \
        --ckpt_dir "../TRAIN/PECore/L14/SoftCPT/TSCA_cntx_15/ckpt" \
        --output_dir "../concordance_index" \
        --batch_size 64 \
        --num_workers 3 \
        --num_prompt 0

    python3 concordance_index_computation.py \
        --dataset_path "$dataset_path" \
        --ckpt_dir "../TRAIN/PECore/L14/SoftCPT/TSCA_cntx_20/ckpt" \
        --output_dir "../concordance_index" \
        --batch_size 64 \
        --num_workers 3 \
        --num_prompt 0

    python3 concordance_index_computation.py \
        --dataset_path "$dataset_path" \
        --ckpt_dir "../TRAIN/PECore/L14/SoftCPT/TSCA_cntx_25/ckpt" \
        --output_dir "../concordance_index" \
        --batch_size 64 \
        --num_workers 3 \
        --num_prompt 0

    python3 concordance_index_computation.py \
        --dataset_path "$dataset_path" \
        --ckpt_dir "../TRAIN/PECore/L14/SoftCPT/TSCA_cntx_15_vpt_10/ckpt" \
        --output_dir "../concordance_index" \
        --batch_size 64 \
        --num_workers 3 \
        --num_prompt 10

    python3 concordance_index_computation.py \
        --dataset_path "$dataset_path" \
        --ckpt_dir "../TRAIN/PECore/L14/SoftCPT/TSCA_cntx_20_vpt_10/ckpt" \
        --output_dir "../concordance_index" \
        --batch_size 64 \
        --num_workers 3 \
        --num_prompt 10

    python3 concordance_index_computation.py \
        --dataset_path "$dataset_path" \
        --ckpt_dir "../TRAIN/PECore/L14/SoftCPT/TSCA_cntx_25_vpt_10/ckpt" \
        --output_dir "../concordance_index" \
        --batch_size 64 \
        --num_workers 3 \
        --num_prompt 10
done