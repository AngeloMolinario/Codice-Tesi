echo "TESTING SoftCPT/TSCA/cntx 15"
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
echo "TESTING SoftCPT/TSCA/cntx 20"
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
echo "TESTING SoftCPT/TSCA/cntx 25"
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
echo "Testing coop/emotion 15"
python3 test_new.py --model_type "PECoreVPT" \
                    --num_prompt 0 \
                    --dataset_path "../processed_datasets/datasets_with_standard_labels/RAF-DB" \
                    --ckpt_dir "../TRAIN/PECore/L14/coop/Emotion_cntx_15/ckpt/" \
                    --batch_size 128 --no_tqdm
echo "#######################################################################"
echo "Testing coop/emotion 20"
python3 test_new.py --model_type "PECoreVPT" \
                    --num_prompt 0 \
                    --dataset_path "../processed_datasets/datasets_with_standard_labels/RAF-DB" \
                    --ckpt_dir "../TRAIN/PECore/L14/coop/Emotion_cntx_20/ckpt/" \
                    --batch_size 128 --no_tqdm
echo "#######################################################################"
echo "Testing coop/emotion 25"
python3 test_new.py --model_type "PECoreVPT" \
                    --num_prompt 0 \
                    --dataset_path "../processed_datasets/datasets_with_standard_labels/RAF-DB" \
                    --ckpt_dir "../TRAIN/PECore/L14/coop/Emotion_cntx_25/ckpt/" \
                    --batch_size 128 --no_tqdm
echo "#######################################################################"
echo "Testing vpt/emotion 15"
python3 test_new.py --model_type "PECoreVPT" \
                    --num_prompt 10 \
                    --dataset_path "../processed_datasets/datasets_with_standard_labels/RAF-DB" \
                    --ckpt_dir "../TRAIN/PECore/L14/vpt/Emotion_cntx_15_vpt_10/ckpt/" \
                    --batch_size 128 --no_tqdm
echo "#######################################################################"
echo "Testing vpt/emotion 20"
python3 test_new.py --model_type "PECoreVPT" \
                    --num_prompt 10 \
                    --dataset_path "../processed_datasets/datasets_with_standard_labels/RAF-DB" \
                    --ckpt_dir "../TRAIN/PECore/L14/vpt/Emotion_cntx_20_vpt_10/ckpt/" \
                    --batch_size 128 --no_tqdm
echo "#######################################################################"
echo "Testing vpt/emotion 25"
python3 test_new.py --model_type "PECoreVPT" \
                    --num_prompt 10 \
                    --dataset_path "../processed_datasets/datasets_with_standard_labels/RAF-DB" \
                    --ckpt_dir "../TRAIN/PECore/L14/vpt/Emotion_cntx_25_vpt_10/ckpt/" \
                    --batch_size 128 --no_tqdm
echo "#######################################################################"
echo "Testing coop/age 15"
python3 test_new.py --model_type "PECoreVPT" \
                    --num_prompt 0 \
                    --dataset_path "../processed_datasets/datasets_with_standard_labels/UTKFace" \
                    "../processed_datasets/datasets_with_standard_labels/FairFace" \
                    "../processed_datasets/datasets_with_standard_labels/VggFace2-Test" \
                    --ckpt_dir "../TRAIN/PECore/L14/coop/Age_cntx_15/ckpt/" \
                    --batch_size 128 --no_tqdm
echo "#######################################################################"
echo "Testing coop/gender 15"
python3 test_new.py --model_type "PECoreVPT" \
                    --num_prompt 0 \
                    --dataset_path "../processed_datasets/datasets_with_standard_labels/RAF-DB" \
                    "../processed_datasets/datasets_with_standard_labels/UTKFace" \
                    "../processed_datasets/datasets_with_standard_labels/FairFace" \
                    "../processed_datasets/datasets_with_standard_labels/CelebA_HQ" \
                    "../processed_datasets/datasets_with_standard_labels/VggFace2-Test" \
                    --ckpt_dir "../TRAIN/PECore/L14/coop/Gender_cntx_15/ckpt/" \
                    --batch_size 128 --no_tqdm
echo "#######################################################################"
echo "Testing coop/gender 20"
python3 test_new.py --model_type "PECoreVPT" \
                    --num_prompt 0 \
                    --dataset_path "../processed_datasets/datasets_with_standard_labels/RAF-DB" \
                    "../processed_datasets/datasets_with_standard_labels/UTKFace" \
                    "../processed_datasets/datasets_with_standard_labels/FairFace" \
                    "../processed_datasets/datasets_with_standard_labels/CelebA_HQ" \
                    "../processed_datasets/datasets_with_standard_labels/VggFace2-Test" \
                    --ckpt_dir "../TRAIN/PECore/L14/coop/Gender_cntx_20/ckpt/" \
                    --batch_size 128 --no_tqdm
echo "#######################################################################"
echo "Testing coop/gender 25"
python3 test_new.py --model_type "PECoreVPT" \
                    --num_prompt 0 \
                    --dataset_path "../processed_datasets/datasets_with_standard_labels/RAF-DB" \
                    "../processed_datasets/datasets_with_standard_labels/UTKFace" \
                    "../processed_datasets/datasets_with_standard_labels/FairFace" \
                    "../processed_datasets/datasets_with_standard_labels/CelebA_HQ" \
                    "../processed_datasets/datasets_with_standard_labels/VggFace2-Test" \
                    --ckpt_dir "../TRAIN/PECore/L14/coop/Gender_cntx_25/ckpt/" \
                    --batch_size 128 --no_tqdm