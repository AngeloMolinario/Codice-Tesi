# How to Use This Code

To use this code and reproduce the training or test results, please install the required packages.

```bash
pip install -r requirements.txt
```

## Dataset Preparation
All the datasets used for training and testing were preprocessed using the code available in the Dataset_preprocessing folder to produce centered cropped face images of a single person from each image.

(An exception is made for the Lagenda dataset because it already provides annotations and face locations for all faces in each image)

To obtain the cropped faces for each dataset, run the following Python script:
```bash
python3 dataset_processing.py --folder ../datasets_with_standard_labels/CelebA_HQ --output_dir ../processed_datasets --num_threads 4 --size 384
```

or for simplicity run the script process.sh that calls the Python script for all the datasets used.

After processing the datasets, each of them has been split into training and validation sets using a fixed seed and an 80/20 ratio. Also in this case a Python script is used for the splitting and is called only for the datasets used for training:
```bash
python3 dataset/split_csv.py ../processed_datasets/datasets_with_standard_labels/CelebA_HQ/train/labels.csv --train_ratio 0.8 --seed 2025 --rename_original_csv
```
Also in this case a bash script is provided to automatically call the script on all the datasets of interest:
```bash
./script/split_dataset.sh
```

Note: The uploaded ../processed_datasets.zip already contains the processed and split datasets.

## Start the Training

There are two training scripts that have been used to run the experiments: one for single-task training and one for multitask training. Both work in the same way: they take a JSON configuration file that contains all the information needed for training. Using the configuration files, it is possible to select the type of experiment to run and the hyperparameters.

For example, to run a single-task CoOp experiment with context length 15, the command is:
```bash
python3 coop_train.py ./config/coop/emotion/PE_coop_15.json
```
For the multitask training, run:
```bash
python3 train_multitask.py ./config/coop/emotion/PE_coop_15.json
```

All the experiments can be reproduced either by running the scripts independently for each configuration or by using the bash scripts in the script folder, which run the training and testing for particular single-task experiments or for multitask experiments. For example, to run all the experiments for the single task of emotion recognition with the two different approaches used (CoOp and VPT) you can simply run:
```bash
./script/coop_pe_emotion.sh
```

## Configuration

All the configuration files are stored in the [config/](config/) directory in JSON format. In particular, the configuration files for each task can be found in the [config/coop/<task_name>](config/coop/) folder, while the multitask configuration files can be found in the [config/](config/) folder.

### Configuration Parameters Reference

Below is a comprehensive list of all parameters that can be used in the configuration files:

#### Core Training Parameters

**`TUNING`** (string)
- **Description**: Type of tuning method to use. Options: `"softcpt"` (Soft Context Prompt Tuning), `"coop"` (Context Optimization)
- **How to switch from CoOp to Soft**: Change `"TUNING": "coop"` to `"TUNING": "softcpt"`
- **Example**: `"TUNING": "softcpt"`

**`MODEL`** (string)
- **Description**: Base model architecture to use
- **Example**: `"MODEL": "pecore"`

**`TASK`** (integer)
- **Description**: Task identifier. Use `-1` for multitask training, or specific task index (0, 1, 2, etc.) for single-task training
- **How to switch from multitask to single-task**: Change `"TASK": -1` to `"TASK": 0` (or 1, 2 for other tasks)
- **Example**: `"TASK": -1` (multitask) or `"TASK": 0` (single-task for first task)

**`MODEL_TYPE`** (string)
- **Description**: Specific model variant/configuration
- **Example**: `"MODEL_TYPE": "PE-Core-L14-336"`

#### Prompt Configuration

**`NUM_VISUAL_PROMPT`** (integer)
- **Description**: Number of visual prompt tokens to use in VPT (Visual Prompt Tuning). Set to `0` to disable VPT and use text-only prompting
- **How to switch from VPT to text-only**: Change `"NUM_VISUAL_PROMPT": 10` to `"NUM_VISUAL_PROMPT": 0`
- **Example**: `"NUM_VISUAL_PROMPT": 10` (VPT enabled) or `"NUM_VISUAL_PROMPT": 0` (text-only)

**`NUM_TEXT_CNTX`** (integer)
- **Description**: Number of text context tokens for prompt learning
- **Example**: `"NUM_TEXT_CNTX": 25`

**`TASK_NAMES`** (array of strings)
- **Description**: Natural language descriptions of each task
- **Example**: `["age estimation from face picture", "gender recognition from facial features", "emotion classification from facial expression"]`

**`CLASSES`** (array of arrays)
- **Description**: Class labels for each task. Each inner array corresponds to one task
- **Example**: 
```json
[
    ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"],
    ["male", "female"],
    ["surprise", "fear", "disgust", "happy", "sad", "angry", "neutral"]
]
```

**`NAMED_TRAINABLE_PARAMETERS`** (array of strings)
- **Description**: Names of model components that should be trainable during training
- **Example**: `["prompt_learner", "task_prompt_learner", "prompt_gen"]`

#### Dataset Configuration

**`DATASET_NAMES`** (object)
- **Description**: Mapping of task ID to dataset name(s). Use key `"-1"` with array of datasets for multitask, or specific task keys with single dataset for single-task
- **How to switch from multitask to single-task**: Change from `{"-1": ["Dataset1", "Dataset2"]}` to `{"0": "Dataset1"}` (and set `TASK` to 0)
- **Example**: 
  - Multitask: `{"-1": ["FairFace", "RAF-DB", "CelebA_HQ", "Lagenda"]}`
  - Single-task: `{"0": "FairFace"}`

**`DATASET_ROOT`** (string)
- **Description**: Root directory path where processed datasets are stored
- **Example**: `"DATASET_ROOT": "../processed_datasets/datasets_with_standard_labels"`

**`BALANCE_TASK`** (object)
- **Description**: Task specific ratio in the merged dataset. Keys are task IDs (as strings), values are target ratio.
- **Example**: `{"2": 0.33}` (We want task 2 to represent the 33% of the merged dataset)

#### Model Loading

**`PRETRAINED_CPT`** (string, optional)
- **Description**: Path to pretrained checkpoint file. Omit this parameter or remove it entirely to train from scratch
- **How to switch from pretrained to from-scratch**: Remove the `"PRETRAINED_CPT"` key from the JSON file
- **Example**: `"PRETRAINED_CPT": "../TRAIN/PECore/L14/SoftCPT/TSCA_cntx_25/ckpt/softCPT_tokens_bval.pt"`

#### Training Hyperparameters

**`BATCH_SIZE`** (integer)
- **Description**: Number of samples per training batch
- **Example**: `"BATCH_SIZE": 60`

**`LR`** (float)
- **Description**: Learning rate for the optimizer
- **Example**: `"LR": 0.002`

**`EPOCHS`** (integer)
- **Description**: Maximum number of training epochs
- **Example**: `"EPOCHS": 50`

**`PATIENCE`** (integer)
- **Description**: Number of epochs with no improvement after which training will be stopped (early stopping)
- **Example**: `"PATIENCE": 7`

#### Loss and Regularization

**`EMD_WEIGHT`** (float)
- **Description**: Weight for Earth Mover's Distance loss component
- **Example**: `"EMD_WEIGHT": 30`

**`EMD_OMEGA`** (float)
- **Description**: Omega parameter for EMD loss
- **Example**: `"EMD_OMEGA": 2.0`

**`EMD_MU`** (float)
- **Description**: Mu parameter for EMD loss
- **Example**: `"EMD_MU": -0.0025`

#### Data Loading

**`NUM_WORKERS`** (integer)
- **Description**: Number of subprocesses to use for data loading
- **Example**: `"NUM_WORKERS": 3`

**`PREFETCH_FACTOR`** (integer)
- **Description**: Number of batches loaded in advance by each worker
- **Example**: `"PREFETCH_FACTOR": 1`

#### Output and Logging

**`OUTPUT_DIR`** (string)
- **Description**: Directory path where training outputs (checkpoints, logs) will be saved
- **Example**: `"OUTPUT_DIR": "../TRAIN/PECore/L14/SoftCPT/TSCA_cntx_25_vpt_10"`

**`VERBOSE`** (boolean)
- **Description**: Enable verbose logging output
- **Example**: `"VERBOSE": true`

**`USE_TQDM`** (boolean)
- **Description**: Enable tqdm progress bars during training
- **Example**: `"USE_TQDM": false`

### Example Configuration for Multitask Training ([config/vpt_10_cn25.json](config/vpt_10_cn25.json))

```json
{
    "TUNING" : "softcpt",
    "MODEL"  : "pecore",
    "TASK"   : -1,
    "MODEL_TYPE" : "PE-Core-L14-336",

    "NUM_VISUAL_PROMPT" : 10,
    "NUM_TEXT_CNTX" : 25,
    "TASK_NAMES": ["age estimation from face picture", "gender recognition from facial features", "emotion classification from facial expression"], 
    "CLASSES": [
        ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"],
        ["male", "female"],
        ["surprise", "fear", "disgust", "happy", "sad", "angry", "neutral"]
    ],    
    "NAMED_TRAINABLE_PARAMETERS": [
        "prompt_learner",
        "task_prompt_learner", 
        "prompt_gen"
    ],

    "DATASET_NAMES": {
        "-1" : ["FairFace", "RAF-DB", "CelebA_HQ", "Lagenda"]     
    },

    "PRETRAINED_CPT" : "../TRAIN/PECore/L14/SoftCPT/TSCA_cntx_25/ckpt/softCPT_tokens_bval.pt",


    "CSP": false,
    "EMD_WEIGHT" : 30,
    "EMD_OMEGA" : 2.0,
    "EMD_MU": -0.0025,
    "DATASET_ROOT": "../processed_datasets/datasets_with_standard_labels",
    "BALANCE_TASK" : {"2" : 0.33},
    "BATCH_SIZE" : 60,
    "NUM_WORKERS" : 3,
    "LR" : 0.002,
    "OUTPUT_DIR" : "../TRAIN/PECore/L14/SoftCPT/TSCA_cntx_25_vpt_10",
    "PREFETCH_FACTOR" : 1,
    "VERBOSE": true,
    "USE_TQDM": false,
    "EPOCHS" : 50,
    "PATIENCE" : 7

}
```

### Quick Configuration Switching Guide

**To switch from VPT to text-only prompting:**
1. Set `"NUM_VISUAL_PROMPT": 0`
2. Remove the `"PRETRAINED_CPT"` key from the JSON file

**To switch from CoOp to Soft prompting:**
- Change `"TUNING": "coop"` to `"TUNING": "softcpt"`

**To switch from multitask to single-task training:**
1. Change `"TASK": -1` to `"TASK": 0` (or 1, 2 for different tasks)
2. Change `"DATASET_NAMES": {"-1": ["Dataset1", "Dataset2", ...]}` to `"DATASET_NAMES": {"0": "Dataset1"}`

**Example: Text-only Soft prompting configuration ([config/softpe_20.json](config/softpe_20.json))**
```json
{
    "TUNING" : "softcpt",
    "NUM_VISUAL_PROMPT" : 0,  // Text-only (no VPT)
    "NUM_TEXT_CNTX" : 20,
    // ... no PRETRAINED_CPT key means training from scratch and using only the text model for prompting the model
}
```

**Example: VPT with Soft prompting configuration ([config/vpt_10_cn25.json](config/vpt_10_cn25.json))**
```json
{
    "TUNING" : "softcpt",
    "NUM_VISUAL_PROMPT" : 10,  // VPT enabled
    "NUM_TEXT_CNTX" : 25,
    "PRETRAINED_CPT" : "../path/to/pretrained.pt",  // Load pretrained weights
    // ...
}
```