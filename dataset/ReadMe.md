# Intelligent CSV Splitter for Stratified Datasets

This script provides an advanced method for splitting a CSV dataset into training and validation sets. It is designed to handle complex stratification scenarios where samples may have multiple labels. The primary goal is to ensure that the validation set is a true representation of the data distribution, especially for rare label combinations.

## Splitting Logic

The script employs a multi-step stratification strategy to ensure a high-quality split:

1.  **Identify Label Combinations**: The script first reads all label columns (every column except the first one). For each row (sample), it creates a unique "label combination" tuple. For example, if a row has `Gender='Male'` and `Emotion='Happy'`, its combination is `('Male', 'Happy')`.

2.  **Guaranteed Validation Set Coverage**: To prevent any specific data category from being absent in the validation set, the script guarantees its representation. For every unique label combination that appears more than once in the dataset, it moves **one** sample to a preliminary validation set. This ensures that the model will be validated on all types of data present in the dataset.

3.  **Fill to Ratio**: After guaranteeing coverage, the script calculates how many more samples are needed to meet the desired validation set size (defined by `train_ratio`). It then randomly selects the required number of samples from the remaining pool and adds them to the validation set.

4.  **Special Handling of 'Age' Column**:
    *   If a column named `Age` is present and contains data, the script temporarily maps the numerical age values into categorical groups (e.g., `25` becomes `20-29`).
    *   This mapping is used **only** for the stratification process to ensure a balanced distribution across different age groups.
    *   Before saving the final `train.csv` and `validation.csv` files, the script **restores the original age values**. This means your output data remains unchanged, while the split benefits from the intelligent grouping.

## CSV File Structure

For the script to work correctly, your input CSV file must follow this structure:

*   **Comma-separated** values.
*   The **first column** must be a unique identifier for the sample, such as a file path (e.g., `image_path`).
*   **All subsequent columns** are treated as labels for stratification.
*   The file must have a header row.

#### Example `input.csv`:

```csv
path,Age,Gender,Emotion
data/img_001.jpg,25,Male,Happy
data/img_002.jpg,42,Female,Sad
data/img_003.jpg,,Male,Neutral
data/img_004.jpg,7,Female,Happy
```

## Usage

You can run the script from the command line.

#### Basic Command:

```bash
python split_csv.py path/to/your/input.csv
```

#### Command with Options:

You can customize the behavior using the following arguments:

*   `input_csv`: (Required) The path to the input CSV file.
*   `--output_dir`: The directory where the output `train` and `val` folders will be saved. The script attempts to infer this from the input path but can be set manually.
*   `--train_ratio`: The proportion of the dataset to allocate to the training set. Defaults to `0.8` (i.e., 80% train, 20% validation).
*   `--seed`: A random seed for reproducibility. Defaults to `42`.

#### Full Example:

This command splits `dataset.csv` into an 85% training set and a 15% validation set, using a specific random seed.

```bash
python split_csv.py C:\Users\user\data\dataset.csv --train_ratio 0.85 --seed 123
```

## Output

The script will create two subdirectories, `train` and `val`, inside the determined output directory.

```
<output_dir>/
├── train/
│   └── train.csv
└── val/
    └── validation.csv
```

It also prints a detailed summary of the split, including the final class distribution for each label in both the training and validation sets.