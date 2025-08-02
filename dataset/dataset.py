import os
import torch
from torch import nn
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image

def map_age_to_group(age):
    """
    Map individual age values to age group categories.
    
    Age groups:
    - 0: 0-2 years
    - 1: 3-9 years  
    - 2: 10-19 years
    - 3: 20-29 years
    - 4: 30-39 years
    - 5: 40-49 years
    - 6: 50-59 years
    - 7: 60-69 years
    - 8: 70+ years
    
    Args:
        age: individual age value or age category string
        
    Returns:
        age group category (0-8) or -1 for missing values
    """
    if age == -1:  # Missing value
        return -1
    
    # Check if age is already a category (contains "-" or "more")
    if isinstance(age, str):
        age_str = str(age).lower()
        if "-" in age_str or "more" in age_str:
            # Age is already categorized, map to appropriate group
            if "0-2" == age_str:
                return 0
            elif "3-9" == age_str:
                return 1
            elif "10-19" == age_str:
                return 2
            elif "20-29" == age_str:
                return 3
            elif "30-39" == age_str:
                return 4
            elif "40-49" == age_str:
                return 5
            elif "50-59" == age_str:
                return 6
            elif "60-69" == age_str:
                return 7
            elif "70" in age_str or ("more" in age_str or "+" in age_str):
                return 8
            else:
                return -1  # Unknown category format
    
    # If age is numeric, perform standard mapping
    try:
        age_num = float(age)
        if 0 <= age_num <= 2:
            return 0
        elif 3 <= age_num <= 9:
            return 1
        elif 10 <= age_num <= 19:
            return 2
        elif 20 <= age_num <= 29:
            return 3
        elif 30 <= age_num <= 39:
            return 4
        elif 40 <= age_num <= 49:
            return 5
        elif 50 <= age_num <= 59:
            return 6
        elif 60 <= age_num <= 69:
            return 7
        elif age_num >= 70:
            return 8
        else:
            return -1  # Invalid age
    except (ValueError, TypeError):
        return -1  # Cannot convert to numeric

class BaseDataset(Dataset):
    def __init__(self, root:str, transform=None, split="train"):
        self.root = root
        self.transform = transform
        self.split = split
        self.base_root = root.split("datasets_with_standard_labels")[0]
        # Path to the split folder (train or test)
        self.split_path = os.path.join(root, split)
        
        # Path to images folder and labels CSV
        self.labels_path = os.path.join(self.split_path, "labels.csv")
        
        # Load the CSV file
        self.data = pd.read_csv(self.labels_path)
        
        # Extract relevant columns, handling missing values
        self.paths = self.data['Path'].values
        self.genders = self.data['Gender'].fillna(-1).values
        
        # Process ages: convert to age groups
        raw_ages = self.data['Age'].fillna(-1).values
        self.age_groups = [map_age_to_group(age) for age in raw_ages]
        self.emotions = self.data['Facial Emotion'].fillna(-1).values

    def __len__(self):
        # Return the number of images in the dataset
        return len(self.data)

    def __getitem__(self, idx):
        # Load an image and its corresponding label
        img_path = os.path.join(self.base_root, self.paths[idx].replace("\\","/")+".jpg")
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
        
        # Create label dictionary with all attributes (missing values are -1)
        # Age is now returned as age group instead of raw age
        label = {
            'gender': torch.tensor(self.genders[idx]),
            'age': torch.tensor(self.age_groups[idx]),
            'emotion': torch.tensor(self.emotions[idx])
        }
        
        return image, label


class MultiDataset(Dataset):
    def __init__(self, dataset_names, transform=None, split="train", datasets_root="datasets_with_standard_labels"):
        """
        Initialize MultiDataset with multiple datasets.
        
        Args:
            dataset_names (list): List of dataset names to load
            transform: Optional transforms to apply to images
            split (str): Split to use ("train" or "test")
            datasets_root (str): Root directory containing all datasets
        """
        self.dataset_names = dataset_names
        self.transform = transform
        self.split = split
        self.datasets_root = datasets_root
        
        # Load all datasets
        self.datasets = []
        self.dataset_lengths = []
        self.cumulative_lengths = [0]
        
        for dataset_name in dataset_names:
            dataset_path = os.path.join(datasets_root, dataset_name)
            try:
                dataset = BaseDataset(root=dataset_path, transform=transform, split=split)
                self.datasets.append(dataset)
                self.dataset_lengths.append(len(dataset))
                self.cumulative_lengths.append(self.cumulative_lengths[-1] + len(dataset))
            except Exception as e:
                print(f"Warning: Could not load dataset {dataset_name}: {e}")
                continue
        
        if not self.datasets:
            raise ValueError("No datasets could be loaded successfully")
        
        # Total length is the sum of all dataset lengths
        self.total_length = self.cumulative_lengths[-1]
        
        print(f"Loaded {len(self.datasets)} datasets with total {self.total_length} samples:")
        for i, name in enumerate([name for name in dataset_names if i < len(self.datasets)]):
            print(f"  - {name}: {self.dataset_lengths[i]} samples")

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        """
        Get item from the appropriate dataset based on the global index.
        
        Args:
            idx: Global index across all datasets
            
        Returns:
            tuple: (image, label) with the same format as BaseDataset
        """
        if idx >= self.total_length or idx < 0:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.total_length}")
        
        # Find which dataset this index belongs to
        dataset_idx = 0
        for i, cumulative_length in enumerate(self.cumulative_lengths[1:], 1):
            if idx < cumulative_length:
                dataset_idx = i - 1
                break
        
        # Calculate local index within the specific dataset
        local_idx = idx - self.cumulative_lengths[dataset_idx]
        
        # Get the sample from the appropriate dataset
        return self.datasets[dataset_idx][local_idx]
    
    def get_dataset_info(self):
        """
        Get information about the loaded datasets.
        
        Returns:
            dict: Information about each dataset including name, length, and cumulative ranges
        """
        info = {}
        for i, name in enumerate(self.dataset_names[:len(self.datasets)]):
            info[name] = {
                'length': self.dataset_lengths[i],
                'start_idx': self.cumulative_lengths[i],
                'end_idx': self.cumulative_lengths[i + 1] - 1
            }
        return info



if __name__ == "__main__":
    # Example usage
    dataset = BaseDataset(
        root="./datasets_with_standard_labels/UTKFace",
        transform=None,  # Add any transforms if needed
        split="test"
    )
    
    print(f"Total samples in multi-dataset: {len(dataset)}")
    for i in range(len(dataset)):
        image, label = dataset[i]
        print(f"Sample {i}: Image shape , Label {label}")
        if i == 50:
            break