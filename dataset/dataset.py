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
            if "0-2" in age_str:
                return 0
            elif "3-9" in age_str:
                return 1
            elif "10-19" in age_str:
                return 2
            elif "20-29" in age_str:
                return 3
            elif "30-39" in age_str:
                return 4
            elif "40-49" in age_str:
                return 5
            elif "50-59" in age_str:
                return 6
            elif "60-69" in age_str:
                return 7
            elif "70" in age_str and ("more" in age_str or "+" in age_str):
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
        self.identities = self.data['Identity'].fillna(-1).values

    def __len__(self):
        # Return the number of images in the dataset
        return len(self.data)

    def __getitem__(self, idx):
        # Load an image and its corresponding label
        img_path = self.paths[idx]+".jpg"
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
        
        # Create label dictionary with all attributes (missing values are -1)
        # Age is now returned as age group instead of raw age
        label = {
            'gender': self.genders[idx],
            'age': self.age_groups[idx],  # Now returns age group (0-8)
            'emotion': self.emotions[idx],
            'identity': self.identities[idx]
        }
        
        return image, label

