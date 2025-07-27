import os
import torch
from torch import nn
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image

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
        self.ages = self.data['Age'].fillna(-1).values
        self.ethnicities = self.data['Ethnicity'].fillna(-1).values
        self.emotions = self.data['Facial Emotion'].fillna(-1).values
        self.identities = self.data['Identity'].fillna(-1).values

    def __len__(self):
        # Return the number of images in the dataset
        return len(self.data)

    def __getitem__(self, idx):
        # Load an image and its corresponding label
        img_path = self.paths[idx]+".jpg"
        print(f"Loading image from: {img_path}")
        image = Image.open(img_path).convert('RGB')
        
        # Apply tra nsforms if provided
        if self.transform:
            image = self.transform(image)
        
        # Create label dictionary with all attributes (missing values are -1)
        label = {
            'gender': self.genders[idx],
            'age': self.ages[idx],
            'ethnicity': self.ethnicities[idx],
            'emotion': self.emotions[idx],
            'identity': self.identities[idx]
        }
        
        return image, label

