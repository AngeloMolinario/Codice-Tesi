from dataset import  MultiDataset
from tqdm import tqdm
import os
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed


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
        if 0 <= age_num < 3:
            return 0
        elif 3 <= age_num < 10:
            return 1
        elif 10 <= age_num < 20:
            return 2
        elif 20 <= age_num < 30:
            return 3
        elif 30 <= age_num < 40:
            return 4
        elif 40 <= age_num < 50:
            return 5
        elif 50 <= age_num < 60:
            return 6
        elif 60 <= age_num < 70:
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
        
        # Create label dictionary with all attributes (missing values are -1)
        # Age is now returned as age group instead of raw age
        label = {
            'gender': torch.tensor(self.genders[idx]),
            'age': torch.tensor(self.age_groups[idx]),
            'emotion': torch.tensor(self.emotions[idx])
        }
        
        return None, label

class MultiDataset(Dataset):
    def __init__(self, dataset_names, transform=None, split="train", datasets_root="datasets_with_standard_labels", all_datasets=False):
        """
        Initialize MultiDataset with multiple datasets.
        
        Args:
            dataset_names (list): List of dataset names to load
            transform: Optional transforms to apply to images
            split (str): Split to use ("train" or "test")
            datasets_root (str): Root directory containing all datasets
            all_datasets: (bool): if True, load all datasets in the root directory and ignore dataset_names
        """
        self.dataset_names = dataset_names
        self.transform = transform
        self.split = split
        self.datasets_root = datasets_root
        
        if all_datasets:
            # If all_datasets is True, load all datasets in the root directory
            self.dataset_names = [d for d in os.listdir(datasets_root) if os.path.isdir(os.path.join(datasets_root, d))]
            print(f"Loading all datasets: {self.dataset_names}")

        # Load all datasets
        self.datasets = []
        self.dataset_lengths = []
        self.cumulative_lengths = [0]
        successfully_loaded_datasets = []
        
        for dataset_name in self.dataset_names:
            dataset_path = os.path.join(datasets_root, dataset_name)
            if os.path.exists(os.path.join(dataset_path, split)):    
                try:
                    dataset = BaseDataset(root=dataset_path, transform=transform, split=split)
                    self.datasets.append(dataset)
                    self.dataset_lengths.append(len(dataset))
                    self.cumulative_lengths.append(self.cumulative_lengths[-1] + len(dataset))
                    successfully_loaded_datasets.append(dataset_name)
                    print(f"Loaded {len(dataset)} samples from {dataset_path}/{split}")
                except Exception as e:
                    print(f"Warning: Could not load dataset {dataset_name}: {e}")
            else:
                print(f"Warning: Split '{split}' not found in {dataset_path}")
        
        # Update dataset_names to only include successfully loaded datasets
        self.dataset_names = successfully_loaded_datasets                           
        
        if not self.datasets:
            raise ValueError("No datasets could be loaded successfully")
        
        # Total length is the sum of all dataset lengths
        self.total_length = self.cumulative_lengths[-1]
        
        print(f"Loaded {len(self.datasets)} datasets with total {self.total_length} samples:")
        for i, dataset_name in enumerate(self.dataset_names):
            if i < len(self.datasets):
                print(f"  - {dataset_name}: {self.dataset_lengths[i]} samples")

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

def print_utils(labels, str1, dic1):
    print(f"\nTotal {labels} labels: {str1}\nDistribution:")
    for key, value in dic1.items():
        print(f"\t- {key}: {value}")

def process_dataset_chunk(dataset, start_idx, end_idx, thread_id):
    """Process a chunk of the dataset and return counts and distributions"""
    gender_count = 0
    age_count = 0
    emotion_count = 0
    
    age_distribution = {}
    gender_distribution = {}
    emotion_distribution = {}
    
    # Create progress bar for this thread
    chunk_size = end_idx - start_idx
    pbar = tqdm(range(start_idx, end_idx), 
                desc=f"Thread {thread_id}", 
                position=thread_id, 
                leave=False)
    
    for i in pbar:
        _, labels = dataset[i]
        
        # Convert tensors to Python values
        age_val = labels['age'].item() if hasattr(labels['age'], 'item') else int(labels['age'])
        gender_val = labels['gender'].item() if hasattr(labels['gender'], 'item') else int(labels['gender'])
        emotion_val = labels['emotion'].item() if hasattr(labels['emotion'], 'item') else int(labels['emotion'])
        
        if age_val != -1:
            age_count += 1
            age_distribution[age_val] = age_distribution.get(age_val, 0) + 1
        if gender_val != -1:
            gender_count += 1
            gender_distribution[gender_val] = gender_distribution.get(gender_val, 0) + 1
        if emotion_val != -1:
            emotion_count += 1
            emotion_distribution[emotion_val] = emotion_distribution.get(emotion_val, 0) + 1
    
    return {
        'age_count': age_count,
        'gender_count': gender_count,
        'emotion_count': emotion_count,
        'age_distribution': age_distribution,
        'gender_distribution': gender_distribution,
        'emotion_distribution': emotion_distribution
    }

def merge_distributions(dist1, dist2):
    """Merge two distribution dictionaries"""
    merged = dist1.copy()
    for key, value in dist2.items():
        merged[key] = merged.get(key, 0) + value
    return merged

def compute_inverse_frequency(total, class_count, per_class_count):
    """Compute inverse frequency for a class"""
    if class_count == 0:
        return 0
    return total / (class_count * per_class_count)

def compute_task_weights(age_counts, gender_counts, emotion_counts):

    inverse_weight = [1/age_counts , 1/gender_counts, 1/emotion_counts]
    sum_weights = sum(inverse_weight)
    normalized_weights = [w / sum_weights for w in inverse_weight]
    return normalized_weights

if __name__ == "__main__":
    NUM_THREADS = 2  # Adjust this number based on your system
    
    datasetName = "FairFace"
    root_path = "./../processed_datasets/datasets_with_standard_labels/"

    dataset = MultiDataset(
        dataset_names=[datasetName],
        transform=None,
        split="train",
        datasets_root=root_path,
        all_datasets=True
    )

    dataset_size = len(dataset)
    chunk_size = dataset_size // NUM_THREADS
    
    # Create thread pool and submit tasks
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = []
        
        for i in range(NUM_THREADS):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size if i < NUM_THREADS - 1 else dataset_size
            
            future = executor.submit(process_dataset_chunk, dataset, start_idx, end_idx, i)
            futures.append(future)
        
        # Collect results
        results = []
        for future in as_completed(futures):
            results.append(future.result())
    
    # Aggregate results
    total_gender = 0
    total_age = 0
    total_emotion = 0
    
    final_age_distribution = {}
    final_gender_distribution = {}
    final_emotion_distribution = {}
    
    for result in results:
        total_age += result['age_count']
        total_gender += result['gender_count']
        total_emotion += result['emotion_count']
        
        final_age_distribution = merge_distributions(final_age_distribution, result['age_distribution'])
        final_gender_distribution = merge_distributions(final_gender_distribution, result['gender_distribution'])
        final_emotion_distribution = merge_distributions(final_emotion_distribution, result['emotion_distribution'])
    
    age_class_count = len(final_age_distribution)
    gender_class_count = len(final_gender_distribution)
    emotion_class_count = len(final_emotion_distribution)

    age_class_weight = {}
    gender_class_weight = {}
    emotion_class_weight = {}

    for key in final_age_distribution:
        age_class_weight[key] = compute_inverse_frequency(total_age, age_class_count, final_age_distribution[key])

    for key in final_gender_distribution:
        gender_class_weight[key] = compute_inverse_frequency(total_gender, gender_class_count, final_gender_distribution[key])

    for key in final_emotion_distribution:
        emotion_class_weight[key] = compute_inverse_frequency(total_emotion, emotion_class_count, final_emotion_distribution[key])
    # Print results
    print("#"*30)
    print(f"\nTotal images: {dataset_size}\n")
    print("#"*30, "AGE", "#"*30)
    print_utils("age", total_age, final_age_distribution)
    print("Class weights for age distribution:")
    for key, value in age_class_weight.items():
        print(f"\t- {key}: {value:.4f}")
    print()
    print("#"*30, "GENDER", "#"*30)
    print_utils("gender", total_gender, final_gender_distribution)
    print("Class weights for gender distribution:")
    for key, value in gender_class_weight.items():
        print(f"\t- {key}: {value:.4f}")
    print()
    
    print("#"*30, "EMOTION", "#"*30)
    print_utils("emotion", total_emotion, final_emotion_distribution)
    print("Class weights for emotion distribution:")
    for key, value in sorted(emotion_class_weight.items()):
        print(f"\t- {key}: {value:.4f}")
    print()

    print("#"*30, "TASK WEIGHTS", "#"*30)
    task_weights = compute_task_weights(total_age, total_gender, total_emotion)
    print(f"Age task weight: {task_weights[0]:.4f}")
    print(f"Gender task weight: {task_weights[1]:.4f}")
    print(f"Emotion task weight: {task_weights[2]:.4f}")


    report = {
        "analysis_info": {
            "total_images": dataset_size,
            "num_threads_used": NUM_THREADS,
            "datasets_used": dataset.dataset_names,
            "split": "train"
        },
        "dataset_info": dataset.get_dataset_info(),
        "age_analysis": {
            "total_samples_with_age": total_age,
            "num_classes": age_class_count,
            "distribution": final_age_distribution,
            "class_weights": age_class_weight,
            "task_weight": task_weights[0]
        },
        "gender_analysis": {
            "total_samples_with_gender": total_gender,
            "num_classes": gender_class_count,
            "distribution": final_gender_distribution,
            "class_weights": gender_class_weight,
            "task_weight": task_weights[1]
        },
        "emotion_analysis": {
            "total_samples_with_emotion": total_emotion,
            "num_classes": emotion_class_count,
            "distribution": final_emotion_distribution,
            "class_weights": emotion_class_weight,
            "task_weight": task_weights[2]
        },
        "task_weights": {
            "age": task_weights[0],
            "gender": task_weights[1], 
            "emotion": task_weights[2]
        }
    }
    
    # Save JSON report
    report_filename = f"dataset_analysis_report.json"
    with open(report_filename, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    