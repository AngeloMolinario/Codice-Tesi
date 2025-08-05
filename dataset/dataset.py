import os
import torch
from torch import nn
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
import random

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
        
        # Round to nearest integer if decimal >= 0.5
        age_rounded = round(age_num)
        
        if 0 <= age_rounded <= 2:
            return 0
        elif 3 <= age_rounded <= 9:
            return 1
        elif 10 <= age_rounded <= 19:
            return 2
        elif 20 <= age_rounded <= 29:
            return 3
        elif 30 <= age_rounded <= 39:
            return 4
        elif 40 <= age_rounded <= 49:
            return 5
        elif 50 <= age_rounded <= 59:
            return 6
        elif 60 <= age_rounded <= 69:
            return 7
        elif age_rounded >= 70:
            return 8
        else:
            print(f"Warning: Age {age_num} (rounded to {age_rounded}) is out of expected range")
            return -1  # Invalid age
            
    except (ValueError, TypeError) as e:
        print(f"Warning: Cannot convert age '{age}' to float: {e}")
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
        raw_paths = self.data['Path'].values
        # Preprocessa tutti i path una sola volta
        self.img_paths = [os.path.join(self.base_root, path.replace("\\","/")+".jpg") for path in raw_paths]
        
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
        img_path = self.img_paths[idx]  # Path già preprocessato
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
        
        # Create label dictionary with all attributes (missing values are -1)
        # Age is now returned as age group instead of raw age
        label = {
            'gender': torch.tensor(self.genders[idx], dtype=torch.long),
            'age': torch.tensor(self.age_groups[idx], dtype=torch.long),
            'emotion': torch.tensor(self.emotions[idx], dtype=torch.long)
        }
        
        return image, label

    def get_class_weights(self, task):
        """
        Compute class weights for the specified task for this dataset only.
        
        Args:
            task (str): Task name ('age', 'gender', or 'emotion')
            
        Returns:
            torch.Tensor: Class weights tensor
        """
        if not hasattr(self, 'class_weights'):
            self.class_weights = {}

        if self.class_weights.get(task, None) is not None:
            return self.class_weights[task]
            
        print(f"Computing class weights for task: {task} (BaseDataset)")

        # Aggregate data from this dataset only
        if task == 'age':
            all_task_data = self.age_groups
        elif task == 'gender':
            all_task_data = self.genders
        elif task == 'emotion':
            all_task_data = self.emotions
        else:
            raise ValueError(f"Unknown task: {task}. Must be one of 'age', 'gender', 'emotion'")
        
        # Count occurrences of each class (excluding -1 values)
        class_counts = {}
        total_valid_samples = 0
        
        for value in all_task_data:
            if value != -1:  # Exclude missing values
                class_counts[value] = class_counts.get(value, 0) + 1
                total_valid_samples += 1
        
        if total_valid_samples == 0:
            raise ValueError(f"No valid samples found for task {task}")
        
        # Get all class indices and sort them
        class_indices = sorted(class_counts.keys())
        
        weights_array = []
        for class_idx in class_indices:
            count = class_counts[class_idx]
            weight = total_valid_samples / (len(class_counts) * count)
            weights_array.append(weight)
        
        weights_tensor = torch.tensor(weights_array, dtype=torch.float32)
        self.class_weights[task] = weights_tensor
        
        print(f"Class distribution for {task}: {dict(sorted(class_counts.items()))}")
        print(f"Class weights for {task}: {weights_tensor}")
        
        return weights_tensor

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
    
    def get_dataset_info(self, compute_stats=False):
        """
        Get information about the loaded datasets.
        
        Args:
            compute_stats (bool): If True, compute label distributions and missing value counts (expensive).
        
        Returns:
            dict: Information about each dataset including name, length, cumulative ranges, and optionally stats.
        """
        info = {}
        for i, name in enumerate(self.dataset_names[:len(self.datasets)]):
            dataset_info = {
                'length': self.dataset_lengths[i],
                'start_idx': self.cumulative_lengths[i],
                'end_idx': self.cumulative_lengths[i + 1] - 1
            }
            if compute_stats:
                ds = self.datasets[i]
                stats = {}
                # Label distributions (excluding missing values)
                for task in ['age', 'gender', 'emotion']:
                    if task == 'age':
                        data = ds.age_groups
                    elif task == 'gender':
                        data = ds.genders
                    elif task == 'emotion':
                        data = ds.emotions
                    else:
                        continue
                    valid_data = [v for v in data if v != -1]
                    if valid_data:
                        unique, counts = np.unique(valid_data, return_counts=True)
                        stats[f'{task}_distribution'] = dict(zip(unique.tolist(), counts.tolist()))
                        stats[f'{task}_missing'] = int(len(data) - len(valid_data))
                    else:
                        stats[f'{task}_distribution'] = {}
                        stats[f'{task}_missing'] = len(data)
                dataset_info['stats'] = stats
            info[name] = dataset_info
        return info
    
    def get_class_weights(self, task):
        """
        Compute class weights for the specified task across all datasets.
        
        Args:
            task (str): Task name ('age', 'gender', or 'emotion')
            
        Returns:
            torch.Tensor: Class weights tensor
        """
        if not hasattr(self, 'class_weights'):
            self.class_weights = {}

        if self.class_weights.get(task, None) is not None:
            return self.class_weights[task]
            
        print(f"Computing class weights for task: {task}")
        
        # Aggregate data from all datasets
        if task == 'age':
            all_task_data = []
            for dataset in self.datasets:
                all_task_data.extend(dataset.age_groups)
        elif task == 'gender':
            all_task_data = []
            for dataset in self.datasets:
                all_task_data.extend(dataset.genders)
        elif task == 'emotion':
            all_task_data = []
            for dataset in self.datasets:
                all_task_data.extend(dataset.emotions)
        else:
            raise ValueError(f"Unknown task: {task}. Must be one of 'age', 'gender', 'emotion'")
        
        # Count occurrences of each class (excluding -1 values)
        class_counts = {}
        total_valid_samples = 0
        
        for value in all_task_data:
            if value != -1:  # Exclude missing values
                class_counts[value] = class_counts.get(value, 0) + 1
                total_valid_samples += 1
        
        if total_valid_samples == 0:
            raise ValueError(f"No valid samples found for task {task}")
        
        # Get all class indices and sort them
        class_indices = sorted(class_counts.keys())
        
        # Compute inverse frequency weights in order
        weights_array = []
        for class_idx in class_indices:
            count = class_counts[class_idx]
            # Inverse frequency: total_samples / (num_classes * samples_per_class)
            weight = total_valid_samples / (len(class_counts) * count)
            weights_array.append(weight)
        
        # Store and return as tensor
        weights_tensor = torch.tensor(weights_array, dtype=torch.float32)
        self.class_weights[task] = weights_tensor
        
        print(f"Class distribution for {task}: {dict(sorted(class_counts.items()))}")
        print(f"Class weights for {task}: {weights_tensor}")
        
        return weights_tensor    


class BalancedIdxDataset(Dataset):
    """
    A dataset wrapper that provides access only to a balanced subset of samples
    defined by a list of indices, as sampled from a MultiDataset.

    Args:
        multidataset (MultiDataset): The source dataset.
        num_samples_per_class (int): Number of samples per class for balancing.
        ignore_indices (List[int], optional): List of global indices to ignore during sampling.

    Attributes:
        dataset (MultiDataset): Reference to the source dataset.
        balanced_indices (List[int]): List of global indices for balanced sampling.
        class_weights_cache (dict): Cached class weights per task.
        task_weights_cache (torch.Tensor or None): Cached task weights tensor.
    """
    def __init__(self, multidataset, num_samples_per_class, ignore_indices=None):
        self.dataset = multidataset
        self.ignore_indices = set(ignore_indices) if ignore_indices is not None else set()
        self.balanced_indices = self._sample_balanced_indices(num_samples_per_class)
        self.class_weights_cache = {}
        self.task_weights_cache = None

    def _sample_balanced_indices(self, num_samples_per_class):
        used_indices = set(self.ignore_indices)
        merged_indices = []

        for task in ['age', 'gender', 'emotion']:
            valid_class_indices = dict()
            for d_idx, dataset in enumerate(self.dataset.datasets):
                if task == 'age':
                    labels = dataset.age_groups
                elif task == 'gender':
                    labels = dataset.genders
                elif task == 'emotion':
                    labels = dataset.emotions
                else:
                    raise ValueError(f"Unknown task: {task}")
                for l_idx, label in enumerate(labels):
                    if label != -1:
                        global_idx = self.dataset.cumulative_lengths[d_idx] + l_idx
                        if label not in valid_class_indices:
                            valid_class_indices[label] = []
                        valid_class_indices[label].append(global_idx)
            for class_value in sorted(valid_class_indices.keys()):
                candidates = [idx for idx in valid_class_indices[class_value] if idx not in used_indices]
                random.shuffle(candidates)
                chosen = candidates[:min(num_samples_per_class, len(candidates))]
                merged_indices.extend(chosen)
                used_indices.update(chosen)
        return merged_indices

    def __len__(self):
        return len(self.balanced_indices)

    def __getitem__(self, idx):
        global_idx = self.balanced_indices[idx]
        return self.dataset[global_idx]

    def get_class_weights(self, task):
        """
        Compute class weights for the specified task for the balanced dataset only.
        Weights are computed and cached the first time the function is called.

        Args:
            task (str): Task name ('age', 'gender', or 'emotion')

        Returns:
            torch.Tensor: Class weights tensor
        """
        if task in self.class_weights_cache:
            return self.class_weights_cache[task]

        print(f"Computing class weights for task: {task} (BalancedIdxDataset)")

        all_task_data = []
        for idx in self.balanced_indices:
            d_idx = 0
            for i, cum_len in enumerate(self.dataset.cumulative_lengths[1:], 1):
                if idx < cum_len:
                    d_idx = i - 1
                    break
            local_idx = idx - self.dataset.cumulative_lengths[d_idx]
            ds = self.dataset.datasets[d_idx]
            if task == 'age':
                label = ds.age_groups[local_idx]
            elif task == 'gender':
                label = ds.genders[local_idx]
            elif task == 'emotion':
                label = ds.emotions[local_idx]
            else:
                raise ValueError(f"Unknown task: {task}")
            if label != -1:
                all_task_data.append(label)

        class_counts = {}
        total_valid_samples = 0

        for value in all_task_data:
            class_counts[value] = class_counts.get(value, 0) + 1
            total_valid_samples += 1

        if total_valid_samples == 0:
            raise ValueError(f"No valid samples found for task {task} in the balanced subset")

        class_indices = sorted(class_counts.keys())
        weights_array = []
        for class_idx in class_indices:
            count = class_counts[class_idx]
            weight = total_valid_samples / (len(class_counts) * count)
            weights_array.append(weight)

        weights_tensor = torch.tensor(weights_array, dtype=torch.float32)
        self.class_weights_cache[task] = weights_tensor

        print(f"Class distribution for {task}: {dict(sorted(class_counts.items()))}")
        print(f"Class weights for {task}: {weights_tensor}")

        return weights_tensor

    def get_task_weights(self):
        """
        Compute normalized task weights for the balanced dataset only.
        Weights are computed and cached the first time the function is called.
        Task weights are calculated as inverse frequency across tasks
        ('age', 'gender', 'emotion') based on the number of valid samples in the subset,
        and then normalized so their sum is 1.

        Returns:
            torch.Tensor: Normalized task weights tensor of shape (3,)
        """
        if self.task_weights_cache is not None:
            return self.task_weights_cache

        sample_counts = []
        for task in ['age', 'gender', 'emotion']:
            count = 0
            for idx in self.balanced_indices:
                d_idx = 0
                for i, cum_len in enumerate(self.dataset.cumulative_lengths[1:], 1):
                    if idx < cum_len:
                        d_idx = i - 1
                        break
                local_idx = idx - self.dataset.cumulative_lengths[d_idx]
                ds = self.dataset.datasets[d_idx]
                if task == 'age':
                    label = ds.age_groups[local_idx]
                elif task == 'gender':
                    label = ds.genders[local_idx]
                elif task == 'emotion':
                    label = ds.emotions[local_idx]
                else:
                    raise ValueError(f"Unknown task: {task}")
                if label != -1:
                    count += 1
            sample_counts.append(count)
        sample_counts = torch.tensor(sample_counts, dtype=torch.float32)
        # Avoid division by zero for missing tasks
        raw_weights = 1.0 / torch.where(sample_counts == 0, torch.ones_like(sample_counts), sample_counts)
        # Normalize so sum is 1
        normalized_weights = raw_weights / raw_weights.sum()
        self.task_weights_cache = normalized_weights
        print(f"Normalized task weights: {normalized_weights}")
        return normalized_weights

if __name__ == "__main__":

    def compute_class_and_task_weights_from_indices(dataset, idx_list):
        """
        Given a dataset and a list of indices, compute:
        - For each task ('age', 'gender', 'emotion'), the class weights for the subset.
        - For each task, the class distribution for the subset.
        - The task weights, based on the number of (non-missing) samples per task.

        Returns:
            class_weights: dict { 'age': Tensor, 'gender': Tensor, 'emotion': Tensor }
            task_weights: Tensor of shape (3,) corresponding to ['age', 'gender', 'emotion']
            class_distributions: dict { 'age': {class: count, ...}, ... }
        """
        task_labels = { 'age': [], 'gender': [], 'emotion': [] }
        class_distributions = { 'age': {}, 'gender': {}, 'emotion': {} }
        for idx in idx_list:
            if isinstance(dataset, MultiDataset):
                d_idx = 0
                for i, cum_len in enumerate(dataset.cumulative_lengths[1:], 1):
                    if idx < cum_len:
                        d_idx = i - 1
                        break
                local_idx = idx - dataset.cumulative_lengths[d_idx]
                ds = dataset.datasets[d_idx]
                age = ds.age_groups[local_idx]
                gender = ds.genders[local_idx]
                emotion = ds.emotions[local_idx]
            else:  # BaseDataset
                age = dataset.age_groups[idx]
                gender = dataset.genders[idx]
                emotion = dataset.emotions[idx]
            if age != -1:
                task_labels['age'].append(age)
                class_distributions['age'][age] = class_distributions['age'].get(age, 0) + 1
            if gender != -1:
                task_labels['gender'].append(gender)
                class_distributions['gender'][gender] = class_distributions['gender'].get(gender, 0) + 1
            if emotion != -1:
                task_labels['emotion'].append(emotion)
                class_distributions['emotion'][emotion] = class_distributions['emotion'].get(emotion, 0) + 1

        class_weights = {}
        sample_counts = []

        for task in ['age', 'gender', 'emotion']:
            labels = task_labels[task]
            class_counts = class_distributions[task]
            total_valid_samples = len(labels)
            sample_counts.append(total_valid_samples)
            if total_valid_samples == 0 or len(class_counts) == 0:
                weights_tensor = torch.tensor([], dtype=torch.float32)
            else:
                class_indices = sorted(class_counts.keys())
                weights_array = []
                for class_idx in class_indices:
                    count = class_counts[class_idx]
                    weight = total_valid_samples / (len(class_counts) * count)
                    weights_array.append(weight)
                weights_tensor = torch.tensor(weights_array, dtype=torch.float32)
            class_weights[task] = weights_tensor

        # Task weights: inverse frequency (tasks with fewer samples have higher weight)
        sample_counts = torch.tensor(sample_counts, dtype=torch.float32)
        task_weights = sample_counts.sum() / (len(sample_counts) * sample_counts)

        return class_weights, task_weights, class_distributions
    # Example usage
    dataset = BaseDataset(
        root="../processed_datasets/datasets_with_standard_labels/UTKFace",
        transform=None,  # Add any transforms if needed
        split="test"
    )
    
    print(f"Total samples in multi-dataset: {len(dataset)}")
    for i in range(len(dataset)):
        image, label = dataset[i]
        print(f"Sample {i}: Image shape , Label {label}")
        if i == 5:
            break

    dataset = MultiDataset(
        dataset_names=["FairFace", "CelebA_HQ", "RAF-DB"],  # Adjust these names as needed
        transform=None,  # Add any transforms if needed
        split="train",
        datasets_root="../processed_datasets/datasets_with_standard_labels",
        all_datasets=True  # Set to True to load all datasets in the root directory
    )
    print(f"MultiDataset loaded successfully! Total samples: {len(dataset)}")
    for i in range(len(dataset)):
        image, label = dataset[i]
        print(f"Sample {i}: Label {label}")
        if i == 5:
            break
    
    print("Dataset information:")
    try:
        info = dataset.get_dataset_info(compute_stats=True)
        for name, details in info.items():
            print(f"Dataset: {name}")
            print(f"  Number of samples: {details['length']}")
            print(f"  Global index range: [{details['start_idx']} - {details['end_idx']}]")
            stats = details.get('stats', {})
            for task in ['age', 'gender', 'emotion']:
                dist = stats.get(f"{task}_distribution", {})
                missing = stats.get(f"{task}_missing", None)
                print(f"  {task.capitalize()} statistics:")
                if dist:
                    print(f"    Distribution:")
                    for k, v in dist.items():
                        print(f"      Class {k}: {v} samples")
                else:
                    print("    Distribution: (none)")
                print(f"    Missing: {missing} samples")
            print()
    except Exception as e:
        print(f"Error loading MultiDataset: {e}")


    print("\n\n\n")
    print("TESTING BALANCED SAMPLING")
    merged_indices = dataset.sample_balanced_indices(num_samples_per_class=1000)
    class_weights, task_weights, class_distributions  = compute_class_and_task_weights_from_indices(dataset, merged_indices)
    print("=== Balanced Sampling Summary ===")
    for task in ['age', 'gender', 'emotion']:
        print(f"\nTask: {task.capitalize()}")
        dist = class_distributions.get(task, {})
        if dist:
            print("  Class Distribution:")
            for k, v in dist.items():
                print(f"    Class {k}: {v} samples")
        else:
            print("  Class Distribution: (none)")
        weights = class_weights.get(task)
        if weights is not None and len(weights) > 0:
            print(f"  Class Weights: {weights.numpy()}")
        else:
            print("  Class Weights: N/A")
    print("\nTask Weights: ", task_weights.numpy())
    print("===============================")



    # Example of using BalancedIdxDataset
    balanced_dataset = BalancedIdxDataset(dataset, num_samples_per_class=10000)
    print(f"BalancedIdxDataset created with {len(balanced_dataset)} samples")
    for i in range(len(balanced_dataset)):
        image, label = balanced_dataset[i]
        print(f"Balanced Sample {i}: Label {label}")
        if i == 5:
            break

    # Example of getting class weights from BalancedIdxDataset
    try:
        class_weights = balanced_dataset.get_class_weights('age')
        print(f"Class weights for 'age': {class_weights.numpy()}")
        class_weights = balanced_dataset.get_class_weights('emotion')
        print(f"Class weights for 'emotion': {class_weights.numpy()}")
        class_weights = balanced_dataset.get_class_weights('gender')
        print(f"Class weights for 'gender': {class_weights.numpy()}")
        print("Class weights computed successfully for all tasks.")
        print("\n\n")
        print(f"Task weights: {balanced_dataset.get_task_weights().numpy()}")
    except Exception as e:
        print(f"Error getting class weights: {e}")
