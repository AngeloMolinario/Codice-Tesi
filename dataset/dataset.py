import os
import torch
from torch import nn
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
import random
import torchvision.transforms as tform
import torchvision.transforms as transforms

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
    def __init__(self, root:str, transform=None, split="train", verbose=False):
        self.verbose = verbose
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
        return len(self.data)

    def __getitem__(self, idx):
        '''
        Get item by index.
        Return a tuple (image, label) where:
        - image is a tensor of shape (3, H, W) after applying transforms
        - label is a list of tensors with attributes:
                 0: age group (0-8)
                 1: gender (0-1)
                 2: emotion (0-6)
        '''
        # Load an image and its corresponding label
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)        

        # Create label array with all attributes (missing values are -1)
        # Age is now returned as age group instead of raw age
        label = [
            torch.tensor(self.age_groups[idx], dtype=torch.long),
            torch.tensor(self.genders[idx], dtype=torch.long),            
            torch.tensor(self.emotions[idx], dtype=torch.long)
        ]
        
        
        return image, label

    def get_class_weights(self, task):
        """
        Compute class weights for the specified task for this dataset only.
        
        Args:
            task (str): Task name ('age', 'gender', or 'emotion')
            
        Returns:
            torch.Tensor: Class weights tensor
        """

        # Check if class weights have already been computed
        if not hasattr(self, 'class_weights'):
            self.class_weights = {}

        if self.class_weights.get(task, None) is not None:
            return self.class_weights[task]


        if self.verbose:    
            print(f"Computing class weights for task: {task} (BaseDataset)")

        # Get the relevant task data for which to compute weights
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
        

        # Compute the inverse frequency weights for the task's classes
        # Weights are computed as total_samples / (num_classes * samples_per_class)
        weights_array = []
        for class_idx in class_indices:
            count = class_counts[class_idx]
            weight = total_valid_samples / (len(class_counts) * count)
            weights_array.append(weight)
        

        weights_tensor = torch.tensor(weights_array, dtype=torch.float32)
        self.class_weights[task] = weights_tensor
        
        if self.verbose:
            print(f"Class distribution for {task}: {dict(sorted(class_counts.items()))}")
            print(f"Class weights for {task}: {weights_tensor}")
        
        return weights_tensor

class MultiDataset(Dataset):
    def __init__(self, dataset_names, transform=None, split="train", datasets_root="datasets_with_standard_labels", all_datasets=False, verbose=False):
        """
        Initialize MultiDataset with multiple datasets.
        
        Args:
            dataset_names (list): List of dataset names to load
            transform: Optional transforms to apply to images
            split (str): Split to use ("train" or "test")
            datasets_root (str): Root directory containing all datasets
            all_datasets: (bool): if True, load all datasets in the root directory and ignore dataset_names
            verbose (bool): if True, print detailed information about the loading process
        """
        self.dataset_names = dataset_names
        self.transform = transform
        self.split = split
        self.datasets_root = datasets_root
        self.verbose = verbose
        
        if all_datasets:
            # If all_datasets is True, load all datasets in the root directory
            self.dataset_names = [d for d in os.listdir(datasets_root) if os.path.isdir(os.path.join(datasets_root, d))]
            if self.verbose:
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
                    # If the path to the split exists, load the dataset using the BaseDataset class
                    dataset = BaseDataset(root=dataset_path, transform=transform, split=split)
                    self.datasets.append(dataset)
                    self.dataset_lengths.append(len(dataset))
                    self.cumulative_lengths.append(self.cumulative_lengths[-1] + len(dataset))
                    successfully_loaded_datasets.append(dataset_name)
                    if self.verbose:
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
        
        if self.verbose:
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
        # Check if the index is within the valid range
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
        Utility function to get information about the loaded dataset.
        
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
        
        if self.verbose:
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
        
        if self.verbose:
            print(f"Class distribution for {task}: {dict(sorted(class_counts.items()))}")
            print(f"Class weights for {task}: {weights_tensor}")
        
        return weights_tensor
    
    def get_task_weights(self):
        """
        Compute task weights for multitask learning based on the number of valid samples per task.
        
        Returns:
            torch.Tensor: Task weights tensor [age_weight, gender_weight, emotion_weight]
        """
        if hasattr(self, 'task_weights') and self.task_weights is not None:
            return self.task_weights
        
        # Count valid samples for each task across all datasets
        task_counts = {'age': 0, 'gender': 0, 'emotion': 0}
        
        for dataset in self.datasets:
            # Count valid samples (non-missing labels) for each task
            task_counts['age'] += sum(1 for age in dataset.age_groups if age != -1)
            task_counts['gender'] += sum(1 for gender in dataset.genders if gender != -1)
            task_counts['emotion'] += sum(1 for emotion in dataset.emotions if emotion != -1)
        
        # Calculate inverse frequency weights
        total_samples = self.total_length
        weights = []
        
        for task in ['age', 'gender', 'emotion']:
            if task_counts[task] > 0:
                # Inverse frequency: total_samples / task_samples
                weight = total_samples / task_counts[task]
            else:
                weight = 1.0  # Default weight if no samples
            weights.append(weight)
        
        # Normalize weights so they sum to 3 (number of tasks)
        weights_tensor = torch.tensor(weights, dtype=torch.float32)
        weights_tensor = weights_tensor * 3.0 / weights_tensor.sum()
        
        self.task_weights = weights_tensor
        
        if self.verbose:
            print(f"Task sample counts: {task_counts}")
            print(f"Task weights: {weights_tensor}")
        
        return weights_tensor

class TaskBalanceDataset(Dataset):
    def __init__(self, dataset_names, transform=None, split="train", 
                 datasets_root="datasets_with_standard_labels", all_datasets=False, 
                 verbose=False, balance_task=None):
        """
        Initialize TaskBalanceDataset with multiple datasets and optional task balancing.
        
        Args:
            dataset_names (list): List of dataset names to load
            transform: Optional transforms to apply to images
            split (str): Split to use ("train" or "test")
            datasets_root (str): Root directory containing all datasets
            all_datasets (bool): if True, load all datasets in the root directory and ignore dataset_names
            verbose (bool): if True, print detailed information about the loading process
            balance_task (dict): task_name -> desired_fraction (e.g. {"emotion": 0.25})
        """
        self.dataset_names = dataset_names
        self.transform = transform
        self.split = split
        self.datasets_root = datasets_root
        self.verbose = verbose
        self.balance_task = balance_task  # e.g. {"emotion": 0.25}
        
        if all_datasets:
            # If all_datasets is True, load all datasets in the root directory
            self.dataset_names = [d for d in os.listdir(datasets_root) if os.path.isdir(os.path.join(datasets_root, d))]
            if self.verbose:
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
                    # If the path to the split exists, load the dataset using the BaseDataset class
                    dataset = BaseDataset(root=dataset_path, transform=transform, split=split)
                    self.datasets.append(dataset)
                    self.dataset_lengths.append(len(dataset))
                    self.cumulative_lengths.append(self.cumulative_lengths[-1] + len(dataset))
                    successfully_loaded_datasets.append(dataset_name)
                    if self.verbose:
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
        
        if self.verbose:
            print(f"Loaded {len(self.datasets)} datasets with total {self.total_length} samples:")
            for i, dataset_name in enumerate(self.dataset_names):
                if i < len(self.datasets):
                    print(f"  - {dataset_name}: {self.dataset_lengths[i]} samples")

        # Build flattened index mapping
        self._build_flattened_index()

        # Apply task balancing if requested
        if self.balance_task is not None:
            self._apply_task_balancing()

    def _build_flattened_index(self):
        """Build a flattened index mapping for efficient access."""
        self.index_map = []
        for dataset_idx, dataset in enumerate(self.datasets):
            for local_idx in range(len(dataset)):
                self.index_map.append((dataset_idx, local_idx))

    def _apply_task_balancing(self):
        """Apply task balancing by duplicating samples to reach desired fractions."""
        original_len = len(self.index_map)
        
        for task, desired_fraction in self.balance_task.items():
            # Find indices with valid labels for that task
            valid_indices = [
                i for i, (ds_idx, loc_idx) in enumerate(self.index_map)
                if self._is_valid_task_label(ds_idx, loc_idx, task)
            ]
            current_count = len(valid_indices)
            
            # Calculate target count so that valid samples represent desired_fraction of the FINAL dataset
            # If we want X% of final dataset to have valid labels for this task:
            # current_count + n_to_add = desired_fraction * (original_len + n_to_add)
            # Solving for n_to_add:
            # n_to_add = (desired_fraction * original_len - current_count) / (1 - desired_fraction)
            
            if desired_fraction >= 1.0:
                if self.verbose:
                    print(f"[{task}] desired fraction {desired_fraction} >= 1.0, skipping balancing.")
                continue
                
            target_count_numerator = desired_fraction * original_len - current_count
            target_count_denominator = 1 - desired_fraction
            
            if target_count_denominator <= 0:
                if self.verbose:
                    print(f"[{task}] desired fraction {desired_fraction} too high, skipping balancing.")
                continue
                
            n_to_add = int(target_count_numerator / target_count_denominator)
            
            if n_to_add <= 0:
                if self.verbose:
                    final_dataset_size = original_len
                    actual_percentage = (current_count / final_dataset_size) * 100
                    print(f"[{task}] already has {current_count}/{final_dataset_size} ({actual_percentage:.1f}%) >= {desired_fraction*100:.1f}%, no duplication needed.")
                continue

            # Add the duplicated samples
            extra_indices = random.choices(valid_indices, k=n_to_add)
            extra_mapped = [self.index_map[i] for i in extra_indices]
            self.index_map.extend(extra_mapped)
            
            # Calculate final statistics
            final_dataset_size = len(self.index_map)
            final_valid_count = current_count + n_to_add
            actual_percentage = (final_valid_count / final_dataset_size) * 100

            if self.verbose:
                print(f"[{task}] duplicated {n_to_add} samples.")
                print(f"[{task}] final dataset size: {final_dataset_size}")
                print(f"[{task}] valid samples: {final_valid_count}/{final_dataset_size} ({actual_percentage:.1f}%)")
                print(f"[{task}] target was {desired_fraction*100:.1f}%")

        # Shuffle the index map to mix original and duplicated samples
        random.shuffle(self.index_map)

    def _is_valid_task_label(self, dataset_idx, local_idx, task):
        """Check if a sample has a valid (non-missing) label for the given task."""
        dataset = self.datasets[dataset_idx]
        if task == 'age':
            return dataset.age_groups[local_idx] != -1
        elif task == 'gender':
            return dataset.genders[local_idx] != -1
        elif task == 'emotion':
            return dataset.emotions[local_idx] != -1
        return False

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        """Get item using the flattened index mapping."""
        if idx >= len(self.index_map) or idx < 0:
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.index_map)}")
        
        dataset_idx, local_idx = self.index_map[idx]
        return self.datasets[dataset_idx][local_idx]
    
    def get_dataset_info(self, compute_stats=False):
        """
        Utility function to get information about the loaded dataset.
        
        Args:
            compute_stats (bool): If True, compute label distributions and missing value counts (expensive).
        
        Returns:
            dict: Information about each dataset including name, length, and optionally stats.
        """
        info = {}
        for i, name in enumerate(self.dataset_names[:len(self.datasets)]):
            dataset_info = {
                'original_length': self.dataset_lengths[i],
                'current_samples_in_index': sum(1 for ds_idx, _ in self.index_map if ds_idx == i)
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
        Compute class weights for the specified task based on the balanced dataset.
        
        Args:
            task (str): Task name ('age', 'gender', or 'emotion')
            
        Returns:
            torch.Tensor: Class weights tensor
        """
        if not hasattr(self, 'class_weights'):
            self.class_weights = {}

        if self.class_weights.get(task, None) is not None:
            return self.class_weights[task]
        
        if self.verbose:
            print(f"Computing class weights for task: {task} (TaskBalanceDataset)")
        
        # Collect all task data using the balanced index mapping
        all_task_data = []
        for dataset_idx, local_idx in self.index_map:
            dataset = self.datasets[dataset_idx]
            if task == 'age':
                all_task_data.append(dataset.age_groups[local_idx])
            elif task == 'gender':
                all_task_data.append(dataset.genders[local_idx])
            elif task == 'emotion':
                all_task_data.append(dataset.emotions[local_idx])
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
        
        if self.verbose:
            print(f"Class distribution for {task}: {dict(sorted(class_counts.items()))}")
            print(f"Class weights for {task}: {weights_tensor}")
        
        return weights_tensor    
    
    def get_task_weights(self):
        """
        Compute task weights for multitask learning based on the number of valid samples per task.
        
        Returns:
            torch.Tensor: Task weights tensor [age_weight, gender_weight, emotion_weight]
        """
        if hasattr(self, 'task_weights') and self.task_weights is not None:
            return self.task_weights
        
        # Count valid samples for each task using the balanced index mapping
        task_counts = {'age': 0, 'gender': 0, 'emotion': 0}
        
        for dataset_idx, local_idx in self.index_map:
            dataset = self.datasets[dataset_idx]
            
            # Count valid samples (non-missing labels) for each task
            if dataset.age_groups[local_idx] != -1:
                task_counts['age'] += 1
            if dataset.genders[local_idx] != -1:
                task_counts['gender'] += 1
            if dataset.emotions[local_idx] != -1:
                task_counts['emotion'] += 1
        
        # Calculate inverse frequency weights
        total_samples = len(self.index_map)
        weights = []
        
        for task in ['age', 'gender', 'emotion']:
            if task_counts[task] > 0:
                # Inverse frequency: total_samples / task_samples
                weight = total_samples / task_counts[task]
            else:
                weight = 1.0  # Default weight if no samples
            weights.append(weight)
        
        # Normalize weights so they sum to 3 (number of tasks)
        weights_tensor = torch.tensor(weights, dtype=torch.float32)
        weights_tensor = weights_tensor * 3.0 / weights_tensor.sum()
        
        self.task_weights = weights_tensor
        
        if self.verbose:
            print(f"Task sample counts: {task_counts}")
            print(f"Task weights: {weights_tensor}")
        
        return weights_tensor

if __name__ == "__main__":
    # Test transforms for BalancedDataset
    import torchvision.transforms as transforms
    
    # Test TaskBalanceDataset
    print("Testing TaskBalanceDataset...")
    
    # Define some basic transforms
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    try:
        # Test with emotion balancing
        print("\n=== Testing with emotion balancing (25%) ===")
        dataset_with_balance = TaskBalanceDataset(
            dataset_names=["test_dataset"],  # Replace with actual dataset names
            transform=test_transforms,
            split="train",
            datasets_root="../datasets_with_standard_labels",
            verbose=True,
            all_datasets=True,
            balance_task={"emotion": 0.25}
        )
        print(f"Dataset length with emotion balancing: {len(dataset_with_balance)}")
                
        # Test getting a sample
        if len(dataset_with_balance) > 0:
            print("\n=== Testing sample retrieval ===")
            sample_image, sample_labels = dataset_with_balance[0]
            print(f"Sample image shape: {sample_image.shape}")
            print(f"Sample labels: age={sample_labels[0].item()}, gender={sample_labels[1].item()}, emotion={sample_labels[2].item()}")
        
        # Test dataset info
        print("\n=== Testing dataset info ===")
        info = dataset_with_balance.get_dataset_info(compute_stats=True)
        for dataset_name, dataset_info in info.items():
            print(f"Dataset {dataset_name}:")
            print(f"  Original length: {dataset_info['original_length']}")
            print(f"  Current samples in index: {dataset_info['current_samples_in_index']}")
            if 'stats' in dataset_info:
                stats = dataset_info['stats']
                for stat_name, stat_value in stats.items():
                    print(f"  {stat_name}: {stat_value}")
        
        # Test class weights
        print("\n=== Testing class weights ===")
        for task in ['age', 'gender', 'emotion']:
            try:
                weights = dataset_with_balance.get_class_weights(task)
                print(f"{task} class weights: {weights}")
            except Exception as e:
                print(f"Could not compute weights for {task}: {e}")
                
        print("\nTaskBalanceDataset tests completed successfully!")


        print("\n=== Testing dataloader ===")
        from torch.utils.data import DataLoader
        dataloader = DataLoader(dataset_with_balance, batch_size=256 , shuffle=True)
        
        # Count valid samples (labels != -1) for each task
        emotion_valid_count = 0
        age_valid_count = 0
        gender_valid_count = 0
        total_samples = 0
        batches_without_emotion = 0  # Count batches with no emotion valid samples
        
        for batch_idx, (images, labels) in enumerate(dataloader):
            labels = [label.numpy() for label in labels]  # Convert labels to numpy for easier printing
            
            # Count valid labels in this batch
            emotion_valid_in_batch = sum(1 for label in labels[2] if label != -1)
            age_valid_in_batch = sum(1 for label in labels[0] if label != -1)
            gender_valid_in_batch = sum(1 for label in labels[1] if label != -1)
            
            emotion_valid_count += emotion_valid_in_batch
            age_valid_count += age_valid_in_batch
            gender_valid_count += gender_valid_in_batch
            total_samples += len(labels[0])
            
            # Count batches without emotion valid samples
            if emotion_valid_in_batch == 0:
                batches_without_emotion += 1
            
            # Print only first 6 batches
            if batch_idx < 6:
                print(f"Batch {batch_idx}: Emotion valid: {emotion_valid_in_batch}/{len(labels[2])}, Age valid: {age_valid_in_batch}/{len(labels[0])}, Gender valid: {gender_valid_in_batch}/{len(labels[1])}")            
            else:
                print(f"{batch_idx}/{len(dataloader)} MultiDataset batches processed", end='.\r')
        total_batches = batch_idx + 1  # Total number of batches processed
        
        print(f"\n=== Summary of valid labels (TaskBalanceDataset - All batches) ===")
        print(f"Total batches processed: {total_batches}")
        print(f"Total samples processed: {total_samples}")
        print(f"Emotion valid samples: {emotion_valid_count}/{total_samples} ({emotion_valid_count/total_samples*100:.1f}%)")
        print(f"Age valid samples: {age_valid_count}/{total_samples} ({age_valid_count/total_samples*100:.1f}%)")
        print(f"Gender valid samples: {gender_valid_count}/{total_samples} ({gender_valid_count/total_samples*100:.1f}%)")
        print(f"Batches without emotion valid samples: {batches_without_emotion}/{total_batches} ({batches_without_emotion/total_batches*100:.1f}%)")
        
        # Count valid samples in the entire dataset
        print(f"\n=== Complete dataset valid labels count ===")
        total_emotion_valid = 0
        total_age_valid = 0
        total_gender_valid = 0
        dataset_size = len(dataset_with_balance)
        
        for dataset_idx, local_idx in dataset_with_balance.index_map:
            dataset = dataset_with_balance.datasets[dataset_idx]
            if dataset.emotions[local_idx] != -1:
                total_emotion_valid += 1
            if dataset.age_groups[local_idx] != -1:
                total_age_valid += 1
            if dataset.genders[local_idx] != -1:
                total_gender_valid += 1
                
        print(f"Complete dataset size: {dataset_size}")
        print(f"Emotion valid samples in complete dataset: {total_emotion_valid}/{dataset_size} ({total_emotion_valid/dataset_size*100:.1f}%)")
        print(f"Age valid samples in complete dataset: {total_age_valid}/{dataset_size} ({total_age_valid/dataset_size*100:.1f}%)")
        print(f"Gender valid samples in complete dataset: {total_gender_valid}/{dataset_size} ({total_gender_valid/dataset_size*100:.1f}%)")

        # Comparison with MultiDataset (without task balancing)
        print("\n" + "="*60)
        print("=== COMPARISON: MultiDataset vs TaskBalanceDataset ===")
        print("="*60)
        
        # Create MultiDataset for comparison
        multi_dataset = MultiDataset(
            dataset_names=["test_dataset"],
            transform=test_transforms,
            split="train",
            datasets_root="../datasets_with_standard_labels",
            all_datasets=True,
            verbose=False
        )
        
        # Count valid samples in MultiDataset
        print(f"\n=== MultiDataset (no balancing) valid labels count ===")
        multi_total_emotion_valid = 0
        multi_total_age_valid = 0
        multi_total_gender_valid = 0
        multi_dataset_size = len(multi_dataset)
        
        for idx in range(multi_dataset_size):
            # Find which dataset and local index this global index corresponds to
            dataset_idx = 0
            for i, cumulative_length in enumerate(multi_dataset.cumulative_lengths[1:], 1):
                if idx < cumulative_length:
                    dataset_idx = i - 1
                    break
            local_idx = idx - multi_dataset.cumulative_lengths[dataset_idx]
            
            dataset = multi_dataset.datasets[dataset_idx]
            if dataset.emotions[local_idx] != -1:
                multi_total_emotion_valid += 1
            if dataset.age_groups[local_idx] != -1:
                multi_total_age_valid += 1
            if dataset.genders[local_idx] != -1:
                multi_total_gender_valid += 1
        
        print(f"MultiDataset size: {multi_dataset_size}")
        print(f"Emotion valid samples: {multi_total_emotion_valid}/{multi_dataset_size} ({multi_total_emotion_valid/multi_dataset_size*100:.1f}%)")
        print(f"Age valid samples: {multi_total_age_valid}/{multi_dataset_size} ({multi_total_age_valid/multi_dataset_size*100:.1f}%)")
        print(f"Gender valid samples: {multi_total_gender_valid}/{multi_dataset_size} ({multi_total_gender_valid/multi_dataset_size*100:.1f}%)")
        
        # Test MultiDataset dataloader
        print(f"\n=== Testing MultiDataset dataloader ===")
        multi_dataloader = DataLoader(multi_dataset, batch_size=256, shuffle=True)
        
        # Count valid samples (labels != -1) for each task in MultiDataset
        multi_emotion_valid_count = 0
        multi_age_valid_count = 0
        multi_gender_valid_count = 0
        multi_total_samples = 0
        multi_batches_without_emotion = 0  # Count batches with no emotion valid samples
        
        for batch_idx, (images, labels) in enumerate(multi_dataloader):
            labels = [label.numpy() for label in labels]  # Convert labels to numpy for easier printing
            
            # Count valid labels in this batch
            emotion_valid_in_batch = sum(1 for label in labels[2] if label != -1)
            age_valid_in_batch = sum(1 for label in labels[0] if label != -1)
            gender_valid_in_batch = sum(1 for label in labels[1] if label != -1)
            
            multi_emotion_valid_count += emotion_valid_in_batch
            multi_age_valid_count += age_valid_in_batch
            multi_gender_valid_count += gender_valid_in_batch
            multi_total_samples += len(labels[0])
            
            # Count batches without emotion valid samples
            if emotion_valid_in_batch == 0:
                multi_batches_without_emotion += 1
            
            # Print only first 6 batches
            if batch_idx < 6:
                print(f"MultiDataset Batch {batch_idx}: Emotion valid: {emotion_valid_in_batch}/{len(labels[2])}, Age valid: {age_valid_in_batch}/{len(labels[0])}, Gender valid: {gender_valid_in_batch}/{len(labels[1])}")            
            else:
                print(f"{batch_idx}/{len(multi_dataloader)} MultiDataset batches processed", end='.\r')
        multi_total_batches = batch_idx + 1  # Total number of batches processed
        
        print(f"\n=== MultiDataset Dataloader Summary (All batches) ===")
        print(f"Total batches processed: {multi_total_batches}")
        print(f"Total samples processed in batches: {multi_total_samples}")
        print(f"Emotion valid samples in batches: {multi_emotion_valid_count}/{multi_total_samples} ({multi_emotion_valid_count/multi_total_samples*100:.1f}%)")
        print(f"Age valid samples in batches: {multi_age_valid_count}/{multi_total_samples} ({multi_age_valid_count/multi_total_samples*100:.1f}%)")
        print(f"Gender valid samples in batches: {multi_gender_valid_count}/{multi_total_samples} ({multi_gender_valid_count/multi_total_samples*100:.1f}%)")
        print(f"Batches without emotion valid samples: {multi_batches_without_emotion}/{multi_total_batches} ({multi_batches_without_emotion/multi_total_batches*100:.1f}%)")
        
        # Show comparison
        print(f"\n=== COMPARISON SUMMARY ===")
        print(f"Dataset sizes:")
        print(f"  MultiDataset (original): {multi_dataset_size}")
        print(f"  TaskBalanceDataset (balanced): {dataset_size}")
        print(f"  Size increase: +{dataset_size - multi_dataset_size} samples ({((dataset_size - multi_dataset_size)/multi_dataset_size*100):.1f}%)")
        
        print(f"\nBatch processing comparison (all batches):")
        print(f"  MultiDataset total batches: {multi_total_batches}")
        print(f"  TaskBalanceDataset total batches: {total_batches}")
        print(f"  MultiDataset samples processed: {multi_total_samples}")
        print(f"  TaskBalanceDataset samples processed: {total_samples}")
        
        print(f"\nEmotion valid samples in batches:")
        print(f"  MultiDataset: {multi_emotion_valid_count}/{multi_total_samples} ({multi_emotion_valid_count/multi_total_samples*100:.1f}%)")
        print(f"  TaskBalanceDataset: {emotion_valid_count}/{total_samples} ({emotion_valid_count/total_samples*100:.1f}%)")
        if multi_total_samples > 0 and total_samples > 0:
            emotion_improvement = (emotion_valid_count/total_samples) - (multi_emotion_valid_count/multi_total_samples)
            print(f"  Improvement in emotion representation: {emotion_improvement*100:.1f} percentage points")
        
        print(f"\nBatches without emotion valid samples:")
        print(f"  MultiDataset: {multi_batches_without_emotion}/{multi_total_batches} ({multi_batches_without_emotion/multi_total_batches*100:.1f}%)")
        print(f"  TaskBalanceDataset: {batches_without_emotion}/{total_batches} ({batches_without_emotion/total_batches*100:.1f}%)")
        if multi_total_batches > 0 and total_batches > 0:
            batch_improvement = (multi_batches_without_emotion/multi_total_batches) - (batches_without_emotion/total_batches)
            print(f"  Reduction in batches without emotion: {batch_improvement*100:.1f} percentage points")
        
        print(f"\nAge valid samples in batches:")
        print(f"  MultiDataset: {multi_age_valid_count}/{multi_total_samples} ({multi_age_valid_count/multi_total_samples*100:.1f}%)")
        print(f"  TaskBalanceDataset: {age_valid_count}/{total_samples} ({age_valid_count/total_samples*100:.1f}%)")
        if multi_total_samples > 0 and total_samples > 0:
            age_improvement = (age_valid_count/total_samples) - (multi_age_valid_count/multi_total_samples)
            print(f"  Improvement in age representation: {age_improvement*100:.1f} percentage points")
        
        print(f"\nGender valid samples in batches:")
        print(f"  MultiDataset: {multi_gender_valid_count}/{multi_total_samples} ({multi_gender_valid_count/multi_total_samples*100:.1f}%)")
        print(f"  TaskBalanceDataset: {gender_valid_count}/{total_samples} ({gender_valid_count/total_samples*100:.1f}%)")
        if multi_total_samples > 0 and total_samples > 0:
            gender_improvement = (gender_valid_count/total_samples) - (multi_gender_valid_count/multi_total_samples)
            print(f"  Improvement in gender representation: {gender_improvement*100:.1f} percentage points")
            
        print(f"\nComplete dataset comparison:")
        print(f"\nEmotion valid samples:")
        print(f"  MultiDataset: {multi_total_emotion_valid}/{multi_dataset_size} ({multi_total_emotion_valid/multi_dataset_size*100:.1f}%)")
        print(f"  TaskBalanceDataset: {total_emotion_valid}/{dataset_size} ({total_emotion_valid/dataset_size*100:.1f}%)")
        print(f"  Increase: +{total_emotion_valid - multi_total_emotion_valid} samples")
        
        print(f"\nAge valid samples:")
        print(f"  MultiDataset: {multi_total_age_valid}/{multi_dataset_size} ({multi_total_age_valid/multi_dataset_size*100:.1f}%)")
        print(f"  TaskBalanceDataset: {total_age_valid}/{dataset_size} ({total_age_valid/dataset_size*100:.1f}%)")
        print(f"  Increase: +{total_age_valid - multi_total_age_valid} samples")
        
        print(f"\nGender valid samples:")
        print(f"  MultiDataset: {multi_total_gender_valid}/{multi_dataset_size} ({multi_total_gender_valid/multi_dataset_size*100:.1f}%)")
        print(f"  TaskBalanceDataset: {total_gender_valid}/{dataset_size} ({total_gender_valid/dataset_size*100:.1f}%)")
        print(f"  Increase: +{total_gender_valid - multi_total_gender_valid} samples")
        
    except Exception as e:
        print(f"Error testing TaskBalanceDataset: {e}")
        print("This is expected if the test datasets are not available.")
        print("The TaskBalanceDataset class has been created and should work with proper dataset paths.")
    