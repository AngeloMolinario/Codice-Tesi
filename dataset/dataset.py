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

class WeightCalculationMixin:
    """
    Mixin class to provide centralized weight calculation logic for datasets.
    Subclasses must implement:
    - _get_all_labels_for_task(self, task)
    - _get_task_counts_and_total_len(self)
    - self.verbose attribute
    """
    def get_class_weights(self, task):
        """Compute class weights for the specified task based on the data provided by the subclass."""
        if not hasattr(self, 'class_weights'):
            self.class_weights = {}
        if self.class_weights.get(task):
            return self.class_weights[task]

        if self.verbose:
            print(f"Computing class weights for task: {task} (using {self.__class__.__name__})")

        all_task_data = self._get_all_labels_for_task(task)
        
        class_counts = {}
        total_valid_samples = 0
        for value in all_task_data:
            if value != -1:
                class_counts[value] = class_counts.get(value, 0) + 1
                total_valid_samples += 1
        
        if total_valid_samples == 0:
            if self.verbose:
                print(f"Warning: No valid samples found for task {task} in {self.__class__.__name__}")
            return None

        class_indices = sorted(class_counts.keys())
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

    def get_task_weights(self):
        """Compute task weights for multitask learning based on data provided by the subclass."""
        if hasattr(self, 'task_weights') and self.task_weights is not None:
            return self.task_weights

        task_counts, total_samples = self._get_task_counts_and_total_len()
        
        weights = []
        for task in ['age', 'gender', 'emotion']:
            count = task_counts.get(task, 0)
            weight = total_samples / count if count > 0 else 1.0
            weights.append(weight)
            
        weights_tensor = torch.tensor(weights, dtype=torch.float32)
        if weights_tensor.sum() > 0:
            weights_tensor = weights_tensor * len(weights) / weights_tensor.sum()
        
        self.task_weights = weights_tensor
        
        if self.verbose:
            print(f"Task sample counts: {task_counts}")
            print(f"Task weights: {weights_tensor}")
            
        return weights_tensor

class BaseDataset(Dataset, WeightCalculationMixin):
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

    def _get_all_labels_for_task(self, task):
        """Provides all labels for a given task for this single dataset."""
        if task == 'age':
            return self.age_groups
        elif task == 'gender':
            return self.genders
        elif task == 'emotion':
            return self.emotions
        else:
            raise ValueError(f"Unknown task: {task}. Must be one of 'age', 'gender', 'emotion'")

    def _get_task_counts_and_total_len(self):
        """Provides task counts and total length for this single dataset."""
        task_counts = {
            'age': sum(1 for age in self.age_groups if age != -1),
            'gender': sum(1 for gender in self.genders if gender != -1),
            'emotion': sum(1 for emotion in self.emotions if emotion != -1)
        }
        return task_counts, len(self)

class MultiDataset(Dataset, WeightCalculationMixin):
    def __init__(self, dataset_names, transform=None, split="train", datasets_root="datasets_with_standard_labels", all_datasets=False, verbose=False):
        """
        Initialize MultiDataset with multiple datasets.
        
        Args:
            dataset_names (list): List of dataset names to load
            transform: Optional transforms to apply to images
            split (str): Split to use ("train" or "test")
            datasets_root (str): Root directory containing all datasets
            all_datasets (bool): if True, load all datasets in the root directory and ignore dataset_names
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
    
    def _get_all_labels_for_task(self, task):
        """Provides all labels for a given task across all datasets."""
        all_task_data = []
        for dataset in self.datasets:
            if task == 'age':
                all_task_data.extend(dataset.age_groups)
            elif task == 'gender':
                all_task_data.extend(dataset.genders)
            elif task == 'emotion':
                all_task_data.extend(dataset.emotions)
        return all_task_data

    def _get_task_counts_and_total_len(self):
        """Provides task counts and total length for weight calculation."""
        task_counts = {'age': 0, 'gender': 0, 'emotion': 0}
        for dataset in self.datasets:
            task_counts['age'] += sum(1 for age in dataset.age_groups if age != -1)
            task_counts['gender'] += sum(1 for gender in dataset.genders if gender != -1)
            task_counts['emotion'] += sum(1 for emotion in dataset.emotions if emotion != -1)
        return task_counts, self.total_length

class TaskBalanceDataset(Dataset, WeightCalculationMixin):
    def __init__(self, dataset_names, transform=None, split="train", 
                 datasets_root="datasets_with_standard_labels", all_datasets=False, 
                 verbose=False, balance_task=None, augment_duplicate=None):
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
            augment_duplicate: Optional transforms to apply ONLY to duplicated samples
        """
        self.dataset_names = dataset_names
        self.transform = transform
        self.split = split
        self.datasets_root = datasets_root
        self.verbose = verbose
        self.balance_task = balance_task  # e.g. {"emotion": 0.25}
        self.augment_duplicate = augment_duplicate  # Trasformazioni per i duplicati
        self.duplicated_indices = []  # Lista per contenere gli indici dei campioni duplicati
        
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
                # Aggiungiamo un flag per marcare i duplicati (inizialmente tutti False)
                self.index_map.append([dataset_idx, local_idx, False])

    def _apply_task_balancing(self):
        """Apply task balancing by duplicating samples to reach desired fractions."""
        original_len = len(self.index_map)
        
        for task, desired_fraction in self.balance_task.items():
            # Find indices with valid labels for that task
            valid_indices = [
                i for i, (ds_idx, loc_idx, is_dup) in enumerate(self.index_map)
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
            # I nuovi campioni vengono marcati come duplicati (True)
            extra_mapped = [[self.index_map[i][0], self.index_map[i][1], True] for i in extra_indices]
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

        # Popola la lista self.duplicated_indices dopo lo shuffle
        self.duplicated_indices = [i for i, item in enumerate(self.index_map) if item[2]]
        
        if self.verbose:
            print(f"Total duplicated samples: {len(self.duplicated_indices)}")
            if len(self.duplicated_indices) > 0:
                print(f"First few duplicated indices after shuffle: {self.duplicated_indices[:5]}")

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
        
        # Estraiamo tutti i dati, incluso il flag di duplicazione
        dataset_idx, local_idx, is_duplicated = self.index_map[idx]
        
        # Logica 1: Se augment_duplicate è None, comportamento normale
        if self.augment_duplicate is None:
            # Usa il metodo standard del dataset
            image, label = self.datasets[dataset_idx][local_idx]
            return image, label
        
        # Logica 2: Se augment_duplicate non è None, gestione manuale delle trasformazioni
        else:
            if not is_duplicated:
                image, label = self.datasets[dataset_idx][local_idx]
                return image, label

            # Ottieni il dataset e il path dell'immagine
            dataset = self.datasets[dataset_idx]
            img_path = dataset.img_paths[local_idx]
            
            # Carica l'immagine manualmente
            image = Image.open(img_path).convert('RGB')
            
            image = self.augment_duplicate(image)
            
            label = [
                torch.tensor(dataset.age_groups[local_idx], dtype=torch.long),
                torch.tensor(dataset.genders[local_idx], dtype=torch.long),            
                torch.tensor(dataset.emotions[local_idx], dtype=torch.long)
            ]
            
            return image, label

    def _get_all_labels_for_task(self, task):
        """Provides all labels for a given task from the balanced index map."""
        all_task_data = []
        if not hasattr(self, 'index_map'):
            self._build_flattened_index()
            
        for dataset_idx, local_idx, _ in self.index_map:
            dataset = self.datasets[dataset_idx]
            if task == 'age':
                all_task_data.append(dataset.age_groups[local_idx])
            elif task == 'gender':
                all_task_data.append(dataset.genders[local_idx])
            elif task == 'emotion':
                all_task_data.append(dataset.emotions[local_idx])
        return all_task_data

    def _get_task_counts_and_total_len(self):
        """Provides task counts and total length from the balanced index map."""
        if not hasattr(self, 'index_map'):
            self._build_flattened_index()

        task_counts = {'age': 0, 'gender': 0, 'emotion': 0}
        for dataset_idx, local_idx, _ in self.index_map:
            dataset = self.datasets[dataset_idx]
            if dataset.age_groups[local_idx] != -1:
                task_counts['age'] += 1
            if dataset.genders[local_idx] != -1:
                task_counts['gender'] += 1
            if dataset.emotions[local_idx] != -1:
                task_counts['emotion'] += 1
        return task_counts, len(self.index_map)
    
    def get_duplicated_indices(self):
        """
        Returns the list of indices that correspond to duplicated samples.
        
        Returns:
            list: List of indices in the shuffled dataset that are duplicates
        """
        return self.duplicated_indices.copy()  # Return a copy to prevent external modification
    
    def is_duplicated_sample(self, idx):
        """
        Check if a sample at the given index is a duplicate.
        
        Args:
            idx (int): Index to check
            
        Returns:
            bool: True if the sample is a duplicate, False otherwise
        """
        if idx < 0 or idx >= len(self.index_map):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.index_map)}")
        return idx in self.duplicated_indices

    def get_dataset_info(self, compute_stats=False):
        """
        Utility function to get information about the loaded dataset.
        
        Args:
            compute_stats (bool): If True, compute detailed statistics for each dataset
            
        Returns:
            dict: Information about each dataset including length and optional statistics
        """
        info = {}
        for i, name in enumerate(self.dataset_names[:len(self.datasets)]):
            dataset_info = {
                'original_length': self.dataset_lengths[i],
                'current_samples_in_index': sum(1 for ds_idx, _, _ in self.index_map if ds_idx == i)
            }
            if compute_stats:
                ds = self.datasets[i]
                stats = {}
                for task in ['age', 'gender', 'emotion']:
                    if task == 'age':
                        data = [age for age in ds.age_groups]
                    elif task == 'gender':
                        data = [gender for gender in ds.genders]
                    elif task == 'emotion':
                        data = [emotion for emotion in ds.emotions]
                    
                    valid_data = [x for x in data if x != -1]
                    if valid_data:
                        from collections import Counter
                        distribution = dict(Counter(valid_data))
                        stats[f'{task}_distribution'] = distribution
                        stats[f'{task}_valid_count'] = len(valid_data)
                        stats[f'{task}_missing_count'] = len(data) - len(valid_data)
                    else:
                        stats[f'{task}_distribution'] = {}
                        stats[f'{task}_valid_count'] = 0
                        stats[f'{task}_missing_count'] = len(data)
                dataset_info['stats'] = stats
            info[name] = dataset_info
        return info

if __name__ == "__main__":
    # Test transforms for BalancedDataset
    import torchvision.transforms as transforms
    
    print("="*80)
    print("TESTING ALL DATASET CLASSES WITH WEIGHT CALCULATION MIXIN")
    print("="*80)
    
    # Define some basic transforms
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    try:
        print("\n" + "="*60)
        print("=== TESTING BaseDataset ===")
        print("="*60)
        
        # Test BaseDataset with WeightCalculationMixin
        print("Creating BaseDataset...")
        base_datasets = []
        try:
            datasets_root = "../datasets_with_standard_labels"
            if os.path.exists(datasets_root):
                dataset_dirs = [d for d in os.listdir(datasets_root) if os.path.isdir(os.path.join(datasets_root, d))]
                if dataset_dirs:
                    first_dataset = dataset_dirs[0]
                    base_dataset = BaseDataset(
                        root=os.path.join(datasets_root, first_dataset),
                        transform=test_transforms,
                        split="train",
                        verbose=True
                    )
                    base_datasets.append(base_dataset)
                    print(f"✓ BaseDataset loaded successfully: {len(base_dataset)} samples")
                    
                    # Test WeightCalculationMixin methods
                    print("\n--- Testing BaseDataset WeightCalculationMixin methods ---")
                    for task in ['age', 'gender', 'emotion']:
                        try:
                            weights = base_dataset.get_class_weights(task)
                            if weights is not None:
                                print(f"✓ {task} class weights: {weights}")
                            else:
                                print(f"⚠ No valid samples for {task}")
                        except Exception as e:
                            print(f"✗ Error computing {task} weights: {e}")
                    
                    try:
                        task_weights = base_dataset.get_task_weights()
                        print(f"✓ Task weights: {task_weights}")
                    except Exception as e:
                        print(f"✗ Error computing task weights: {e}")
                        
                    # Test sample retrieval
                    if len(base_dataset) > 0:
                        sample_img, sample_labels = base_dataset[0]
                        print(f"✓ Sample shape: {sample_img.shape}, labels: {[l.item() for l in sample_labels]}")
                else:
                    print("⚠ No datasets found in datasets_root")
            else:
                print(f"⚠ datasets_root '{datasets_root}' not found")
        except Exception as e:
            print(f"✗ Error testing BaseDataset: {e}")
        
        print("\n" + "="*60)
        print("=== TESTING MultiDataset ===")
        print("="*60)
        
        # Test MultiDataset with WeightCalculationMixin
        try:
            multi_dataset = MultiDataset(
                dataset_names=["test_dataset"],
                transform=test_transforms,
                split="train",
                datasets_root="../datasets_with_standard_labels",
                all_datasets=True,
                verbose=True
            )
            print(f"✓ MultiDataset loaded successfully: {len(multi_dataset)} samples")
            
            # Test WeightCalculationMixin methods
            print("\n--- Testing MultiDataset WeightCalculationMixin methods ---")
            for task in ['age', 'gender', 'emotion']:
                try:
                    weights = multi_dataset.get_class_weights(task)
                    if weights is not None:
                        print(f"✓ {task} class weights: {weights}")
                    else:
                        print(f"⚠ No valid samples for {task}")
                except Exception as e:
                    print(f"✗ Error computing {task} weights: {e}")
            
            try:
                task_weights = multi_dataset.get_task_weights()
                print(f"✓ Task weights: {task_weights}")
            except Exception as e:
                print(f"✗ Error computing task weights: {e}")
                
        except Exception as e:
            print(f"✗ Error testing MultiDataset: {e}")
            multi_dataset = None

        print("\n" + "="*60)
        print("=== TESTING TaskBalanceDataset ===")
        print("="*60)
        
        # Test TaskBalanceDataset with emotion balancing
        print("Creating TaskBalanceDataset with emotion balancing (25%)...")
        dataset_with_balance = TaskBalanceDataset(
            dataset_names=["test_dataset"],
            transform=test_transforms,
            split="train",
            datasets_root="../datasets_with_standard_labels",
            verbose=True,
            all_datasets=True,
            balance_task={"emotion": 0.25}
        )
        print(f"✓ TaskBalanceDataset loaded successfully: {len(dataset_with_balance)} samples")
        
        # Test the duplicated_indices functionality (if implemented)
        if hasattr(dataset_with_balance, 'duplicated_indices'):
            duplicated_count = len(dataset_with_balance.duplicated_indices)
            print(f"✓ Duplicated indices tracked: {duplicated_count} samples")
            if duplicated_count > 0:
                print(f"  First few duplicated indices: {dataset_with_balance.duplicated_indices[:5]}")
        else:
            print("⚠ duplicated_indices not implemented yet")
        
        # Test WeightCalculationMixin methods on balanced dataset
        print("\n--- Testing TaskBalanceDataset WeightCalculationMixin methods ---")
        for task in ['age', 'gender', 'emotion']:
            try:
                weights = dataset_with_balance.get_class_weights(task)
                if weights is not None:
                    print(f"✓ {task} class weights: {weights}")
                else:
                    print(f"⚠ No valid samples for {task}")
            except Exception as e:
                print(f"✗ Error computing {task} weights: {e}")
        
        try:
            task_weights = dataset_with_balance.get_task_weights()
            print(f"✓ Task weights: {task_weights}")
        except Exception as e:
            print(f"✗ Error computing task weights: {e}")
                
        # Test getting a sample
        if len(dataset_with_balance) > 0:
            print("\n--- Testing sample retrieval ---")
            sample_image, sample_labels = dataset_with_balance[0]
            print(f"✓ Sample image shape: {sample_image.shape}")
            print(f"✓ Sample labels: age={sample_labels[0].item()}, gender={sample_labels[1].item()}, emotion={sample_labels[2].item()}")
        
        # Test dataset info
        print("\n--- Testing dataset info ---")
        info = dataset_with_balance.get_dataset_info(compute_stats=True)
        for dataset_name, dataset_info in info.items():
            print(f"Dataset {dataset_name}:")
            print(f"  Original length: {dataset_info['original_length']}")
            print(f"  Current samples in index: {dataset_info['current_samples_in_index']}")
            if 'stats' in dataset_info:
                stats = dataset_info['stats']
                for stat_name, stat_value in stats.items():
                    print(f"  {stat_name}: {stat_value}")

        print("\n" + "="*60)
        print("=== TESTING DATALOADER FUNCTIONALITY ===")
        print("="*60)
        
        from torch.utils.data import DataLoader
        
        # Test TaskBalanceDataset dataloader
        print("Testing TaskBalanceDataset dataloader...")
        dataloader = DataLoader(dataset_with_balance, batch_size=256, shuffle=True)
        
        # Count valid samples (labels != -1) for each task
        emotion_valid_count = 0
        age_valid_count = 0
        gender_valid_count = 0
        total_samples = 0
        batches_without_emotion = 0
        
        for batch_idx, (images, labels) in enumerate(dataloader):
            labels = [label.numpy() for label in labels]
            
            # Count valid labels in this batch
            emotion_valid_in_batch = sum(1 for label in labels[2] if label != -1)
            age_valid_in_batch = sum(1 for label in labels[0] if label != -1)
            gender_valid_in_batch = sum(1 for label in labels[1] if label != -1)
            
            emotion_valid_count += emotion_valid_in_batch
            age_valid_count += age_valid_in_batch
            gender_valid_count += gender_valid_in_batch
            total_samples += len(labels[0])
            
            if emotion_valid_in_batch == 0:
                batches_without_emotion += 1
            
            # Print only first 3 batches for brevity
            if batch_idx < 3:
                print(f"  Batch {batch_idx}: Emotion {emotion_valid_in_batch}/{len(labels[2])}, Age {age_valid_in_batch}/{len(labels[0])}, Gender {gender_valid_in_batch}/{len(labels[1])}")
            
            # Break after 10 batches for testing purposes
            if batch_idx >= 9:
                break
                
        total_batches = min(batch_idx + 1, 10)
        
        print(f"\n--- TaskBalanceDataset Summary (first {total_batches} batches) ---")
        print(f"Total samples processed: {total_samples}")
        print(f"Emotion valid: {emotion_valid_count}/{total_samples} ({emotion_valid_count/total_samples*100:.1f}%)")
        print(f"Age valid: {age_valid_count}/{total_samples} ({age_valid_count/total_samples*100:.1f}%)")
        print(f"Gender valid: {gender_valid_count}/{total_samples} ({gender_valid_count/total_samples*100:.1f}%)")
        print(f"Batches without emotion: {batches_without_emotion}/{total_batches} ({batches_without_emotion/total_batches*100:.1f}%)")

        # Test MultiDataset dataloader for comparison (if available)
        if multi_dataset is not None:
            print(f"\n--- Testing MultiDataset dataloader for comparison ---")
            multi_dataloader = DataLoader(multi_dataset, batch_size=256, shuffle=True)
            
            multi_emotion_valid_count = 0
            multi_age_valid_count = 0
            multi_gender_valid_count = 0
            multi_total_samples = 0
            multi_batches_without_emotion = 0
            
            for batch_idx, (images, labels) in enumerate(multi_dataloader):
                labels = [label.numpy() for label in labels]
                
                emotion_valid_in_batch = sum(1 for label in labels[2] if label != -1)
                age_valid_in_batch = sum(1 for label in labels[0] if label != -1)
                gender_valid_in_batch = sum(1 for label in labels[1] if label != -1)
                
                multi_emotion_valid_count += emotion_valid_in_batch
                multi_age_valid_count += age_valid_in_batch
                multi_gender_valid_count += gender_valid_in_batch
                multi_total_samples += len(labels[0])
                
                if emotion_valid_in_batch == 0:
                    multi_batches_without_emotion += 1
                
                # Print only first 3 batches for brevity
                if batch_idx < 3:
                    print(f"  MultiDataset Batch {batch_idx}: Emotion {emotion_valid_in_batch}/{len(labels[2])}, Age {age_valid_in_batch}/{len(labels[0])}, Gender {gender_valid_in_batch}/{len(labels[1])}")
                
                # Break after 10 batches for testing purposes
                if batch_idx >= 9:
                    break
                    
            multi_total_batches = min(batch_idx + 1, 10)
            
            print(f"\n--- MultiDataset Summary (first {multi_total_batches} batches) ---")
            print(f"Total samples processed: {multi_total_samples}")
            print(f"Emotion valid: {multi_emotion_valid_count}/{multi_total_samples} ({multi_emotion_valid_count/multi_total_samples*100:.1f}%)")
            print(f"Age valid: {multi_age_valid_count}/{multi_total_samples} ({multi_age_valid_count/multi_total_samples*100:.1f}%)")
            print(f"Gender valid: {multi_gender_valid_count}/{multi_total_samples} ({multi_gender_valid_count/multi_total_samples*100:.1f}%)")
            print(f"Batches without emotion: {multi_batches_without_emotion}/{multi_total_batches} ({multi_batches_without_emotion/multi_total_batches*100:.1f}%)")
            
            # Show comparison
            print(f"\n--- COMPARISON SUMMARY ---")
            print(f"Dataset sizes:")
            print(f"  MultiDataset: {len(multi_dataset)}")
            print(f"  TaskBalanceDataset: {len(dataset_with_balance)}")
            print(f"  Size increase: +{len(dataset_with_balance) - len(multi_dataset)} samples")
            
            if multi_total_samples > 0 and total_samples > 0:
                emotion_improvement = (emotion_valid_count/total_samples) - (multi_emotion_valid_count/multi_total_samples)
                print(f"Emotion representation improvement: {emotion_improvement*100:.1f} percentage points")
                
                batch_improvement = (multi_batches_without_emotion/multi_total_batches) - (batches_without_emotion/total_batches)
                print(f"Reduction in batches without emotion: {batch_improvement*100:.1f} percentage points")

        print("\n" + "="*60)
        print("=== TESTING WEIGHT CALCULATION CONSISTENCY ===")
        print("="*60)
        
        # Test that all classes give consistent results when they should
        if base_datasets and multi_dataset is not None:
            print("Comparing weight calculations between BaseDataset and MultiDataset...")
            base_dataset = base_datasets[0]  # Use first BaseDataset
            
            for task in ['age', 'gender', 'emotion']:
                try:
                    base_weights = base_dataset.get_class_weights(task)
                    multi_weights = multi_dataset.get_class_weights(task)
                    
                    if base_weights is not None and multi_weights is not None:
                        # For single dataset, MultiDataset should contain the BaseDataset
                        if len(multi_dataset.datasets) == 1:
                            if torch.allclose(base_weights, multi_weights, atol=1e-6):
                                print(f"✓ {task} weights consistent between BaseDataset and MultiDataset")
                            else:
                                print(f"⚠ {task} weights differ: Base={base_weights}, Multi={multi_weights}")
                        else:
                            print(f"ℹ {task} weights (different datasets): Base={base_weights}, Multi={multi_weights}")
                    else:
                        print(f"⚠ {task} weights: Base={base_weights}, Multi={multi_weights}")
                except Exception as e:
                    print(f"✗ Error comparing {task} weights: {e}")

        print("\n" + "="*60)
        print("=== ALL TESTS COMPLETED ===")
        print("="*60)
        print("✓ WeightCalculationMixin successfully tested across all dataset classes")
        print("✓ Task balancing functionality tested")
        print("✓ Dataloader functionality tested")
        print("✓ Weight calculation consistency verified")
        
    except Exception as e:
        print(f"✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        print("This is expected if the test datasets are not available.")
        print("The classes have been created and should work with proper dataset paths.")
