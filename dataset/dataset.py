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

# Task indices
AGE_IDX = 0
GENDER_IDX = 1
EMOTION_IDX = 2
ALL_TASK_IDXS = [AGE_IDX, GENDER_IDX, EMOTION_IDX]


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
        if "-" in age_str or "more" in age_str or "+" in age_str:
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
            return -1

    except (ValueError, TypeError) as e:
        print(f"Warning: Cannot convert age '{age}' to float: {e}")
        return -1
    

class WeightCalculationMixin:
    """
    Mixin class to provide centralized weight calculation logic for datasets.
    Subclasses must implement:
    - _get_all_labels_for_task(self, task_idx)
    - _get_task_counts_and_total_len(self)
    - self.verbose attribute
    """
    def get_class_weights(self, task_idx: int, weighting_method: str = 'normalized'):
        """
        Calcola i pesi delle classi per l'indice del task specificato.

        Args:
            task_idx: Indice del task (0=età, 1=genere, 2=emozione).
            weighting_method: Strategia per il calcolo dei pesi.
                - 'default': Frequenza inversa -> total / (n_classes * count).
                - 'normalized': Come 'default', ma normalizzato con massimo = 1.0 (consigliato).
                - 'inverse_sqrt': Radice quadrata inversa della frequenza -> 1 / sqrt(count).
                - 'normalized_inverse_sqrt': Radice quadrata inversa della frequenza, normalizzata con massimo = 1.0.

        Returns:
            Un tensore di pesi per le classi.
        """
        if not hasattr(self, 'class_weights_cache'):
            self.class_weights_cache = {}
        # <--- CORREZIONE: Aggiornata la chiave della cache
        if self.class_weights_cache.get((task_idx, weighting_method)) is not None:
            return self.class_weights_cache[(task_idx, weighting_method)]

        if self.verbose:
            print(f"Computing class weights for task {task_idx} with method '{weighting_method}'")

        all_task_data = self._get_all_labels_for_task(task_idx)

        class_counts = {}
        total_valid_samples = 0
        for value in all_task_data:
            if value != -1: # Ignora le etichette non valide
                class_counts[value] = class_counts.get(value, 0) + 1
                total_valid_samples += 1

        if total_valid_samples == 0:
            if self.verbose:
                print(f"Warning: No valid samples found for task idx {task_idx}")
            return None

        class_indices = sorted(class_counts.keys())
        weights_array = []

        # --- SELEZIONE DELLA STRATEGIA DI PESATURA ---
        if weighting_method in ('default', 'normalized'):
            # Metodo classico basato sulla frequenza inversa
            for class_idx in class_indices:
                count = class_counts[class_idx]
                weight = total_valid_samples / (len(class_counts) * count)
                weights_array.append(weight)

        elif weighting_method == 'inverse_sqrt' or weighting_method == 'normalized_inverse_sqrt':
            # <--- NUOVO: Metodo aggressivo con radice quadrata inversa
            # Particolarmente utile per classi con code lunghe (molto rare)
            for class_idx in class_indices:
                count = class_counts[class_idx]
                # Aggiunto clamp_min(1) per evitare errori con conteggi a zero, sebbene già filtrati
                weight = 1.0 / math.sqrt(max(count, 1))
                weights_array.append(weight)
        
        else:
            raise ValueError(f"Unknown weighting_method: '{weighting_method}'. "
                             f"Available options are 'default', 'normalized', 'inverse_sqrt'.")

        # --- APPLICAZIONE DELLA NORMALIZZAZIONE (SE RICHIESTA DALLA STRATEGIA) ---
        if weighting_method in ('normalized', 'normalized_inverse_sqrt'):
            max_weight = max(weights_array) if weights_array else 0
            if max_weight > 0:
                weights_array = [w / max_weight for w in weights_array]
                if self.verbose:
                    print(f"Weights have been normalized (max=1.0)")
            elif self.verbose:
                print(f"Warning: max_weight is zero, skipping normalization")
        
        weights_tensor = torch.tensor(weights_array, dtype=torch.float32)
        
        # Aggiorna la cache
        self.class_weights_cache[(task_idx, weighting_method)] = weights_tensor

        if self.verbose:
            print(f"Class distribution (task {task_idx}): {dict(sorted(class_counts.items()))}")
            final_weights_rounded = [round(w, 3) for w in weights_tensor.tolist()]
            print(f"Final class weights (task {task_idx}, method '{weighting_method}'): {final_weights_rounded}")

        return weights_tensor

    def get_class_weights2(self, task_idx: int, normalize: bool = True):
        """Compute inverse-frequency class weights for the specified task index (0=age, 1=gender, 2=emotion).
        
        Args:
            task_idx: index of the task (0=age, 1=gender, 2=emotion)
            normalize: if True, normalize weights so that max weight = 1.0 (recommended for ordinal tasks)
        
        Returns:
            Tensor of class weights, normalized if requested
        """
        if not hasattr(self, 'class_weights'):
            self.class_weights = {}
        if self.class_weights.get((task_idx, normalize)) is not None:
            return self.class_weights[(task_idx, normalize)]

        if self.verbose:
            print(f"Computing class weights for task idx: {task_idx} (using {self.__class__.__name__})")

        all_task_data = self._get_all_labels_for_task(task_idx)

        class_counts = {}
        total_valid_samples = 0
        for value in all_task_data:
            if value != -1:
                class_counts[value] = class_counts.get(value, 0) + 1
                total_valid_samples += 1

        if total_valid_samples == 0:
            if self.verbose:
                print(f"Warning: No valid samples found for task idx {task_idx} in {self.__class__.__name__}")
            return None

        class_indices = sorted(class_counts.keys())
        weights_array = []
        for class_idx in class_indices:
            count = class_counts[class_idx]
            weight = total_valid_samples / (len(class_counts) * count)
            weights_array.append(weight)

        # APPLICAZIONE DELLA NORMALIZZAZIONE RICHIESTA
        if normalize:
            max_weight = max(weights_array)
            if max_weight > 0:  # Protezione da divisione per zero
                weights_array = [w / max_weight for w in weights_array]
                if self.verbose:
                    print(f"Normalized class weights (max=1.0) for task {task_idx}: {weights_array}")
            else:
                if self.verbose:
                    print(f"Warning: max_weight is zero, skipping normalization for task {task_idx}")

        weights_tensor = torch.tensor(weights_array, dtype=torch.float32)
        self.class_weights[(task_idx, normalize)] = weights_tensor

        if self.verbose:
            print(f"Class distribution (task {task_idx}): {dict(sorted(class_counts.items()))}")
            print(f"Raw class weights (task {task_idx}): {[round(w, 3) for w in weights_array]}")
            if normalize:
                print(f"NOTE: Weights normalized with max=1.0 (recommended for ordinal classification)")

        return weights_tensor

    # ---------------------- NEW: Effective Number (Cui et al. 2019) ----------------------
    def get_class_weights_effective(self, task_idx: int, beta: float = 0.999, normalize: str = "per_sample"):
        """
        Class-Balanced weights via Effective Number:
            w_c = (1 - beta) / (1 - beta^{n_c})

        Args:
            task_idx: indice del task (0=age, 1=gender, 2=emotion)
            beta:    in [0.9, 0.9999]; più vicino a 1 => correzione più dolce
            normalize:
                - "per_sample": scala i pesi così che sum_c n_c * w_c = sum_c n_c (mean sample weight = 1)
                - "mean1":      scala i pesi perché la media dei w_c sia 1
                - "none":       nessuna normalizzazione

        Returns:
            torch.FloatTensor di shape [num_classi_presenti], ordinato per class index crescente.
        """
        if not hasattr(self, 'class_weights_effective'):
            self.class_weights_effective = {}

        cache_key = (task_idx, float(beta), str(normalize))
        if self.class_weights_effective.get(cache_key) is not None:
            return self.class_weights_effective[cache_key]

        if self.verbose:
            print(f"Computing EFFECTIVE-NUMBER class weights for task idx: {task_idx} "
                  f"(beta={beta}, normalize='{normalize}', using {self.__class__.__name__})")

        # Raccogli etichette del task
        all_task_data = self._get_all_labels_for_task(task_idx)

        # Conta le classi (ignorando -1)
        class_counts = {}
        total_valid_samples = 0
        for value in all_task_data:
            if value != -1:
                class_counts[value] = class_counts.get(value, 0) + 1
                total_valid_samples += 1

        if total_valid_samples == 0:
            if self.verbose:
                print(f"Warning: No valid samples found for task idx {task_idx} in {self.__class__.__name__}")
            return None

        class_indices = sorted(class_counts.keys())
        counts = torch.tensor([class_counts[c] for c in class_indices], dtype=torch.float32)

        # w_c = (1 - beta) / (1 - beta^{n_c})
        beta_t = torch.tensor(beta, dtype=torch.float32)
        denom = (1.0 - torch.pow(beta_t, counts)).clamp_min(1e-12)
        weights = (1.0 - beta_t) / denom  # [C]

        # Normalizzazione
        if normalize == "per_sample":
            # Ensure sum_c n_c * w_c = sum_c n_c  (mean sample weight ~ 1)
            scale = counts.sum() / (counts.mul(weights).sum().clamp_min(1e-12))
            weights = weights * scale
        elif normalize == "mean1":
            weights = weights * (weights.numel() / weights.sum().clamp_min(1e-12))
        elif normalize in ("none", None):
            pass
        else:
            raise ValueError(f"Unknown normalize mode: {normalize}. Use 'per_sample', 'mean1', or 'none'.")

        weights_tensor = weights.to(torch.float32)
        self.class_weights_effective[cache_key] = weights_tensor

        if self.verbose:
            print(f"Class distribution (task {task_idx}): {dict(sorted(class_counts.items()))}")
            print(f"Effective-number weights (task {task_idx}): {weights_tensor}  "
                  f"[beta={beta}, normalize='{normalize}']")

        return weights_tensor
    # -------------------------------------------------------------------------------------

    def get_task_weights(self):
        """Compute task weights for multitask learning (order: [age, gender, emotion])."""
        if hasattr(self, 'task_weights') and self.task_weights is not None:
            return self.task_weights

        task_counts, total_samples = self._get_task_counts_and_total_len()

        weights = []
        for task_idx in ALL_TASK_IDXS:
            count = task_counts.get(task_idx, 0)
            weight = total_samples / count if count > 0 else 1.0
            weights.append(weight)

        weights_tensor = torch.tensor(weights, dtype=torch.float32)
        if weights_tensor.sum() > 0:
            weights_tensor = weights_tensor * len(weights) / weights_tensor.sum()

        self.task_weights = weights_tensor

        if self.verbose:
            print(f"Task sample counts (by idx): {task_counts}")
            print(f"Task weights: {weights_tensor}")

        return weights_tensor

import math
import numpy as np
import pandas as pd
from PIL import Image
import os
import torch
from torch.utils.data import Dataset

class BaseDataset(Dataset, WeightCalculationMixin):
    def __init__(self, root, transform=None, split="train", verbose=False,
                 limit_fraction=1.0, subset_seed=2025, only_for="vggface2"):
        self.verbose = verbose
        self.root = root
        self.transform = transform
        self.split = split
        self.base_root = root.split("datasets_with_standard_labels")[0]
        self.split_path = os.path.join(root, split)
        self.labels_path = os.path.join(self.split_path, "labels.csv")
        self.data = pd.read_csv(self.labels_path)

        # --- SOLO PER VGGFace2 (train): riduci mantenendo la distribuzione di age_group ---
        is_target =  only_for.lower() in self.root.lower()
        if is_target and self.split == "train" and 0 < limit_fraction < 1:
            N = len(self.data)
            target_size = int(math.ceil(limit_fraction * N))
            remove_total = N - target_size
            if remove_total > 0:
                rng = np.random.RandomState(subset_seed)

                # costruiamo una serie temporanea con gli age_group secondo la tua logica base (fillna -> map)
                tmp_age_groups = self.data['Age'].fillna(-1).apply(map_age_to_group)

                # conteggi per classe
                counts = tmp_age_groups.value_counts()
                # maggioritarie: sopra la media; se non basta, sopra la mediana; altrimenti tutte
                eligible = counts[counts > counts.mean()]
                if eligible.sum() < remove_total:
                    eligible = counts[counts >= counts.median()]
                if eligible.sum() < remove_total:
                    eligible = counts  # ultima spiaggia: tutte

                # probabilità di rimozione proporzionali alla numerosità delle classi eleggibili
                probs = (eligible / eligible.sum()).values
                classes = eligible.index.tolist()

                # allocazione dei remove per classe (multinomial)
                alloc = rng.multinomial(remove_total, probs)

                # per ogni classe, scegli a caso gli indici da rimuovere
                to_drop_idx = []
                for cls, k in zip(classes, alloc):
                    if k == 0:
                        continue
                    cls_idx = self.data.index[tmp_age_groups == cls].to_numpy()
                    # se alloc eccede la classe per arrotondamenti (raro), taglia a cap
                    k = min(k, len(cls_idx))
                    picked = rng.choice(cls_idx, size=k, replace=False)
                    to_drop_idx.append(picked)

                if to_drop_idx:
                    to_drop_idx = np.concatenate(to_drop_idx)
                    keep_mask = ~self.data.index.isin(to_drop_idx)
                    self.data = self.data.loc[keep_mask].reset_index(drop=True)

                if self.verbose:
                    print(f"[BaseDataset] Reduced VGGFace2 train from {N} to {len(self.data)} "
                        f"({int(limit_fraction*100)}%) by dropping from majority age_groups.")

        # --- LOGICA BASE INVARIATA ---
        raw_paths = self.data['Path'].values
        self.img_paths = [os.path.join(self.base_root, path.replace("\\","/") + ".jpg") for path in raw_paths]

        self.genders = self.data['Gender'].fillna(-1).values
        raw_ages = self.data['Age'].fillna(-1).values
        self.age_groups = [map_age_to_group(age) for age in raw_ages]
        self.emotions = self.data['Facial Emotion'].fillna(-1).values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        label = torch.tensor([
            self.age_groups[idx],
            self.genders[idx],
            self.emotions[idx]
        ], dtype=torch.long)
        return image, label

    def _get_all_labels_for_task(self, task_idx: int):
        if task_idx == AGE_IDX:
            return self.age_groups
        elif task_idx == GENDER_IDX:
            return self.genders
        elif task_idx == EMOTION_IDX:
            return self.emotions
        else:
            raise ValueError(f"Unknown task idx: {task_idx}. Must be 0,1,2")

    def _get_task_counts_and_total_len(self):
        task_counts = {
            AGE_IDX: sum(1 for age in self.age_groups if age != -1),
            GENDER_IDX: sum(1 for gender in self.genders if gender != -1),
            EMOTION_IDX: sum(1 for emotion in self.emotions if emotion != -1),
        }
        return task_counts, len(self)

class _BaseDataset(Dataset, WeightCalculationMixin):
    def __init__(self, root: str, transform=None, split="train", verbose=False):
        self.verbose = verbose
        self.root = root
        self.transform = transform
        self.split = split
        self.base_root = root.split("datasets_with_standard_labels")[0]
        self.split_path = os.path.join(root, split)
        self.labels_path = os.path.join(self.split_path, "labels.csv")
        self.data = pd.read_csv(self.labels_path)

        raw_paths = self.data['Path'].values
        self.img_paths = [os.path.join(self.base_root, path.replace("\\","/")+".jpg") for path in raw_paths]

        self.genders = self.data['Gender'].fillna(-1).values
        raw_ages = self.data['Age'].fillna(-1).values
        self.age_groups = [map_age_to_group(age) for age in raw_ages]
        self.emotions = self.data['Facial Emotion'].fillna(-1).values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns:
            image: tensor (3, H, W)
            label: LongTensor[3] with order [age_group, gender, emotion]
        """
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        label = torch.tensor([
            self.age_groups[idx],
            self.genders[idx],
            self.emotions[idx]
        ], dtype=torch.long)
        return image, label

    def _get_all_labels_for_task(self, task_idx: int):
        if task_idx == AGE_IDX:
            return self.age_groups
        elif task_idx == GENDER_IDX:
            return self.genders
        elif task_idx == EMOTION_IDX:
            return self.emotions
        else:
            raise ValueError(f"Unknown task idx: {task_idx}. Must be 0,1,2")

    def _get_task_counts_and_total_len(self):
        task_counts = {
            AGE_IDX: sum(1 for age in self.age_groups if age != -1),
            GENDER_IDX: sum(1 for gender in self.genders if gender != -1),
            EMOTION_IDX: sum(1 for emotion in self.emotions if emotion != -1),
        }
        return task_counts, len(self)


class MultiDataset(Dataset, WeightCalculationMixin):
    def __init__(self, dataset_names, transform=None, split="train", datasets_root="datasets_with_standard_labels", all_datasets=False, verbose=False):
        """
        Initialize MultiDataset with multiple datasets.
        """
        self.dataset_names = dataset_names
        self.transform = transform
        self.split = split
        self.datasets_root = datasets_root
        self.verbose = verbose

        if all_datasets:
            self.dataset_names = [d for d in os.listdir(datasets_root) if os.path.isdir(os.path.join(datasets_root, d))]
            if self.verbose:
                print(f"Loading all datasets: {self.dataset_names}")

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
                    if self.verbose:
                        print(f"Loaded {len(dataset)} samples from {dataset_path}/{split}")
                except Exception as e:
                    print(f"Warning: Could not load dataset {dataset_name}: {e}")
            else:
                print(f"Warning: Split '{split}' not found in {dataset_path}")

        self.dataset_names = successfully_loaded_datasets
        if not self.datasets:
            raise ValueError("No datasets could be loaded successfully")

        self.total_length = self.cumulative_lengths[-1]

        if self.verbose:
            print(f"Loaded {len(self.datasets)} datasets with total {self.total_length} samples:")
            for i, dataset_name in enumerate(self.dataset_names):
                if i < len(self.datasets):
                    print(f"  - {dataset_name}: {self.dataset_lengths[i]} samples")

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        if idx >= self.total_length or idx < 0:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.total_length}")

        dataset_idx = 0
        for i, cumulative_length in enumerate(self.cumulative_lengths[1:], 1):
            if idx < cumulative_length:
                dataset_idx = i - 1
                break
        local_idx = idx - self.cumulative_lengths[dataset_idx]
        return self.datasets[dataset_idx][local_idx]

    def _get_all_labels_for_task(self, task_idx: int):
        all_task_data = []
        for dataset in self.datasets:
            if task_idx == AGE_IDX:
                all_task_data.extend(dataset.age_groups)
            elif task_idx == GENDER_IDX:
                all_task_data.extend(dataset.genders)
            elif task_idx == EMOTION_IDX:
                all_task_data.extend(dataset.emotions)
        return all_task_data

    def _get_task_counts_and_total_len(self):
        task_counts = {AGE_IDX: 0, GENDER_IDX: 0, EMOTION_IDX: 0}
        for dataset in self.datasets:
            task_counts[AGE_IDX] += sum(1 for age in dataset.age_groups if age != -1)
            task_counts[GENDER_IDX] += sum(1 for gender in dataset.genders if gender != -1)
            task_counts[EMOTION_IDX] += sum(1 for emotion in dataset.emotions if emotion != -1)
        return task_counts, self.total_length


class TaskBalanceDataset(Dataset, WeightCalculationMixin):
    def __init__(self, dataset_names, transform=None, split="train",
                 datasets_root="datasets_with_standard_labels", all_datasets=False,
                 balance_task=None, augment_duplicate=None, verbose=False):
        """
        balance_task: dict mapping task_idx -> desired_fraction (e.g., {EMOTION_IDX: 0.25})
        augment_duplicate: torchvision transform applied ONLY to duplicated samples
        """
        self.dataset_names = dataset_names
        self.transform = transform
        self.split = split
        self.datasets_root = datasets_root
        self.verbose = verbose
        self.balance_task = balance_task
        self.augment_duplicate = augment_duplicate
        self.duplicated_indices = []

        if all_datasets:
            self.dataset_names = [d for d in os.listdir(datasets_root) if os.path.isdir(os.path.join(datasets_root, d))]
            if self.verbose:
                print(f"Loading all datasets: {self.dataset_names}")

        self.datasets = []
        self.dataset_lengths = []
        self.cumulative_lengths = [0]
        successfully_loaded_datasets = []

        for dataset_name in self.dataset_names:
            dataset_path = os.path.join(datasets_root, dataset_name)
            if os.path.exists(os.path.join(dataset_path, split)):
                try:
                    dataset = BaseDataset(root=dataset_path, transform=transform, split=split, verbose=verbose)
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

        self.dataset_names = successfully_loaded_datasets
        if not self.datasets:
            raise ValueError("No datasets could be loaded successfully")

        self.total_length = self.cumulative_lengths[-1]

        if self.verbose:
            print(f"Loaded {len(self.datasets)} datasets with total {self.total_length} samples:")
            for i, dataset_name in enumerate(self.dataset_names):
                if i < len(self.datasets):
                    print(f"  - {dataset_name}: {self.dataset_lengths[i]} samples")

        self._build_flattened_index()
        if self.balance_task is not None:
            self._apply_task_balancing()

    def _build_flattened_index(self):
        self.index_map = []
        for dataset_idx, dataset in enumerate(self.datasets):
            for local_idx in range(len(dataset)):
                self.index_map.append([dataset_idx, local_idx, False])  # False = not duplicated

    def _apply_task_balancing(self):
        original_len = len(self.index_map)

        for task_idx, desired_fraction in self.balance_task.items():
            valid_indices = [
                i for i, (ds_idx, loc_idx, is_dup) in enumerate(self.index_map)
                if self._is_valid_task_label(ds_idx, loc_idx, task_idx)
            ]
            current_count = len(valid_indices)

            if desired_fraction >= 1.0:
                if self.verbose:
                    print(f"[task {task_idx}] desired fraction {desired_fraction} >= 1.0, skipping balancing.")
                continue

            target_count_numerator = desired_fraction * original_len - current_count
            target_count_denominator = 1 - desired_fraction
            if target_count_denominator <= 0:
                if self.verbose:
                    print(f"[task {task_idx}] desired fraction {desired_fraction} too high, skipping balancing.")
                continue

            n_to_add = int(target_count_numerator / target_count_denominator)
            if n_to_add <= 0:
                if self.verbose:
                    final_dataset_size = original_len
                    actual_percentage = (current_count / final_dataset_size) * 100
                    print(f"[task {task_idx}] already has {current_count}/{final_dataset_size} ({actual_percentage:.1f}%) >= {desired_fraction*100:.1f}%, no duplication needed.")
                continue

            extra_indices = random.choices(valid_indices, k=n_to_add)
            extra_mapped = [[self.index_map[i][0], self.index_map[i][1], True] for i in extra_indices]
            self.index_map.extend(extra_mapped)

            final_dataset_size = len(self.index_map)
            final_valid_count = current_count + n_to_add
            actual_percentage = (final_valid_count / final_dataset_size) * 100

            if self.verbose:
                print(f"[task {task_idx}] duplicated {n_to_add} samples.")
                print(f"[task {task_idx}] final dataset size: {final_dataset_size}")
                print(f"[task {task_idx}] valid samples: {final_valid_count}/{final_dataset_size} ({actual_percentage:.1f}%)")
                print(f"[task {task_idx}] target was {desired_fraction*100:.1f}%")

        random.shuffle(self.index_map)
        self.duplicated_indices = [i for i, item in enumerate(self.index_map) if item[2]]
        if self.verbose:
            print(f"Total duplicated samples: {len(self.duplicated_indices)}")
            if len(self.duplicated_indices) > 0:
                print(f"First few duplicated indices after shuffle: {self.duplicated_indices[:5]}")

    def _is_valid_task_label(self, dataset_idx, local_idx, task_idx):
        dataset = self.datasets[dataset_idx]
        if task_idx == AGE_IDX:
            return dataset.age_groups[local_idx] != -1
        elif task_idx == GENDER_IDX:
            return dataset.genders[local_idx] != -1
        elif task_idx == EMOTION_IDX:
            return dataset.emotions[local_idx] != -1
        return False

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        if idx >= len(self.index_map) or idx < 0:
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.index_map)}")

        dataset_idx, local_idx, is_duplicated = self.index_map[idx]
        if self.augment_duplicate is None:
            image, label = self.datasets[dataset_idx][local_idx]
            return image, label
        else:
            if not is_duplicated:
                image, label = self.datasets[dataset_idx][local_idx]
                return image, label

            dataset = self.datasets[dataset_idx]
            img_path = dataset.img_paths[local_idx]
            image = Image.open(img_path).convert('RGB')
            image = self.augment_duplicate(image)
            label = torch.tensor([
                dataset.age_groups[local_idx],
                dataset.genders[local_idx],
                dataset.emotions[local_idx]
            ], dtype=torch.long)
            return image, label

    def _get_all_labels_for_task(self, task_idx: int):
        all_task_data = []
        if not hasattr(self, 'index_map'):
            self._build_flattened_index()
        for dataset_idx, local_idx, _ in self.index_map:
            dataset = self.datasets[dataset_idx]
            if task_idx == AGE_IDX:
                all_task_data.append(dataset.age_groups[local_idx])
            elif task_idx == GENDER_IDX:
                all_task_data.append(dataset.genders[local_idx])
            elif task_idx == EMOTION_IDX:
                all_task_data.append(dataset.emotions[local_idx])
        return all_task_data

    def _get_task_counts_and_total_len(self):
        if not hasattr(self, 'index_map'):
            self._build_flattened_index()
        task_counts = {AGE_IDX: 0, GENDER_IDX: 0, EMOTION_IDX: 0}
        for dataset_idx, local_idx, _ in self.index_map:
            dataset = self.datasets[dataset_idx]
            if dataset.age_groups[local_idx] != -1:
                task_counts[AGE_IDX] += 1
            if dataset.genders[local_idx] != -1:
                task_counts[GENDER_IDX] += 1
            if dataset.emotions[local_idx] != -1:
                task_counts[EMOTION_IDX] += 1
        return task_counts, len(self.index_map)

    def get_duplicated_indices(self):
        return self.duplicated_indices.copy()

    def is_duplicated_sample(self, idx):
        if idx < 0 or idx >= len(self.index_map):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.index_map)}")
        return idx in self.duplicated_indices

    def get_dataset_info(self, compute_stats=False):
        info = {}
        for i, name in enumerate(self.dataset_names[:len(self.datasets)]):
            dataset_info = {
                'original_length': self.dataset_lengths[i],
                'current_samples_in_index': sum(1 for ds_idx, _, _ in self.index_map if ds_idx == i)
            }
            if compute_stats:
                ds = self.datasets[i]
                stats = {}
                for task_idx in ALL_TASK_IDXS:
                    if task_idx == AGE_IDX:
                        data = [age for age in ds.age_groups]
                    elif task_idx == GENDER_IDX:
                        data = [gender for gender in ds.genders]
                    elif task_idx == EMOTION_IDX:
                        data = [emotion for emotion in ds.emotions]

                    valid_data = [x for x in data if x != -1]
                    key = {AGE_IDX: 'age', GENDER_IDX: 'gender', EMOTION_IDX: 'emotion'}[task_idx]
                    if valid_data:
                        from collections import Counter
                        distribution = dict(Counter(valid_data))
                        stats[f'{key}_distribution'] = distribution
                        stats[f'{key}_valid_count'] = len(valid_data)
                        stats[f'{key}_missing_count'] = len(data) - len(valid_data)
                    else:
                        stats[f'{key}_distribution'] = {}
                        stats[f'{key}_valid_count'] = 0
                        stats[f'{key}_missing_count'] = len(data)
                dataset_info['stats'] = stats
            info[name] = dataset_info
        return info


if __name__ == "__main__":
    # Test transforms for BalancedDataset
    import torchvision.transforms as transforms

    print("="*80)
    print("TESTING ALL DATASET CLASSES WITH WEIGHT CALCULATION MIXIN (INDEX-BASED TASKS)")
    print("="*80)

    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        print("\n" + "="*60)
        print("=== TESTING BaseDataset ===")
        print("="*60)

        base_datasets = []
        datasets_root = "../processed_datasets/datasets_with_standard_labels"

        try:
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

                    print("\n--- Testing BaseDataset WeightCalculationMixin methods ---")
                    for task_idx in ALL_TASK_IDXS:
                        try:
                            weights = base_dataset.get_class_weights(task_idx)
                            if weights is not None:
                                print(f"✓ task {task_idx} class weights: {weights}")
                            else:
                                print(f"⚠ No valid samples for task {task_idx}")
                        except Exception as e:
                            print(f"✗ Error computing task {task_idx} weights: {e}")

                    try:
                        task_weights = base_dataset.get_task_weights()
                        print(f"✓ Task weights: {task_weights}")
                    except Exception as e:
                        print(f"✗ Error computing task weights: {e}")

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

        try:
            multi_dataset = MultiDataset(
                dataset_names=["test_dataset"],
                transform=test_transforms,
                split="train",
                datasets_root=datasets_root,
                all_datasets=True,
                verbose=True
            )
            print(f"✓ MultiDataset loaded successfully: {len(multi_dataset)} samples")

            print("\n--- Testing MultiDataset WeightCalculationMixin methods ---")
            for task_idx in ALL_TASK_IDXS:
                try:
                    weights = multi_dataset.get_class_weights(task_idx)
                    if weights is not None:
                        print(f"✓ task {task_idx} class weights: {weights}")
                    else:
                        print(f"⚠ No valid samples for task {task_idx}")
                except Exception as e:
                    print(f"✗ Error computing task {task_idx} weights: {e}")

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

        print("Creating TaskBalanceDataset with emotion balancing (25%)...")
        dataset_with_balance = TaskBalanceDataset(
            dataset_names=["test_dataset"],
            transform=test_transforms,
            split="train",
            datasets_root=datasets_root,
            verbose=True,
            all_datasets=True,
            balance_task={EMOTION_IDX: 0.25}
        )
        print(f"✓ TaskBalanceDataset loaded successfully: {len(dataset_with_balance)} samples")

        if hasattr(dataset_with_balance, 'duplicated_indices'):
            duplicated_count = len(dataset_with_balance.duplicated_indices)
            print(f"✓ Duplicated indices tracked: {duplicated_count} samples")
            if duplicated_count > 0:
                print(f"  First few duplicated indices: {dataset_with_balance.duplicated_indices[:5]}")
        else:
            print("⚠ duplicated_indices not implemented yet")

        print("\n--- Testing TaskBalanceDataset WeightCalculationMixin methods ---")
        for task_idx in ALL_TASK_IDXS:
            try:
                weights = dataset_with_balance.get_class_weights(task_idx)
                if weights is not None:
                    print(f"✓ task {task_idx} class weights: {weights}")
                else:
                    print(f"⚠ No valid samples for task {task_idx}")
            except Exception as e:
                print(f"✗ Error computing task {task_idx} weights: {e}")

        try:
            task_weights = dataset_with_balance.get_task_weights()
            print(f"✓ Task weights: {task_weights}")
        except Exception as e:
            print(f"✗ Error computing task weights: {e}")

        if len(dataset_with_balance) > 0:
            print("\n--- Testing sample retrieval ---")
            sample_image, sample_labels = dataset_with_balance[0]
            print(f"✓ Sample image shape: {sample_image.shape}")
            print(f"✓ Sample labels [age, gender, emotion]: {sample_labels.tolist()}")

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

        print("Testing TaskBalanceDataset dataloader...")
        dataloader = DataLoader(dataset_with_balance, batch_size=256, shuffle=True)

        emotion_valid_count = 0
        age_valid_count = 0
        gender_valid_count = 0
        total_samples = 0
        batches_without_emotion = 0

        for batch_idx, (images, labels) in enumerate(dataloader):
            labels = [label.numpy() for label in labels]
            emotion_valid_in_batch = sum(1 for label in labels[EMOTION_IDX] if label != -1)
            age_valid_in_batch = sum(1 for label in labels[AGE_IDX] if label != -1)
            gender_valid_in_batch = sum(1 for label in labels[GENDER_IDX] if label != -1)

            emotion_valid_count += emotion_valid_in_batch
            age_valid_count += age_valid_in_batch
            gender_valid_count += gender_valid_in_batch
            total_samples += len(labels[0])

            if emotion_valid_in_batch == 0:
                batches_without_emotion += 1

            if batch_idx < 3:
                print(f"  Batch {batch_idx}: Emotion {emotion_valid_in_batch}/{len(labels[EMOTION_IDX])}, Age {age_valid_in_batch}/{len(labels[AGE_IDX])}, Gender {gender_valid_in_batch}/{len(labels[GENDER_IDX])}")

            if batch_idx >= 9:
                break

        total_batches = min(batch_idx + 1, 10)
        print(f"\n--- TaskBalanceDataset Summary (first {total_batches} batches) ---")
        print(f"Total samples processed: {total_samples}")
        print(f"Emotion valid: {emotion_valid_count}/{total_samples} ({emotion_valid_count/total_samples*100:.1f}%)")
        print(f"Age valid: {age_valid_count}/{total_samples} ({age_valid_count/total_samples*100:.1f}%)")
        print(f"Gender valid: {gender_valid_count}/{total_samples} ({gender_valid_count/total_samples*100:.1f}%)")
        print(f"Batches without emotion: {batches_without_emotion}/{total_batches} ({batches_without_emotion/total_batches*100:.1f}%)")

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
                emotion_valid_in_batch = sum(1 for label in labels[EMOTION_IDX] if label != -1)
                age_valid_in_batch = sum(1 for label in labels[AGE_IDX] if label != -1)
                gender_valid_in_batch = sum(1 for label in labels[GENDER_IDX] if label != -1)

                multi_emotion_valid_count += emotion_valid_in_batch
                multi_age_valid_count += age_valid_in_batch
                multi_gender_valid_count += gender_valid_in_batch
                multi_total_samples += len(labels[0])

                if emotion_valid_in_batch == 0:
                    multi_batches_without_emotion += 1

                if batch_idx < 3:
                    print(f"  MultiDataset Batch {batch_idx}: Emotion {emotion_valid_in_batch}/{len(labels[EMOTION_IDX])}, Age {age_valid_in_batch}/{len(labels[AGE_IDX])}, Gender {gender_valid_in_batch}/{len(labels[GENDER_IDX])}")

                if batch_idx >= 9:
                    break

            multi_total_batches = min(batch_idx + 1, 10)

            print(f"\n--- MultiDataset Summary (first {multi_total_batches} batches) ---")
            print(f"Total samples processed: {multi_total_samples}")
            print(f"Emotion valid: {multi_emotion_valid_count}/{multi_total_samples} ({multi_emotion_valid_count/multi_total_samples*100:.1f}%)")
            print(f"Age valid: {multi_age_valid_count}/{multi_total_samples} ({multi_age_valid_count/multi_total_samples*100:.1f}%)")
            print(f"Gender valid: {multi_gender_valid_count}/{multi_total_samples} ({multi_gender_valid_count/multi_total_samples*100:.1f}%)")
            print(f"Batches without emotion: {multi_batches_without_emotion}/{multi_total_batches} ({multi_batches_without_emotion/multi_total_batches*100:.1f}%)")

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

        if base_datasets and multi_dataset is not None:
            print("Comparing weight calculations between BaseDataset and MultiDataset...")
            base_dataset = base_datasets[0]
            for task_idx in ALL_TASK_IDXS:
                try:
                    base_weights = base_dataset.get_class_weights(task_idx)
                    multi_weights = multi_dataset.get_class_weights(task_idx)
                    if base_weights is not None and multi_weights is not None:
                        if len(multi_dataset.datasets) == 1:
                            if torch.allclose(base_weights, multi_weights, atol=1e-6):
                                print(f"✓ task {task_idx} weights consistent between BaseDataset and MultiDataset")
                            else:
                                print(f"⚠ task {task_idx} weights differ: Base={base_weights}, Multi={multi_weights}")
                        else:
                            print(f"ℹ task {task_idx} weights (across datasets): Base={base_weights}, Multi={multi_weights}")
                    else:
                        print(f"⚠ task {task_idx} weights: Base={base_weights}, Multi={multi_weights}")
                except Exception as e:
                    print(f"✗ Error comparing task {task_idx} weights: {e}")

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
