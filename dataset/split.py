import os
import numpy as np
from collections import defaultdict, Counter
import random
from dataset import BaseDataset, MultiDataset

def intelligent_split(dataset, train_ratio=0.8, random_seed=42, verbose=True):
    """
    Esegue uno split stratificato avanzato per dataset multi-task.
    1. Identifica tutte le combinazioni uniche di etichette (es. maschio-felice-giovane).
    2. Assicura che almeno un campione per ogni combinazione unica sia nel set di validazione.
    3. Raggiunge il ratio di split desiderato aggiungendo campioni in modo proporzionale
       per mantenere la distribuzione originale del dataset.
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"INTELLIGENT SPLITTING (Stratified Combinations): {type(dataset).__name__}")
        print(f"{'='*80}")
        print(f"Target train ratio: {train_ratio:.2f}")

    random.seed(random_seed)
    np.random.seed(random_seed)

    try:
        total_samples = len(dataset)
        if total_samples == 0:
            print("Dataset is empty. Cannot perform split.")
            return [], [], {}

        # Step 1: Estrarre etichette e raggruppare indici per combinazione di etichette
        combination_indices = defaultdict(list)
        for idx in range(total_samples):
            labels = {}
            if hasattr(dataset, 'age_groups'):  # Per BaseDataset
                labels['age'] = dataset.age_groups[idx]
                labels['gender'] = dataset.genders[idx]
                labels['emotion'] = dataset.emotions[idx]
            else:  # Per MultiDataset
                sample = dataset[idx]
                if isinstance(sample, tuple) and len(sample) >= 2 and isinstance(sample[1], dict):
                    sample_labels_data = sample[1]
                    labels['age'] = sample_labels_data.get('age_group', -1)
                    labels['gender'] = sample_labels_data.get('gender', -1)
                    labels['emotion'] = sample_labels_data.get('emotion', -1)
            
            # Crea una tupla immutabile per la combinazione di etichette
            label_combination = (labels.get('age', -1), labels.get('gender', -1), labels.get('emotion', -1))
            combination_indices[label_combination].append(idx)

        if verbose:
            print(f"\nFound {len(combination_indices)} unique label combinations.")
            # Opzionale: stampare alcune combinazioni per verifica
            # for combo, indices in list(combination_indices.items())[:5]:
            #     print(f"  - Combo {combo}: {len(indices)} samples")

        # Step 2: Garantire almeno un campione per ogni combinazione nel validation set
        val_indices = set()
        # Usiamo una copia degli indici per poterli modificare
        available_indices_per_combo = {combo: list(indices) for combo, indices in combination_indices.items()}

        for combo, indices in available_indices_per_combo.items():
            if indices:
                random.shuffle(indices)
                guaranteed_idx = indices.pop(0) # Prende e rimuove il primo indice
                val_indices.add(guaranteed_idx)

        if verbose:
            print(f"Guaranteed {len(val_indices)} samples in validation set to cover all combinations.")

        # Step 3: Riempimento proporzionale del validation set fino al ratio desiderato
        target_val_size = round(total_samples * (1 - train_ratio))
        
        # Pool di tutti gli indici rimanenti disponibili
        remaining_pool = []
        for combo, indices in available_indices_per_combo.items():
            remaining_pool.extend(indices)
        
        random.shuffle(remaining_pool)

        num_to_add = target_val_size - len(val_indices)
        if num_to_add > 0:
            # Aggiunge i campioni mancanti dal pool di rimanenti
            additional_indices = remaining_pool[:num_to_add]
            val_indices.update(additional_indices)

        # Step 4: Crea i set finali di indici
        all_indices = set(range(total_samples))
        val_indices_list = sorted(list(val_indices))
        train_indices = sorted(list(all_indices - set(val_indices_list)))

        final_train_size = len(train_indices)
        final_val_size = len(val_indices_list)
        final_train_ratio = final_train_size / total_samples if total_samples > 0 else 0

        if verbose:
            print(f"\n--- FINAL SPLIT RESULTS ---")
            print(f"Target validation size: {target_val_size}")
            print(f"Training:   {final_train_size} samples ({final_train_ratio:.2%})")
            print(f"Validation: {final_val_size} samples ({(1-final_train_ratio):.2%})")

            # --- Calculate and print class distributions ---
            def get_labels_for_index(idx, ds):
                labels = {}
                if hasattr(ds, 'age_groups'):  # For BaseDataset
                    labels['age'] = ds.age_groups[idx]
                    labels['gender'] = ds.genders[idx]
                    labels['emotion'] = ds.emotions[idx]
                else:  # For MultiDataset
                    sample = ds[idx]
                    if isinstance(sample, tuple) and len(sample) >= 2 and isinstance(sample[1], dict):
                        sample_labels_data = sample[1]
                        labels['age'] = sample_labels_data.get('age_group', -1)
                        labels['gender'] = sample_labels_data.get('gender', -1)
                        labels['emotion'] = sample_labels_data.get('emotion', -1)
                return labels

            train_dist = defaultdict(Counter)
            for idx in train_indices:
                labels = get_labels_for_index(idx, dataset)
                for task, label in labels.items():
                    train_dist[task][label] += 1

            val_dist = defaultdict(Counter)
            for idx in val_indices_list:
                labels = get_labels_for_index(idx, dataset)
                for task, label in labels.items():
                    val_dist[task][label] += 1

            print("\n--- Train Set Class Distribution ---")
            for task, counts in sorted(train_dist.items()):
                print(f"  Task '{task}':")
                for label, count in sorted(counts.items()):
                    print(f"    - Class {label}: {count} samples")

            print("\n--- Validation Set Class Distribution ---")
            for task, counts in sorted(val_dist.items()):
                print(f"  Task '{task}':")
                for label, count in sorted(counts.items()):
                    print(f"    - Class {label}: {count} samples")

        split_info = {
            'dataset_type': type(dataset).__name__,
            'total_samples': total_samples,
            'train_samples': final_train_size,
            'val_samples': final_val_size,
            'target_train_ratio': train_ratio,
            'actual_train_ratio': final_train_ratio,
            'train_indices': train_indices,
            'val_indices': val_indices_list,
            'unique_combinations': len(combination_indices)
        }

        return train_indices, val_indices_list, split_info

    except Exception as e:
        if verbose:
            import traceback
            print(f"Error during intelligent split: {e}")
            traceback.print_exc()
        return None, None, None

# Example usage:
if __name__ == "__main__":
    from dataset import BaseDataset, MultiDataset
    
    # Example with BaseDataset
    dataset = BaseDataset(
        root="../datasets_with_standard_labels/RAF-DB", 
        split="train"
    )
    
    train_indices, val_indices, split_info = intelligent_split(
        dataset=dataset,
        train_ratio=0.9,
        random_seed=42,
        verbose=True
    )
    
    if train_indices is not None:
        print(f"\nSplit completed successfully!")
        print(f"Train indices: {len(train_indices)} samples")
        print(f"Val indices: {len(val_indices)} samples")
        print(f"Unique combinations found: {split_info['unique_combinations']}")
    print("\n\n\n")
    print("="*15)
    # Example with MultiDataset
    multi_dataset = MultiDataset(
        dataset_names=[],
        datasets_root="../datasets_with_standard_labels",
        split="train",
        all_datasets=True
    )
    
    train_indices, val_indices, split_info = intelligent_split(
        dataset=multi_dataset,
        train_ratio=0.9,
        random_seed=42,
        verbose=True
    )
    if train_indices is not None:
        print(f"\nSplit completed successfully!")
        print(f"Train indices: {len(train_indices)} samples")
        print(f"Val indices: {len(val_indices)} samples")
        print(f"Unique combinations found: {split_info['unique_combinations']}")
