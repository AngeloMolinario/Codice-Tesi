import torch
from torchvision.transforms import transforms as T
import os

from transformers import AutoConfig

from core.vision_encoder import transforms
from wrappers.PerceptionEncoder.pe import PECore
from wrappers.promptopt.prompt_learner import CustomModel
from wrappers.SigLip2.SigLip2Model import Siglip2Model
from wrappers.tokenizer import PETokenizer, SigLip2Tokenizer

from dataset.dataset import BaseDataset, MultiDataset, TaskBalanceDataset

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def get_image_transform(config):
    '''
        Get the image transformation used during the pretraining of the specific model.
    '''
    model_name = config.MODEL.lower()
    if model_name == 'pecore':
        return transforms.get_image_transform(224)
    elif model_name == 'siglip2':
        return T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
        ])

    raise ValueError(f"Unknown model name: {model_name}")

def get_tokenizer(config):
    ''' Get the tokenizer for the specific model.
        The tokenizer used are wrapped in a specific class so to use the correct one for each model with the correct initialization parameters
    '''
    if config.MODEL.lower() == 'pecore':
        return PETokenizer.get_instance(32) # The PECore model used has a context length of 32
    elif config.MODEL.lower() == 'siglip2':
        return SigLip2Tokenizer.get_instance(64) # The Siglip2 model used has a contenxt length of 64

def get_model(config):
    ''' This method look at the configuration file and return the correct model initialized with pretrained weights and the specified attributes'''
    tuning = config.TUNING.lower()
    model_name = config.MODEL.lower()

    if tuning == "softcpt":
        if model_name == "pecore":
            base_model = PECore.from_config("PE-Core-B16-224", pretrained=True, num_prompt=config.NUM_VISUAL_PROMPT)
            model = CustomModel(
                n_ctx=config.NUM_TEXT_CNTX,
                tasknames=config.TASK_NAMES,
                classnames=config.CLASSES,
                model=base_model,
                tokenizer=transforms.get_text_tokenizer(base_model.text_model.context_length)
            )
            return model
        
        elif model_name == "siglip2":
            base_model = Siglip2Model(
                config=AutoConfig.from_pretrained("google/siglip2-base-patch16-224", cache_dir="./hf_models"),
                num_prompts=config.NUM_VISUAL_PROMPT
            )
            base_model.load_model(path="./hf_models/model.pth", map_location="cpu")
            model = CustomModel(
                n_ctx=config.NUM_TEXT_CNTX,
                tasknames=config.TASK_NAMES,
                classnames=config.CLASSES,
                model=base_model,
                tokenizer=get_tokenizer(config)
            )
            return model
        else:
            raise ValueError(f"Unknown model name: {model_name}")

    elif tuning == "vpt":
        if model_name == "pecore":
            model = PECore.from_config("PE-Core-B16-224", pretrained=True, num_prompt=config.NUM_VISUAL_PROMPT)
            return model
        
        elif model_name == "siglip2":
            model = Siglip2Model(
                config = AutoConfig.from_pretrained("google/siglip2-base-patch16-224", cache_dir="./hf_models"),
                num_prompts=config.NUM_VISUAL_PROMPT
            )
            model.load_model(path="./hf_models/model.pth", map_location="cpu")
            return model
        
        else:
            raise ValueError(f"Unknown model name: {model_name}")
    
    elif tuning == "vvpt":
        raise NotImplementedError(f"Model {model_name} is not implemented for VVPT tuning.")
    else:
        raise ValueError(f"Unknown tuning method: {tuning}")

def get_dataset(config, split, transform=None, augmentation_transform=None):

    balance_task = None
    if hasattr(config, "BALANCE_TASK"):
        balance_task = config.BALANCE_TASK
    task = config.TASK
    datasets = config.DATASET_NAMES[task] # Take the dataset names for the specific task
    
    if len(datasets) == 1:
        return BaseDataset(
            root=os.path.join(config.DATASET_ROOT, datasets[0]),
            transform=transform,
            split=split,
            verbose=config.VERBOSE
        )
    else:
        if int(task) == -1:
            # This is the only case in which balancing the dataset may be needed
            if balance_task is not None and split=="train":
                return TaskBalanceDataset(
                    dataset_names=datasets,
                    transform=transform,
                    split=split,
                    datasets_root=config.DATASET_ROOT,
                    all_datasets=len(datasets) == 0, # if the dataset name list is empty load all the dataset in the root folder
                    balance_task=balance_task,
                    augment_duplicate=augmentation_transform,
                    verbose=config.VERBOSE
                )
        # The multitask doesn't need to be balanced, or i'm in a specific task scenario, or i want the validation set
        return MultiDataset(
            dataset_names=datasets,
            transform=transform,
            split=split,
            datasets_root=config.DATASET_ROOT,
            all_datasets=len(datasets) == 0, # if the dataset name list is empty load all the dataset in the root folder
            verbose=config.VERBOSE
        )    

def analyze_age_errors(all_preds_list, all_labels_list, all_probs_list, class_names, task_names, accuracies, output_dir):
    """
    Genera i grafici di analisi errori per l'età (task 0) e salva anche un file .npy
    con tutti i dati necessari per ricreare i grafici.

    File salvato: {output_dir}/analysis_data.npy
    Contenuti principali salvati:
      - class_names, task_names, accuracies
      - per-class:
          n_samples, mean_probs, argmax_counts, argmax_norm_counts
      - offset_matrix (+ ticks)
      - prob_matrix
    """

    os.makedirs(output_dir, exist_ok=True)

    preds = all_preds_list[0].numpy()
    labels = all_labels_list[0].numpy()
    probs = all_probs_list[0].numpy()
    num_classes = len(class_names)

    # Helper per annotare i valori sopra le barre e bloccare y in [0, 1]
    def _annotate_bars_and_fix_ylim(ax, bars):
        ax.set_ylim(0, 1)
        for rect in bars:
            h = rect.get_height()
            # Posiziona l'etichetta poco sopra la barra ma entro il limite 1.0
            y = min(h + 0.02, 0.98)
            ax.text(rect.get_x() + rect.get_width() / 2.0, y, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=9)

    # --- 1 - Distribuzione di probabilità  per ogni classe (salvata separatamente)
    prob_dir = os.path.join(output_dir, "Prob_distribution_per_class")
    os.makedirs(prob_dir, exist_ok=True)

    # Dati da salvare
    per_class_data = []
    n_samples_per_class = np.zeros(num_classes, dtype=int)

    for c in range(num_classes):
        mask = labels == c
        n_c = int(mask.sum())
        n_samples_per_class[c] = n_c

        mean_probs = None
        norm_counts = None

        if n_c > 0:
            mean_probs = probs[mask].mean(axis=0)  # [num_classes]
            denom = mean_probs.sum()
            norm_counts = (mean_probs / denom) if denom > 0 else np.zeros_like(mean_probs)

            plt.figure(figsize=(8, 5))
            ax = plt.gca()
            bars = ax.bar(class_names, norm_counts)
            plt.xticks(rotation=45)
            plt.xlabel("Classi")
            plt.ylabel("Frazione di campioni (normalizzata)")
            plt.title(f"Distribuzione di probabilitÃ  normalizzata - {class_names[c]}")
            _annotate_bars_and_fix_ylim(ax, bars)
            plt.tight_layout()
            plt.savefig(os.path.join(prob_dir, f"class_{c}_{class_names[c]}.png"))
            plt.close()

        per_class_data.append({
            "class_index": c,
            "class_name": class_names[c],
            "n_samples": n_c,
            "mean_probs": None if mean_probs is None else mean_probs.astype(np.float32),
            "norm_counts": None if norm_counts is None else norm_counts.astype(np.float32),
        })

    # --- 1b - Distribuzione dell'argmax per ogni classe reale
    argmax_dir = os.path.join(output_dir, "Argmax_distribution_per_class")
    os.makedirs(argmax_dir, exist_ok=True)

    for c in range(num_classes):
        mask = labels == c
        if mask.sum() > 0:
            pred_counts = np.bincount(preds[mask], minlength=num_classes)
            norm_pred_counts = pred_counts / pred_counts.sum() if pred_counts.sum() > 0 else np.zeros(num_classes, dtype=float)

            plt.figure(figsize=(8, 5))
            ax = plt.gca()
            bars = ax.bar(class_names, norm_pred_counts)
            plt.xticks(rotation=45)
            plt.xlabel("Classi Predette (Argmax)")
            plt.ylabel("Frazione di campioni")
            plt.title(f"Distribuzione argmax predetto - {class_names[c]}")
            _annotate_bars_and_fix_ylim(ax, bars)
            plt.tight_layout()
            plt.savefig(os.path.join(argmax_dir, f"class_{c}_{class_names[c]}.png"))
            plt.close()

            # arricchisci il blocco per-class con conteggi argmax
            per_class_data[c]["argmax_counts"] = pred_counts.astype(np.int32)
            per_class_data[c]["argmax_norm_counts"] = norm_pred_counts.astype(np.float32)
        else:
            per_class_data[c]["argmax_counts"] = None
            per_class_data[c]["argmax_norm_counts"] = None

    # --- 2 - Accuracy per singolo task
    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    bars = ax.bar(task_names, accuracies)
    # Annotazioni sulle barre con y in [0,1]
    ax.set_ylim(0, 1)
    for i, rect in enumerate(bars):
        v = rect.get_height()
        y = min(v + 0.02, 0.98)
        ax.text(rect.get_x() + rect.get_width() / 2.0, y, f"{v:.2f}", ha='center', va='bottom')
    plt.ylabel("Accuracy")
    plt.title("Accuracy per task")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "2_accuracy_per_task.png"))
    plt.close()

    # --- 3 - Distribuzione errori con valori numerici
    offset_ticks = np.arange(-(num_classes - 1), num_classes)
    offset_matrix = np.zeros((num_classes, 2 * num_classes - 1), dtype=np.float32)
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() > 0:
            offsets = preds[mask] - labels[mask]
            for j, off in enumerate(offset_ticks):
                offset_matrix[c, j] = np.mean(offsets == off)

    plt.figure(figsize=(12, 6))
    sns.heatmap(offset_matrix, annot=True, fmt=".2f",
                xticklabels=offset_ticks,
                yticklabels=class_names, cmap="Blues")
    plt.xlabel("Offset (Pred - Reale)")
    plt.ylabel("Classe Reale")
    plt.title("Distribuzione normalizzata degli errori (Task 0 - Age)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "3_error_distribution_with_prob.png"))
    plt.close()

    # --- 3b - Distribuzione errori con valori numerici (non normalizzata)
    offset_matrix_counts = np.zeros((num_classes, 2 * num_classes - 1), dtype=np.int32)
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() > 0:
            offsets = preds[mask] - labels[mask]
            for j, off in enumerate(offset_ticks):
                offset_matrix_counts[c, j] = np.sum(offsets == off)

    plt.figure(figsize=(12, 6))
    sns.heatmap(offset_matrix_counts, annot=True, fmt="d",
                xticklabels=offset_ticks,
                yticklabels=class_names, cmap="Blues")
    plt.xlabel("Offset (Pred - Reale)")
    plt.ylabel("Classe Reale")
    plt.title("Distribuzione assoluta degli errori (Task 0 - Age)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "3b_error_distribution_absolute.png"))
    plt.close()

    # --- 4 - ProbabilitÃ  medie di scelta
    prob_matrix = np.zeros((num_classes, num_classes), dtype=np.float32)
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() > 0:
            prob_matrix[c] = probs[mask].mean(axis=0)

    plt.figure(figsize=(10, 8))
    sns.heatmap(prob_matrix, annot=True, fmt=".2f",
                xticklabels=class_names, yticklabels=class_names, cmap="YlOrRd")
    plt.xlabel("Classe Predetta")
    plt.ylabel("Classe Reale")
    plt.title("ProbabilitÃ  media di scelta (Task 0 - Age)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "4_avg_prob_matrix.png"))
    plt.close()

    # ===== Salvataggio dati necessari per ricreare i grafici =====
    data_to_save = {
        "class_names": list(class_names),
        "task_names": list(task_names),
        "accuracies": np.array(accuracies, dtype=np.float32),
        "num_classes": int(num_classes),
        "n_samples_per_class": n_samples_per_class,
        "per_class": per_class_data,                 # lista di dict (vedi sopra)
        "offset_matrix": offset_matrix,              # [C, 2C-1]
        "offset_ticks": offset_ticks,                # [-C+1, ..., C-1]
        "prob_matrix": prob_matrix,                  # [C, C]
        # opzionali utili per ricostruzioni complete:
        "preds": preds.astype(np.int32),
        "labels": labels.astype(np.int32),
        # evitiamo di salvare 'probs' complete se sono enormi; decommenta se vuoi raw probs:
        "probs": probs.astype(np.float32),
    }
    np.save(os.path.join(output_dir, "analysis_data.npy"), data_to_save, allow_pickle=True)

    print(f"[INFO] Grafici salvati in: {output_dir}")
    print(f"[INFO] Dati per i grafici salvati in: {os.path.join(output_dir, 'analysis_data.npy')}")
