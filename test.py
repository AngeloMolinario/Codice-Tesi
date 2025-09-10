import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torchvision.transforms import transforms as T

from transformers import AutoConfig
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, precision_recall_fscore_support
import seaborn as sns

from argparse import ArgumentParser
from tqdm import tqdm

import numpy as np
import os
from matplotlib import pyplot as plt

from core.vision_encoder.pe import CLIP
from core.vision_encoder.config import *
from core.vision_encoder.transforms import get_image_transform

from dataset.dataset import BaseDataset

from wrappers.PerceptionEncoder.pe import PECore, PECore_Vision
# rimosso: from wrappers.SigLip2 import text   # evitato shadowing del nome "text"
from wrappers.SigLip2.SigLip2Model import Siglip2Model, Siglip2Vision
from wrappers.tokenizer import *

from tabulate import tabulate

CLASSES = [
    ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"],
    ["male", "female"],
    ["surprise", "fear", "disgust", "happy", "sad", "angry", "neutral"]
]

def print_accuracy_table(top1_acc, top2_acc, onebin_acc, outputdir=None):
    """
    Stampa una tabella con l'accuracy per task (@1 e @2) e l'accuracy media sui 3 task.
    """
    avg_top1 = sum(top1_acc) / len(top1_acc)
    avg_top2 = sum(top2_acc) / len(top2_acc)
    tasks = ['age', "gender", "emotion"]

    table_data = [
        [f"Task {tasks[i]}", f"{top1_acc[i]:.4f}", f"{top2_acc[i]:.4f}", f"{onebin_acc[i]:.4f}" if i == 0 else "N/A"] for i in range(len(top1_acc))
    ]
    table_data.append(["Media", f"{avg_top1:.4f}", f"{avg_top2:.4f}", "N/A"])

    headers = ["Task", "Top-1 Accuracy", "Top-2 Accuracy", "1-Bin Accuracy (Age)"]
    table_str = tabulate(table_data, headers=headers, tablefmt="grid")
    print(table_str)

    if outputdir is not None:
        with open(os.path.join(outputdir, "accuracy.txt"), 'w') as f:
            f.write(table_str)

    return table_str


def compute_prf_metrics(all_true_labels, all_pred_labels, classes_per_task):
    """Compute macro Precision/Recall/F1 per task and per-class PRF for Age.

    Returns:
      - prf_task: dict with keys 'precision', 'recall', 'f1' each a list len=3
      - prf_age_per_class: dict with keys 'precision', 'recall', 'f1' as lists (len = n_age_classes)
    """
    prf_task = {"precision": [0.0, 0.0, 0.0], "recall": [0.0, 0.0, 0.0], "f1": [0.0, 0.0, 0.0]}
    prf_age_per_class = None

    for task_idx in range(3):
        y_true = all_true_labels[task_idx]
        y_pred = all_pred_labels[task_idx]
        if y_true and y_pred:
            n_cls = len(classes_per_task[task_idx])
            labels = list(range(n_cls))
            try:
                p, r, f1, _ = precision_recall_fscore_support(
                    y_true, y_pred, labels=labels, average='macro', zero_division=0
                )
                prf_task["precision"][task_idx] = float(p)
                prf_task["recall"][task_idx] = float(r)
                prf_task["f1"][task_idx] = float(f1)
            except Exception:
                pass

            # For Age task (idx 0), also compute per-class PRF
            if task_idx == 0:
                try:
                    p_c, r_c, f1_c, _ = precision_recall_fscore_support(
                        y_true, y_pred, labels=labels, average=None, zero_division=0
                    )
                    prf_age_per_class = {
                        'precision': [float(x) for x in p_c],
                        'recall': [float(x) for x in r_c],
                        'f1': [float(x) for x in f1_c],
                    }
                except Exception:
                    prf_age_per_class = None
        else:
            # leave zeros if no samples
            pass

    return prf_task, prf_age_per_class


def write_final_report(outputdir, top1_acc, top2_acc, onebin_acc, age_per_class_metrics=None,
                       prf_task=None, prf_age_per_class=None):
    """
    Scrive un report finale come UN'UNICA TABELLA che contiene:
    - Righe complessive per Task (Age/Gender/Emotion) con Top-1, Top-2 e 1-Bin (solo Age)
    - Righe per ciascun age-bin con Top-1, Top-2 e 1-Bin
    """
    os.makedirs(outputdir, exist_ok=True)

    rows = []
    # Overall rows
    tasks = ['age', 'gender', 'emotion']
    for i in range(len(top1_acc)):
        prec = recall = f1 = "N/A"
        if prf_task is not None:
            try:
                prec = f"{float(prf_task['precision'][i]):.4f}"
                recall = f"{float(prf_task['recall'][i]):.4f}"
                f1 = f"{float(prf_task['f1'][i]):.4f}"
            except Exception:
                pass
        rows.append([
            f"Task {tasks[i]} (Overall)",
            f"{top1_acc[i]:.4f}",
            f"{top2_acc[i]:.4f}",
            f"{onebin_acc[i]:.4f}" if i == 0 else "N/A",
            prec,
            recall,
            f1,
        ])
    # Average row computed ONLY on Top-1 across tasks
    avg_top1 = sum(top1_acc) / max(1, len(top1_acc))
    rows.append(["Media (Top-1)", f"{avg_top1:.4f}", "N/A", "N/A", "N/A", "N/A", "N/A"])

    # Per-age-bin rows (if available)
    if age_per_class_metrics is not None and all(
        k in age_per_class_metrics for k in ("classes", "top1_accuracy", "top2_accuracy", "onebin_accuracy")
    ):
        classes = age_per_class_metrics["classes"]
        t1 = age_per_class_metrics["top1_accuracy"]
        t2 = age_per_class_metrics["top2_accuracy"]
        ob = age_per_class_metrics["onebin_accuracy"]
        # Try to attach per-class P/R/F1 if available
        p_cls = r_cls = f1_cls = None
        if prf_age_per_class is not None and all(k in prf_age_per_class for k in ("precision", "recall", "f1")):
            p_cls = prf_age_per_class["precision"]
            r_cls = prf_age_per_class["recall"]
            f1_cls = prf_age_per_class["f1"]
        for i, cname in enumerate(classes):
            pr = f"{p_cls[i]:.4f}" if p_cls is not None and i < len(p_cls) else "N/A"
            rc = f"{r_cls[i]:.4f}" if r_cls is not None and i < len(r_cls) else "N/A"
            f1v = f"{f1_cls[i]:.4f}" if f1_cls is not None and i < len(f1_cls) else "N/A"
            rows.append([f"Age Bin: {cname}", f"{t1[i]:.4f}", f"{t2[i]:.4f}", f"{ob[i]:.4f}", pr, rc, f1v])

    headers = ["Elemento", "Top-1", "Top-2", "1-Bin (Age)", "Precision", "Recall", "F1"]
    table_str = tabulate(rows, headers=headers, tablefmt="grid")

    with open(os.path.join(outputdir, "final_report.txt"), "w", encoding="utf-8") as f:
        f.write(table_str)
    print(f"\nFinal report written to {os.path.join(outputdir, 'final_report.txt')}")

    

def plot_error_distribution(all_true_labels, pred_labels, class_names, output_dir):
    """
    Crea e salva istogrammi per ogni classe vera, mostrando:
    - Numero di predizioni corrette (bin in posizione della classe corretta).
    - Numero di errori verso ogni altra classe (bin ordinati).
    
    Args:
        all_true_labels (list): etichette vere.
        pred_labels (list): etichette predette (Top-1).
        class_names (list): nomi classi del task.
        output_dir (str): directory dove salvare le immagini.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from collections import Counter

    os.makedirs(output_dir, exist_ok=True)
    y_true = np.asarray(all_true_labels, dtype=int)
    y_pred = np.asarray(pred_labels, dtype=int)
    num_classes = len(class_names)

    for target_class in range(num_classes):
        mask = (y_true == target_class)
        preds_for_class = y_pred[mask] if mask.any() else np.array([], dtype=int)
        total_samples_for_class = len(preds_for_class)

        if total_samples_for_class == 0:
            print(f"Nessun campione per la classe {class_names[target_class]}, grafico non generato.")
            continue

        # Conta le predizioni: corretti ed errori
        counts = np.bincount(preds_for_class, minlength=num_classes)
        correct_count = counts[target_class]
        error_counts = counts.copy()
        error_counts[target_class] = 0  # Rimuovi i corretti dagli errori

        # Prepara i dati per il grafico
        labels = []
        values = []
        colors = []

        # Inserisci "Corretto" nella posizione corretta
        labels.insert(target_class, f"{class_names[target_class]}")
        values.insert(target_class, correct_count)
        colors.insert(target_class, "green")

        # Inserisci gli errori nelle posizioni corrette
        for i in range(num_classes):
            if i != target_class:
                labels.insert(i, f"{class_names[i]}")
                values.insert(i, error_counts[i])
                colors.insert(i, "red")

        # Crea il grafico
        plt.figure(figsize=(12, 6))
        bars = plt.bar(labels, values, color=colors)

        # Annotazioni sopra le barre
        for bar, val in zip(bars, values):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(1, int(0.01 * max(values))),
                f"{int(val)}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold"
            )

        # Imposta la scala verticale
        max_val = max(values) if values else 0
        pad = max(1, int(0.05 * max_val)) if max_val > 0 else 1
        plt.ylim(0, max_val + pad)

        plt.title(f"Distribuzione per Classe Vera: {class_names[target_class]} (Totale campioni: {total_samples_for_class})")
        plt.xlabel("Tipo di Predizione")
        plt.ylabel("Numero di Campioni")
        plt.xticks(rotation=45, ha="right")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()

        # Salva il grafico
        file_path = os.path.join(output_dir, f"distribution_class_{target_class}_{class_names[target_class]}.png")
        plt.savefig(file_path, dpi=150)
        plt.close()
        print(f"Saved distribution plot for class {class_names[target_class]} to {file_path}")

def save_confusion_matrices_as_images(confusion_matrices, class_names_per_task, output_dir):
    """
    Salva le confusion matrix come heatmap con assi etichettati.

    Args:
        confusion_matrices (list[np.ndarray]): lista di CM per i 3 task (ordine: age, gender, emotion).
        class_names_per_task (list[list[str]]): nomi classi per ciascun task.
        output_dir (str): directory di output.
    """
    os.makedirs(output_dir, exist_ok=True)

    for task_idx, cm in enumerate(confusion_matrices):
        if cm is None:
            continue
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names_per_task[task_idx])
        fig, ax = plt.subplots(figsize=(8, 6))
        disp.plot(ax=ax, cmap='viridis', xticks_rotation=45)
        title_map = ["Age", "Gender", "Emotion"]
        plt.title(f"Confusion Matrix - {title_map[task_idx]}")
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        file_path = os.path.join(output_dir, f"confusion_matrix_{title_map[task_idx].lower()}.png")
        plt.tight_layout()
        plt.savefig(file_path, dpi=150)
        plt.close()
        print(f"Confusion Matrix for {title_map[task_idx]} saved to {file_path}")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def _discover_in_ckpt_dir(ckpt_dir: str):
    """
    From a single ckpt directory, discover useful artifacts:
    - vision checkpoint (prefer vision_ckpt.pt in ckpt_dir, else parent dir)
    - list of VPT token files (vpt_token*.pt) in ckpt_dir (sorted by name)
    - text features file (text_features*.pt) in ckpt_dir (pick a sensible one)
    """
    ckpt_dir = os.path.abspath(ckpt_dir)
    parent_dir = os.path.dirname(ckpt_dir)

    # Vision checkpoint
    vision_ckpt = None
    candidates = [
        os.path.join(ckpt_dir, "vision_ckpt.pt"),
        os.path.join(parent_dir, "vision_ckpt.pt"),
        os.path.join(parent_dir, "full_training_model.pt"),  # acceptable by our loaders
    ]
    for p in candidates:
        if os.path.isfile(p):
            vision_ckpt = p
            break

    # VPT tokens inside ckpt dir (only best-accuracy)
    vpt_tokens = []
    if os.path.isdir(ckpt_dir):
        for fn in sorted(os.listdir(ckpt_dir)):
            if fn.startswith("vpt_token") and fn.endswith(".pt") and "bval" in fn:
                vpt_tokens.append(os.path.join(ckpt_dir, fn))

    # Text features inside ckpt dir
    text_feats = None
    # Prefer only best-accuracy text features
    #bacc_path = os.path.join(ckpt_dir, "text_features_bacc.pt")
    bacc_path = os.path.join(ckpt_dir, "text_features_bval.pt")
    if not os.path.exists(bacc_path):
        bacc_path = os.path.join(ckpt_dir, "text_features.pt")
    if os.path.isfile(bacc_path):
        text_feats = bacc_path
    print(f"Discovered in '{ckpt_dir}': vision_ckpt='{vision_ckpt}', {len(vpt_tokens)} VPT tokens, text_feats='{text_feats}'")
    return vision_ckpt, vpt_tokens, text_feats


def _load_text_features_if_any(model, tokenizer, text_ckpt_path, device):
    """Load text features if a checkpoint is provided; otherwise compute on-the-fly when possible."""
    text_features = None
    if text_ckpt_path is not None and os.path.isfile(text_ckpt_path):
        try:
            obj = torch.load(text_ckpt_path, map_location=device)
            if isinstance(obj, dict) and "text_features" in obj:
                text_features = obj["text_features"]
            elif torch.is_tensor(obj):
                text_features = obj
            else:
                print(f"Warning: unknown text feature format in '{text_ckpt_path}', computing on-the-fly.")
        except Exception as e:
            print(f"Warning: failed to load text features from '{text_ckpt_path}': {e}")

    if text_features is None:
        if hasattr(model, "text_model"):
            print("Building text features on-the-fly...")
            exit(0)
        else:
            raise RuntimeError("No text features provided and model has no text_model to build them.")

    return text_features


def load_model(model_type, num_prompt, ckpt_dir, device, siglip2_repo_id="google/siglip2-base-patch16-224", pe_vision_config="PE-Core-L14-336"):
    # Discover artifacts from ckpt_dir
    vision_ckpt, vpt_tokens, text_feats_path = _discover_in_ckpt_dir(ckpt_dir)
    
    if model_type == 'PECoreBase':
        model = PECore_Vision(
            vision_cfg=PE_VISION_CONFIG[pe_vision_config],
            num_prompt=num_prompt
        )
        model.load_baseline(vision_ckpt, device)
        return model, get_image_transform(model.image_size), PETokenizer(32), text_feats_path

    elif model_type == 'Siglip2Base':
        model = Siglip2Vision(
            AutoConfig.from_pretrained(siglip2_repo_id, cache_dir="./hf_models"),
            num_prompt=num_prompt
        )
        model.load_baseline(vision_ckpt, device)
        image_transforms = T.Compose([
            T.Resize((model.image_size, model.image_size)),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])  
        return model, image_transforms, SigLip2Tokenizer(64), text_feats_path

    elif model_type == 'PECoreVPT':
        model = PECore_Vision(
            vision_cfg=PE_VISION_CONFIG[pe_vision_config],
            num_prompt=num_prompt
        )
        model.load_baseline(vision_ckpt, device)
        # Load first available VPT token, if present
        if vpt_tokens:
            try:
                model.load_VPT_token(vpt_tokens[0], device)
            except Exception as e:
                print(f"Warning: failed to load VPT token '{vpt_tokens[0]}': {e}")
        return model, get_image_transform(model.image_size), PETokenizer(32), text_feats_path

    elif model_type == 'PECoreSoftCPT':
        model = PECore_Vision(
            vision_cfg=PE_VISION_CONFIG[pe_vision_config],
            num_prompt=0
        )
        model.load_baseline(vision_ckpt, device)
        return model, get_image_transform(model.image_size), PETokenizer(32), text_feats_path

    elif model_type == 'PECoreVPT_single':
        model = PECore_Vision(
            vision_cfg=PE_VISION_CONFIG[pe_vision_config],
            num_prompt=num_prompt
        )
        model.load_baseline(vision_ckpt, device)
        # Load up to 3 tokens if available
        for idx, tok in enumerate(vpt_tokens[:3]):
            try:
                model.load_VPT_token(tok, device)
            except Exception as e:
                print(f"Warning: failed to load VPT token '{tok}': {e}")
        return model, get_image_transform(model.image_size), PETokenizer(32), text_feats_path

    elif model_type == 'Siglip2VPT':
        model = Siglip2Vision(
            AutoConfig.from_pretrained(siglip2_repo_id, cache_dir="./hf_models"),
            num_prompt=num_prompt
        )
        model.load_baseline(vision_ckpt, device)
    
        try:
            print(vpt_tokens)
            model.load_VPT_token(vpt_tokens[0], device)
        except Exception as e:
            print(f"Warning: failed to load VPT token '{vpt_tokens[0]}': {e}")
        image_transforms = T.Compose([
            T.Resize((model.image_size, model.image_size)),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        return model, image_transforms, SigLip2Tokenizer(64), text_feats_path

    elif model_type == 'Siglip2VPT_single':
        model = Siglip2Vision(
            AutoConfig.from_pretrained(siglip2_repo_id, cache_dir="./hf_models"),
            num_prompt=num_prompt
        )
        model.load_baseline(vision_ckpt, device)
        for tok in vpt_tokens[:3]:
            try:
                model.load_VPT_token(tok, device)
            except Exception as e:
                print(f"Warning: failed to load VPT token '{tok}': {e}")
        image_transforms = T.Compose([
            T.Resize((model.image_size, model.image_size)),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        return model, image_transforms, SigLip2Tokenizer(64), text_feats_path

    elif model_type == 'Siglip2SoftCPT':
        model = Siglip2Vision(
            AutoConfig.from_pretrained(siglip2_repo_id, cache_dir="./hf_models"),
            num_prompt=0
        )
        model.load_baseline(vision_ckpt, device)
        image_transforms = T.Compose([
            T.Resize((model.image_size, model.image_size)),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        return model, image_transforms, SigLip2Tokenizer(64), text_feats_path

    else:
        raise NotImplementedError(f"Model type {model_type} not implemented.")

def get_image_features(model, image, normalize=True):
    if hasattr(model, "get_image_features"):
        return model.get_image_features(image=image, normalize=normalize)
    else:
        raise NotImplementedError("The model does not have a get_image_features method.")

def compute_logit(image_features, text_features, model):
    scale_logit = 1.0
    scale_bias = 0.0
    if hasattr(model, "logit_scale"):
        scale_logit = model.logit_scale.exp()
    if hasattr(model, "logit_bias"):
        scale_bias = model.logit_bias
    return scale_logit * (image_features @ text_features.t()) + scale_bias

def validate(model, dataloader, device, use_tqdm):
    model.eval()
    total_correct_top1 = [0, 0, 0]
    total_correct_top2 = [0, 0, 0]
    total_correct_1bin = [0, 0, 0]  # Nuova metrica
    total_samples = [0, 0, 0]

    all_true_labels = [[] for _ in range(3)]
    all_pred_labels = [[] for _ in range(3)]

    # Age per-class metrics (9-bin space)
    age_num_classes = None
    age_total_per_class = None
    age_correct_top1_per_class = None
    age_correct_top2_per_class = None
    age_correct_1bin_per_class = None

    iterator = tqdm(dataloader) if use_tqdm else dataloader
    with torch.no_grad():
        for i, (images, labels) in enumerate(iterator):
            images = images.to(device)
            labels = labels.to(device)

            logits = model.forward(images)  # atteso: list/tuple di 3 tensori [B, C_task] o None

            for task_idx, task_logits in enumerate(logits):
                # Se il modello non restituisce i logits per questo task, ignoralo
                if task_logits is None:
                    continue
                task_labels = labels[:, task_idx]
                top2_preds = task_logits.topk(2, dim=-1).indices

                valid_indices = task_labels != -1
                valid_task_labels = task_labels[valid_indices]
                valid_top2_preds = top2_preds[valid_indices]

                total_correct_top1[task_idx] += (valid_top2_preds[:, 0] == valid_task_labels).sum().item()
                total_correct_top2[task_idx] += ((valid_top2_preds == valid_task_labels.unsqueeze(1)).sum(dim=-1) > 0).sum().item()
                total_samples[task_idx] += valid_task_labels.size(0)

                all_true_labels[task_idx].extend(valid_task_labels.cpu().tolist())
                all_pred_labels[task_idx].extend(valid_top2_preds[:, 0].cpu().tolist())

                # Calcolo 1-bin accuracy e per-class metrics per il task "age"
                if task_idx == 0:
                    # Initialize per-class containers lazily
                    if age_num_classes is None:
                        age_num_classes = task_logits.size(1)
                        age_total_per_class = [0] * age_num_classes
                        age_correct_top1_per_class = [0] * age_num_classes
                        age_correct_top2_per_class = [0] * age_num_classes
                        age_correct_1bin_per_class = [0] * age_num_classes

                    vt_cpu = valid_task_labels.detach().cpu()
                    vp_cpu = valid_top2_preds.detach().cpu()
                    for j in range(vt_cpu.size(0)):
                        t = int(vt_cpu[j].item())
                        p1 = int(vp_cpu[j, 0].item())
                        p2 = int(vp_cpu[j, 1].item())
                        age_total_per_class[t] += 1
                        if p1 == t:
                            age_correct_top1_per_class[t] += 1
                        if p1 == t or p2 == t:
                            age_correct_top2_per_class[t] += 1
                        if abs(p1 - t) <= 1:
                            age_correct_1bin_per_class[t] += 1
                        if abs(p1 - t) <= 1:
                            total_correct_1bin[task_idx] += 1

            if not use_tqdm and i % 30 == 0:
                print(f"{i}/{len(dataloader)} batches processed", end='\r')

    top1_accuracy = [total_correct_top1[i] / max(1, total_samples[i]) for i in range(3)]
    top2_accuracy = [total_correct_top2[i] / max(1, total_samples[i]) for i in range(3)]
    onebin_accuracy = [0.0] * 3  # Inizializza con zeri
    onebin_accuracy[0] = total_correct_1bin[0] / max(1, total_samples[0])  # Calcola solo per age

    # Build age per-class metrics dict (9-bin space)
    def _safe_div(num, den):
        return float(num) / float(den) if den and den > 0 else 0.0

    if age_num_classes is None:
        # No valid age labels encountered; fallback to current CLASSES[0]
        age_classes = list(CLASSES[0])
        n = len(age_classes)
        age_top1_acc_per_class = [0.0] * n
        age_top2_acc_per_class = [0.0] * n
        age_onebin_acc_per_class = [0.0] * n
    else:
        age_classes = list(CLASSES[0])[:age_num_classes]
        age_top1_acc_per_class = [_safe_div(c, t) for c, t in zip(age_correct_top1_per_class, age_total_per_class)]
        age_top2_acc_per_class = [_safe_div(c, t) for c, t in zip(age_correct_top2_per_class, age_total_per_class)]
        age_onebin_acc_per_class = [_safe_div(c, t) for c, t in zip(age_correct_1bin_per_class, age_total_per_class)]

    age_per_class_metrics = {
        'classes': age_classes,
        'top1_accuracy': age_top1_acc_per_class,
        'top2_accuracy': age_top2_acc_per_class,
        'onebin_accuracy': age_onebin_acc_per_class,
    }

    return top1_accuracy, top2_accuracy, onebin_accuracy, all_true_labels, all_pred_labels, age_per_class_metrics


def ValidatePaliGemma(model, dataloader, device, use_tqdm, age5_classes=None):
    # Deprecated: Paligemma evaluation removed.
    raise NotImplementedError("Paligemma evaluation has been removed.")
    """
    Computes validation metrics with age mapped from 9 bins
    ["0-2","3-9","10-19","20-29","30-39","40-49","50-59","60-69","70+"]
    into 5 bins ["0-9","10-19","20-39","40-59","60+"] and evaluates on the 5-bin space.

    Returns:
      - top1_accuracy: list[float] len=3 (per task)
      - top2_accuracy: list[float] len=3 (per task)
      - onebin_accuracy: list[float] len=3 (per task; only age is computed)
      - all_true_labels: list[list[int]] len=3 (age mapped to 5-bin indices)
      - all_pred_labels: list[list[int]] len=3 (top-1 preds; age in 5-bin indices)
      - age_per_class_metrics: {
            'classes': list[str] (5),
            'top1_accuracy': list[float] (5),
            'top2_accuracy': list[float] (5),
            'onebin_accuracy': list[float] (5)
        }
    """
    import torch
    from tqdm import tqdm

    if age5_classes is None:
        age5_classes = ["0-9", "10-19", "20-39", "40-59", "60+"]
    n_age5 = len(age5_classes)

    # Mapping: 9->5 (index-based)
    # 0:"0-2", 1:"3-9"           -> 0:"0-9"
    # 2:"10-19"                  -> 1:"10-19"
    # 3:"20-29", 4:"30-39"       -> 2:"20-39"
    # 5:"40-49", 6:"50-59"       -> 3:"40-59"
    # 7:"60-69", 8:"70+"         -> 4:"60+"
    age9_to_age5 = [0, 0, 1, 2, 2, 3, 3, 4, 4]
    age9_groups = [
        [0, 1],  # -> 0 "0-9"
        [2],     # -> 1 "10-19"
        [3, 4],  # -> 2 "20-39"
        [5, 6],  # -> 3 "40-59"
        [7, 8],  # -> 4 "60+"
    ]

    def safe_div(num, den):
        return float(num) / float(den) if den > 0 else 0.0

    model.eval()
    total_correct_top1 = [0, 0, 0]
    total_correct_top2 = [0, 0, 0]
    total_correct_1bin = [0, 0, 0]  # only meaningful for age
    total_samples = [0, 0, 0]

    # Per-class age counters in 5-bin space
    age_total_per_class = [0] * n_age5
    age_correct_top1_per_class = [0] * n_age5
    age_correct_top2_per_class = [0] * n_age5
    age_correct_1bin_per_class = [0] * n_age5

    all_true_labels = [[] for _ in range(3)]
    all_pred_labels = [[] for _ in range(3)]

    iterator = tqdm(dataloader) if use_tqdm else dataloader
    with torch.no_grad():
        for i, (images, labels) in enumerate(iterator):
            images = images.to(device)
            labels = labels.to(device)

            logits = model.forward(images)  # list/tuple of 3 tensors [B, C_task]

            for task_idx, task_logits in enumerate(logits):
                task_labels = labels[:, task_idx]
                valid_indices = task_labels != -1
                if valid_indices.sum().item() == 0:
                    continue

                valid_task_labels = task_labels[valid_indices]

                if task_idx == 0:
                    # Age task: handle 9->5 mapping if needed
                    C_age = task_logits.size(1)
                    if C_age == 9:
                        # Aggregate logits into 5 bins using log-sum-exp
                        grouped_logits = []
                        for grp in age9_groups:
                            grouped_logits.append(torch.logsumexp(task_logits[:, grp], dim=1))
                        age5_logits = torch.stack(grouped_logits, dim=1)  # [B, 5]

                        valid_age5_logits = age5_logits[valid_indices]  # [b_valid, 5]
                        top2_preds = valid_age5_logits.topk(2, dim=-1).indices  # 5-bin indices

                        # Map true labels (9->5)
                        map_tensor = torch.tensor(age9_to_age5, device=valid_task_labels.device, dtype=torch.long)
                        mapped_true = map_tensor[valid_task_labels]  # [b_valid], values in 0..4

                        # Metrics in 5-bin space
                        total_correct_top1[task_idx] += (top2_preds[:, 0] == mapped_true).sum().item()
                        total_correct_top2[task_idx] += (
                            (top2_preds == mapped_true.unsqueeze(1)).sum(dim=-1) > 0
                        ).sum().item()
                        total_samples[task_idx] += mapped_true.size(0)

                        # 1-bin overall (within ±1 bin on top-1)
                        total_correct_1bin[task_idx] += (torch.abs(top2_preds[:, 0] - mapped_true) <= 1).sum().item()

                        # Track labels/preds (mapped)
                        all_true_labels[task_idx].extend(mapped_true.detach().cpu().tolist())
                        all_pred_labels[task_idx].extend(top2_preds[:, 0].detach().cpu().tolist())

                        # Per-class accumulators
                        mt_cpu = mapped_true.detach().cpu()
                        tp_cpu = top2_preds.detach().cpu()
                        for j in range(mt_cpu.size(0)):
                            t = int(mt_cpu[j].item())
                            age_total_per_class[t] += 1
                            p1 = int(tp_cpu[j, 0].item())
                            p2 = int(tp_cpu[j, 1].item())
                            if p1 == t:
                                age_correct_top1_per_class[t] += 1
                            if p1 == t or p2 == t:
                                age_correct_top2_per_class[t] += 1
                            if abs(p1 - t) <= 1:
                                age_correct_1bin_per_class[t] += 1

                    elif C_age == n_age5:
                        # Already 5-bin model; compute directly
                        valid_logits = task_logits[valid_indices]
                        top2_preds = valid_logits.topk(2, dim=-1).indices
                        mapped_true = valid_task_labels  # already 0..4

                        total_correct_top1[task_idx] += (top2_preds[:, 0] == mapped_true).sum().item()
                        total_correct_top2[task_idx] += (
                            (top2_preds == mapped_true.unsqueeze(1)).sum(dim=-1) > 0
                        ).sum().item()
                        total_samples[task_idx] += mapped_true.size(0)
                        total_correct_1bin[task_idx] += (torch.abs(top2_preds[:, 0] - mapped_true) <= 1).sum().item()

                        all_true_labels[task_idx].extend(mapped_true.detach().cpu().tolist())
                        all_pred_labels[task_idx].extend(top2_preds[:, 0].detach().cpu().tolist())

                        mt_cpu = mapped_true.detach().cpu()
                        tp_cpu = top2_preds.detach().cpu()
                        for j in range(mt_cpu.size(0)):
                            t = int(mt_cpu[j].item())
                            age_total_per_class[t] += 1
                            p1 = int(tp_cpu[j, 0].item())
                            p2 = int(tp_cpu[j, 1].item())
                            if p1 == t:
                                age_correct_top1_per_class[t] += 1
                            if p1 == t or p2 == t:
                                age_correct_top2_per_class[t] += 1
                            if abs(p1 - t) <= 1:
                                age_correct_1bin_per_class[t] += 1
                    else:
                        raise ValueError(f"Unexpected number of age classes: {C_age}. Expected 9 or {n_age5}.")
                else:
                    # Other tasks unchanged
                    valid_logits = task_logits[valid_indices]
                    top2_preds = valid_logits.topk(2, dim=-1).indices

                    total_correct_top1[task_idx] += (top2_preds[:, 0] == valid_task_labels).sum().item()
                    total_correct_top2[task_idx] += (
                        (top2_preds == valid_task_labels.unsqueeze(1)).sum(dim=-1) > 0
                    ).sum().item()
                    total_samples[task_idx] += valid_task_labels.size(0)

                    all_true_labels[task_idx].extend(valid_task_labels.detach().cpu().tolist())
                    all_pred_labels[task_idx].extend(top2_preds[:, 0].detach().cpu().tolist())

            if not use_tqdm and i % 30 == 0:
                print(f"{i}/{len(dataloader)} batches processed", end='\r')

    top1_accuracy = [total_correct_top1[i] / max(1, total_samples[i]) for i in range(3)]
    top2_accuracy = [total_correct_top2[i] / max(1, total_samples[i]) for i in range(3)]
    onebin_accuracy = [0.0, 0.0, 0.0]
    onebin_accuracy[0] = total_correct_1bin[0] / max(1, total_samples[0])

    age_top1_acc_per_class = [safe_div(c, t) for c, t in zip(age_correct_top1_per_class, age_total_per_class)]
    age_top2_acc_per_class = [safe_div(c, t) for c, t in zip(age_correct_top2_per_class, age_total_per_class)]
    age_onebin_acc_per_class = [safe_div(c, t) for c, t in zip(age_correct_1bin_per_class, age_total_per_class)]

    age_per_class_metrics = {
        'classes': list(age5_classes),
        'top1_accuracy': age_top1_acc_per_class,
        'top2_accuracy': age_top2_acc_per_class,
        'onebin_accuracy': age_onebin_acc_per_class,
    }

    return (
        top1_accuracy,
        top2_accuracy,
        onebin_accuracy,
        all_true_labels,
        all_pred_labels,
        age_per_class_metrics,
    )

def main(model_type, dataset_path, batch_size, output_path, use_tqdm, num_prompt, ckpt_dir, pe_vision_config, siglip2_repo_id):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(output_path, exist_ok=True)

    model, image_processor, tokenizer, text_feats_path = load_model(
        model_type=model_type,
        num_prompt=num_prompt,
        ckpt_dir=ckpt_dir,
        device=device,
        pe_vision_config=pe_vision_config,
        siglip2_repo_id=siglip2_repo_id
    )
    model.to(device)

    # Carica o ricostruisce le text-features (dalla cartella ckpt)
    text_features = _load_text_features_if_any(model, tokenizer, text_feats_path, device)

    # Se il modello usa internamente le text features, allegale
    if hasattr(model, "text_features"):
        model.text_features = text_features

    print(f"Number of parameters : {count_parameters(model)}")

    dataset = BaseDataset(
        root=dataset_path,
        transform=image_processor,
        split="test",
        verbose=False
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(4, os.cpu_count() or 0)
    )
    top1_acc, top2_acc, onebin_acc, all_true_labels, all_pred_labels, age_metrics = validate(model, dataloader, device, use_tqdm)
    # Age classes remain as defined in CLASSES[0]

    # Precision/Recall/F1 (macro) per task + per-class for Age
    prf_task, prf_age_cls = compute_prf_metrics(all_true_labels, all_pred_labels, CLASSES)

    print_accuracy_table(top1_acc, top2_acc, onebin_acc, output_path)
    write_final_report(
        output_path,
        top1_acc,
        top2_acc,
        onebin_acc,
        age_per_class_metrics=age_metrics,
        prf_task=prf_task,
        prf_age_per_class=prf_age_cls,
    )

    # Confusion matrices (stampa + salvataggio immagini)
    confusion_mats = []
    task_names = ["Age", "Gender", "Emotion"]
    for task_idx in range(3):
        # Ottieni le etichette uniche presenti nei dati veri e predetti
        unique_labels = sorted(set(all_true_labels[task_idx] + all_pred_labels[task_idx]))

        # Verifica se ci sono etichette presenti
        if not unique_labels:
            print(f"No labels present for Task {task_names[task_idx]}, skipping confusion matrix.")
            confusion_mats.append(None)  # Aggiungi None per indicare che non c'è matrice
            continue

        # Costruisci la matrice di confusione solo con le etichette presenti
        cm = confusion_matrix(all_true_labels[task_idx], all_pred_labels[task_idx], labels=unique_labels)
        confusion_mats.append(cm)
        print(f"\nConfusion Matrix for Task {task_names[task_idx]}:\n{cm}")

    save_confusion_matrices_as_images(confusion_mats, CLASSES, os.path.join(output_path, "confusion_matrices"))

    # Istogrammi degli errori per ciascun task
    plot_error_distribution(all_true_labels[0], all_pred_labels[0], CLASSES[0], os.path.join(output_path, "age_error_distributions"))
    plot_error_distribution(all_true_labels[1], all_pred_labels[1], CLASSES[1], os.path.join(output_path, "gender_error_distributions"))
    plot_error_distribution(all_true_labels[2], all_pred_labels[2], CLASSES[2], os.path.join(output_path, "emotion_error_distributions"))

def _process_single_dataset(model, image_processor, device, batch_size, dataset_path, output_path, use_tqdm):
    os.makedirs(output_path, exist_ok=True)

    dataset = BaseDataset(
        root=dataset_path,
        transform=image_processor,
        split="test",
        verbose=False
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(4, os.cpu_count() or 0)
    )

    top1_acc, top2_acc, onebin_acc, all_true_labels, all_pred_labels, age_metrics = validate(model, dataloader, device, use_tqdm)

    # Precision/Recall/F1 (macro) per task + per-class for Age
    prf_task, prf_age_cls = compute_prf_metrics(all_true_labels, all_pred_labels, CLASSES)

    print_accuracy_table(top1_acc, top2_acc, onebin_acc, output_path)
    write_final_report(
        output_path,
        top1_acc,
        top2_acc,
        onebin_acc,
        age_per_class_metrics=age_metrics,
        prf_task=prf_task,
        prf_age_per_class=prf_age_cls,
    )

    confusion_mats = []
    task_names = ["Age", "Gender", "Emotion"]
    for task_idx in range(3):
        unique_labels = sorted(set(all_true_labels[task_idx] + all_pred_labels[task_idx]))
        if not unique_labels:
            print(f"No labels present for Task {task_names[task_idx]}, skipping confusion matrix.")
            confusion_mats.append(None)
            continue

        cm = confusion_matrix(all_true_labels[task_idx], all_pred_labels[task_idx], labels=unique_labels)
        confusion_mats.append(cm)
        print(f"\nConfusion Matrix for Task {task_names[task_idx]}:\n{cm}")

    save_confusion_matrices_as_images(confusion_mats, CLASSES, os.path.join(output_path, "confusion_matrices"))

    plot_error_distribution(all_true_labels[0], all_pred_labels[0], CLASSES[0], os.path.join(output_path, "age_error_distributions"))
    plot_error_distribution(all_true_labels[1], all_pred_labels[1], CLASSES[1], os.path.join(output_path, "gender_error_distributions"))
    plot_error_distribution(all_true_labels[2], all_pred_labels[2], CLASSES[2], os.path.join(output_path, "emotion_error_distributions"))

    # Return collected metrics so that multi-dataset entrypoint can build a summary
    return top1_acc, top2_acc, onebin_acc

def main_multi(model_type, dataset_paths, batch_size, output_base_path, use_tqdm, num_prompt, ckpt_dir, pe_vision_config, siglip2_repo_id):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model once
    model, image_processor, tokenizer, text_feats_path = load_model(
        model_type=model_type,
        num_prompt=num_prompt,
        ckpt_dir=ckpt_dir,
        device=device,
        pe_vision_config=pe_vision_config,
        siglip2_repo_id=siglip2_repo_id
    )
    model.to(device)

    text_features = _load_text_features_if_any(model, tokenizer, text_feats_path, device)
    if hasattr(model, "text_features"):
        model.text_features = text_features

    print(f"Number of parameters : {count_parameters(model)}")

    base_out = output_base_path if output_base_path is not None else 'output'
    os.makedirs(base_out, exist_ok=True)

    valid_paths = []
    for dp in dataset_paths:
        if os.path.isdir(dp):
            valid_paths.append(dp)
        else:
            print(f"Warning: dataset path '{dp}' does not exist. Skipping.")

    if len(valid_paths) == 0:
        raise RuntimeError("No valid dataset paths provided.")

    # Collect per-dataset accuracies for summary (Age/Gender/Emotion)
    dataset_names = []
    age_top1_list, age_top2_list = [], []
    gender_top1_list, gender_top2_list = [], []
    emotion_top1_list, emotion_top2_list = [], []

    for dp in valid_paths:
        ds_name = os.path.basename(os.path.normpath(dp))
        out_dir = os.path.join(output_base_path if output_base_path is not None else base_out, ds_name)
        print(f"\n=== Running evaluation on dataset: {ds_name} ===")
        top1_acc, top2_acc, _onebin_acc = _process_single_dataset(
            model=model,
            image_processor=image_processor,
            device=device,
            batch_size=batch_size,
            dataset_path=dp,
            output_path=out_dir,
            use_tqdm=use_tqdm,
        )

        # Store per-task accuracies (indices: 0=age,1=gender,2=emotion)
        dataset_names.append(ds_name)
        def _safe_get(lst, idx):
            try:
                return float(lst[idx])
            except Exception:
                return 0.0
        age_top1_list.append(_safe_get(top1_acc, 0))
        age_top2_list.append(_safe_get(top2_acc, 0))
        gender_top1_list.append(_safe_get(top1_acc, 1))
        gender_top2_list.append(_safe_get(top2_acc, 1))
        emotion_top1_list.append(_safe_get(top1_acc, 2))
        emotion_top2_list.append(_safe_get(top2_acc, 2))

    # Build and save summary table across datasets
    if dataset_names:
        try:
            from tabulate import tabulate
        except Exception:
            tabulate = None

        headers = ["Metric"] + dataset_names
        # Media computed only on Top-1 accuracy across tasks
        avg_top1_per_dataset = [
            (a + g + e) / 3.0 for a, g, e in zip(age_top1_list, gender_top1_list, emotion_top1_list)
        ]
        table = [
            ["Top-1 (Age)"] + [f"{v:.4f}" for v in age_top1_list],
            ["Top-1 (Gender)"] + [f"{v:.4f}" for v in gender_top1_list],
            ["Top-1 (Emotion)"] + [f"{v:.4f}" for v in emotion_top1_list],
            ["Top-2 (Age)"] + [f"{v:.4f}" for v in age_top2_list],
            ["Top-2 (Gender)"] + [f"{v:.4f}" for v in gender_top2_list],
            ["Top-2 (Emotion)"] + [f"{v:.4f}" for v in emotion_top2_list],
            ["Media (Top-1 across tasks)"] + [f"{v:.4f}" for v in avg_top1_per_dataset],
        ]

        # Write a more complete final summary file
        summary_txt_path = os.path.join(base_out, "summary_final.txt")
        try:
            if tabulate is not None:
                table_str = tabulate(table, headers=headers, tablefmt="grid")
            else:
                # Fallback simple formatting
                table_str = "\t".join(headers) + "\n" + "\n".join(["\t".join(row) for row in table])
            with open(summary_txt_path, "w", encoding="utf-8") as f:
                f.write(table_str)
            # Backward compatibility: also write to summary_age.txt with same content
            try:
                with open(os.path.join(base_out, "summary_age.txt"), "w", encoding="utf-8") as f2:
                    f2.write(table_str)
            except Exception as e2:
                print(f"Warning: failed to also write summary_age.txt: {e2}")
            print(f"\nSaved summary table to: {summary_txt_path}")
            if tabulate is not None:
                print("\nSummary (Final across datasets):")
                print(table_str)
        except Exception as e:
            print(f"Failed to write summary table to '{summary_txt_path}': {e}")

def argparse_args():
    parser = ArgumentParser()
    parser.add_argument('--model_type', type=str, default='PECoreBase',
                        choices=['PECoreBase', "Siglip2Base", "PECoreVPT", "Siglip2VPT",
                                 "PECoreSoftCPT", "Siglip2SoftCPT", "PECoreVPT_single", "Siglip2VPT_single"],
                        help='Type of model to use.')
    # Single dataset (backward compatible)
    parser.add_argument('--dataset_path', type=str, help='Path to a single dataset.')
    # Multiple datasets in one run
    parser.add_argument('--dataset_paths', type=str, nargs='+', help='Paths to multiple datasets.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for DataLoader.')
    # In multi-dataset mode, this acts as base if --output_base_path is not given
    parser.add_argument('--output_path', type=str, default='output', help='Output path (single) or base path (multi).')
    parser.add_argument('--output_base_path', type=str, help='Base output path when using --dataset_paths.')
    parser.add_argument('--no_tqdm', action='store_true', help='Disable tqdm progress bar.')
    parser.add_argument('--num_prompt', type=int, default=0, help='Number of prompt tokens to use (only for VPT models).')
    parser.add_argument('--ckpt_dir', type=str, required=True, help='Path to the ckpt directory containing saved artifacts.')
    parser.add_argument('--pe_vision_config', type=str, default='PECore-L14-336', help='PE-Vision configuration to use.')
    parser.add_argument('--siglip2_repo_id', type=str, default="google/siglip2-large-patch16-384", help='HuggingFace repo ID for SigLip-2 model.')
    args = parser.parse_args()
    if not args.dataset_path and not args.dataset_paths:
        parser.error('Please specify --dataset_path or --dataset_paths.')
    return args

if __name__ == "__main__":
    args = argparse_args()
    if args.dataset_paths:
        main_multi(
            model_type=args.model_type,
            dataset_paths=args.dataset_paths,
            batch_size=args.batch_size,
            output_base_path=args.output_base_path or args.output_path,
            use_tqdm=not args.no_tqdm,
            num_prompt=args.num_prompt,
            ckpt_dir=args.ckpt_dir,
            pe_vision_config=args.pe_vision_config,
            siglip2_repo_id=args.siglip2_repo_id,
        )
    else:
        main(
            model_type=args.model_type,
            dataset_path=args.dataset_path,
            batch_size=args.batch_size,
            output_path=args.output_path,
            use_tqdm=not args.no_tqdm,
            num_prompt=args.num_prompt,
            ckpt_dir=args.ckpt_dir,
            pe_vision_config=args.pe_vision_config,
            siglip2_repo_id=args.siglip2_repo_id,
        )
