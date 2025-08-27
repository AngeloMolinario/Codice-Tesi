import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torchvision.transforms import transforms as T

from transformers import AutoConfig
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
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

def print_accuracy_table(top1_acc, top2_acc):
    """
    Stampa una tabella con l'accuracy per task (@1 e @2) e l'accuracy media sui 3 task.
    """
    avg_top1 = sum(top1_acc) / len(top1_acc)
    avg_top2 = sum(top2_acc) / len(top2_acc)
    tasks = ['age', "gender", "emotion"]

    table_data = [
        [f"Task {tasks[i]}", f"{top1_acc[i]:.4f}", f"{top2_acc[i]:.4f}"] for i in range(len(top1_acc))
    ]
    table_data.append(["Media", f"{avg_top1:.4f}", f"{avg_top2:.4f}"])

    headers = ["Task", "Top-1 Accuracy", "Top-2 Accuracy"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

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
        labels.insert(target_class, "Corretto")
        values.insert(target_class, correct_count)
        colors.insert(target_class, "green")

        # Inserisci gli errori nelle posizioni corrette
        for i in range(num_classes):
            if i != target_class:
                labels.insert(i, f"Errore verso {class_names[i]}")
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
        disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
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

def load_model(model_type, num_prompt, ckpt_path, vpt_ckpt, device):
    if model_type == 'PECoreBase':
        model = PECore_Vision(
            vision_cfg=PE_VISION_CONFIG["PE-Core-B16-224"],
            num_prompt=num_prompt
        )
        model.load_baseline(ckpt_path, device)
        return model, get_image_transform(224), PETokenizer(32)

    elif model_type == 'Siglip2Base':
        model = Siglip2Vision(
            AutoConfig.from_pretrained("google/siglip2-base-patch16-224", cache_dir="./hf_models"),
            num_prompt=num_prompt
        )
        model.load_baseline(ckpt_path, device)
        image_transforms = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        return model, image_transforms, SigLip2Tokenizer(64)

    elif model_type == 'PECoreVPT':
        model = PECore_Vision(
            vision_cfg=PE_VISION_CONFIG["PE-Core-B16-224"],
            num_prompt=num_prompt
        )
        model.load_baseline(ckpt_path, device)
        model.load_VPT_token(vpt_ckpt, device)
        return model, get_image_transform(224), PETokenizer(32)

    elif model_type == 'PECoreSoftCPT':
        model = PECore_Vision(
            vision_cfg=PE_VISION_CONFIG["PE-Core-B16-224"],
            num_prompt=0
        )
        if ckpt_path is None:
            raise ValueError("For PECoreSoftCPT you must provide --model_ckpt_path.")
        model.load_baseline(ckpt_path, device)
        return model, get_image_transform(224), PETokenizer(32)

    elif model_type == 'PECoreVPT_single':
        model = PECore_Vision(
            vision_cfg=PE_VISION_CONFIG["PE-Core-B16-224"],
            num_prompt=num_prompt
        )
        if ckpt_path is None or vpt_ckpt is None:
            raise ValueError("For PECoreVPT_single you must provide --model_ckpt_path and --vpt_ckpt_path (3 paths).")
        if isinstance(vpt_ckpt, str):
            vpt_ckpt = [p.strip() for p in vpt_ckpt.split(',')]
        if len(vpt_ckpt) != 3:
            raise ValueError("Expected 3 VPT checkpoint paths for *_single variants.")
        model.load_baseline(ckpt_path, device)
        model.load_VPT_token(vpt_ckpt[0], device)
        model.load_VPT_token(vpt_ckpt[1], device)
        model.load_VPT_token(vpt_ckpt[2], device)
        return model, get_image_transform(224), PETokenizer(32)

    elif model_type == 'Siglip2VPT':
        model = Siglip2Vision(
            AutoConfig.from_pretrained("google/siglip2-base-patch16-224", cache_dir="./hf_models"),
            num_prompt=num_prompt
        )
        if ckpt_path is None or vpt_ckpt is None:
            raise ValueError("For Siglip2VPT you must provide --model_ckpt_path and --vpt_ckpt_path.")
        model.load_baseline(ckpt_path, device)
        model.load_VPT_token(vpt_ckpt, device)
        image_transforms = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        return model, image_transforms, SigLip2Tokenizer(64)

    elif model_type == 'Siglip2VPT_single':
        model = Siglip2Vision(
            AutoConfig.from_pretrained("google/siglip2-base-patch16-224", cache_dir="./hf_models"),
            num_prompt=num_prompt
        )
        if ckpt_path is None or vpt_ckpt is None:
            raise ValueError("For Siglip2VPT_single you must provide --model_ckpt_path and --vpt_ckpt_path (3 paths).")
        if isinstance(vpt_ckpt, str):
            vpt_ckpt = [p.strip() for p in vpt_ckpt.split(',')]
        if len(vpt_ckpt) != 3:
            raise ValueError("Expected 3 VPT checkpoint paths for *_single variants.")
        model.load_baseline(ckpt_path, device)
        model.load_VPT_token(vpt_ckpt[0], device)
        model.load_VPT_token(vpt_ckpt[1], device)
        model.load_VPT_token(vpt_ckpt[2], device)
        image_transforms = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        return model, image_transforms, SigLip2Tokenizer(64)

    elif model_type == 'Siglip2SoftCPT':
        model = Siglip2Vision(
            AutoConfig.from_pretrained("google/siglip2-base-patch16-224", cache_dir="./hf_models"),
            num_prompt=0
        )
        if ckpt_path is None:
            raise ValueError("For Siglip2SoftCPT you must provide --model_ckpt_path.")
        model.load_baseline(ckpt_path, device)
        image_transforms = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        return model, image_transforms, SigLip2Tokenizer(64)

    else:
        raise NotImplementedError(f"Model type {model_type} not implemented.")

def get_text_features(text_model, tokenizer, normalize=True):
    """Crea le text features per (age, gender, emotion)."""
    PROMPTS = [
        [  # Age (9)
            [
                "a close-up portrait photo of a person who appears to be a newborn (0-2 years old), face clearly visible, neutral background",
                "a studio headshot of a person who appears to be a newborn (0-2 years old), looking at the camera",
                "a passport-style photo of a person who appears to be a newborn (0-2 years old), centered face",
                "a detailed facial photograph of a person who appears to be a newborn (0-2 years old), soft lighting",
                "a candid portrait of a person who appears to be a newborn (0-2 years old), minimal shadows"
            ],
            [
                "a close-up portrait photo of a person who appears to be a young child (3-9 years old), face clearly visible, neutral background",
                "a studio headshot of a person who appears to be a young child (3-9 years old), looking at the camera",
                "a passport-style photo of a person who appears to be a young child (3-9 years old), centered face",
                "a detailed facial photograph of a person who appears to be a young child (3-9 years old), soft lighting",
                "a candid portrait of a person who appears to be a young child (3-9 years old), minimal shadows"
            ],
            [
                "a close-up portrait photo of a person who appears to be a teenager (10-19 years old), face clearly visible, neutral background",
                "a studio headshot of a person who appears to be a teenager (10-19 years old), looking at the camera",
                "a passport-style photo of a person who appears to be a teenager (10-19 years old), centered face",
                "a detailed facial photograph of a person who appears to be a teenager (10-19 years old), soft lighting",
                "a candid portrait of a person who appears to be a teenager (10-19 years old), minimal shadows"
            ],
            [
                "a close-up portrait photo of a person who appears to be a young adult (20-29 years old), face clearly visible, neutral background",
                "a studio headshot of a person who appears to be a young adult (20-29 years old), looking at the camera",
                "a passport-style photo of a person who appears to be a young adult (20-29 years old), centered face",
                "a detailed facial photograph of a person who appears to be a young adult (20-29 years old), soft lighting",
                "a candid portrait of a person who appears to be a young adult (20-29 years old), minimal shadows"
            ],
            [
                "a close-up portrait photo of a person who appears to be an adult in their 30s (30-39 years old), face clearly visible, neutral background",
                "a studio headshot of a person who appears to be an adult in their 30s (30-39 years old), looking at the camera",
                "a passport-style photo of a person who appears to be an adult in their 30s (30-39 years old), centered face",
                "a detailed facial photograph of a person who appears to be an adult in their 30s (30-39 years old), soft lighting",
                "a candid portrait of a person who appears to be an adult in their 30s (30-39 years old), minimal shadows"
            ],
            [
                "a close-up portrait photo of a person who appears to be in their 40s (40-49 years old), face clearly visible, neutral background",
                "a studio headshot of a person who appears to be in their 40s (40-49 years old), looking at the camera",
                "a passport-style photo of a person who appears to be in their 40s (40-49 years old), centered face",
                "a detailed facial photograph of a person who appears to be in their 40s (40-49 years old), soft lighting",
                "a candid portrait of a person who appears to be in their 40s (40-49 years old), minimal shadows"
            ],
            [
                "a close-up portrait photo of a person who appears to be in their 50s (50-59 years old), face clearly visible, neutral background",
                "a studio headshot of a person who appears to be in their 50s (50-59 years old), looking at the camera",
                "a passport-style photo of a person who appears to be in their 50s (50-59 years old), centered face",
                "a detailed facial photograph of a person who appears to be in their 50s (50-59 years old), soft lighting",
                "a candid portrait of a person who appears to be in their 50s (50-59 years old), minimal shadows"
            ],
            [
                "a close-up portrait photo of a person who appears to be in their 60s (60-69 years old), face clearly visible, neutral background",
                "a studio headshot of a person who appears to be in their 60s (60-69 years old), looking at the camera",
                "a passport-style photo of a person who appears to be in their 60s (60-69 years old), centered face",
                "a detailed facial photograph of a person who appears to be in their 60s (60-69 years old), soft lighting",
                "a candid portrait of a person who appears to be in their 60s (60-69 years old), minimal shadows"
            ],
            [
                "a close-up portrait photo of a person who appears to be in their 70s or older (70+ years old), face clearly visible, neutral background",
                "a studio headshot of a person who appears to be in their 70s or older (70+ years old), looking at the camera",
                "a passport-style photo of a person who appears to be in their 70s or older (70+ years old), centered face",
                "a detailed facial photograph of a person who appears to be in their 70s or older (70+ years old), soft lighting",
                "a candid portrait of a person who appears to be in their 70s or older (70+ years old), minimal shadows"
            ]
        ],
        [  # Gender (2)
            [
                "a close-up portrait photo of an adult person who appears male-presenting, face clearly visible, plain background",
                "a studio headshot of an adult person who appears male-presenting, looking at the camera, neutral expression",
                "a passport-style photo of an adult person who appears male-presenting, centered face, plain background",
                "a detailed facial photograph of an adult person who appears male-presenting, soft lighting",
                "a candid portrait of an adult person who appears male-presenting, minimal shadows"
            ],
            [
                "a close-up portrait photo of an adult person who appears female-presenting, face clearly visible, plain background",
                "a studio headshot of an adult person who appears female-presenting, looking at the camera, neutral expression",
                "a passport-style photo of an adult person who appears female-presenting, centered face, plain background",
                "a detailed facial photograph of an adult person who appears female-presenting, soft lighting",
                "a candid portrait of an adult person who appears female-presenting, minimal shadows"
            ]
        ],
        [  # Emotion (7)
            [
                "a close-up portrait photo of a person showing surprised expression, raised brows, wide eyes, open mouth, face clearly visible",
                "a studio headshot of a person showing surprised expression, raised brows, wide eyes, open mouth, plain background",
                "a detailed facial photograph of a person showing surprised expression, raised brows, wide eyes, open mouth, soft lighting",
                "a candid portrait of a person showing surprised expression, raised brows, wide eyes, open mouth, minimal shadows",
                "a passport-style photo of a person showing surprised expression, raised brows, wide eyes, open mouth"
            ],
            [
                "a close-up portrait photo of a person showing fearful expression, wide eyes, brows raised and drawn together, slightly open mouth, face clearly visible",
                "a studio headshot of a person showing fearful expression, wide eyes, brows raised and drawn together, slightly open mouth, plain background",
                "a detailed facial photograph of a person showing fearful expression, wide eyes, brows raised and drawn together, slightly open mouth, soft lighting",
                "a candid portrait of a person showing fearful expression, wide eyes, brows raised and drawn together, slightly open mouth, minimal shadows",
                "a passport-style photo of a person showing fearful expression, wide eyes, brows raised and drawn together, slightly open mouth"
            ],
            [
                "a close-up portrait photo of a person showing disgusted expression, nose wrinkling, upper lip raised, face clearly visible",
                "a studio headshot of a person showing disgusted expression, nose wrinkling, upper lip raised, plain background",
                "a detailed facial photograph of a person showing disgusted expression, nose wrinkling, upper lip raised, soft lighting",
                "a candid portrait of a person showing disgusted expression, nose wrinkling, upper lip raised, minimal shadows",
                "a passport-style photo of a person showing disgusted expression, nose wrinkling, upper lip raised"
            ],
            [
                "a close-up portrait photo of a person showing happy expression, smile, raised cheeks, lip corners pulled up, face clearly visible",
                "a studio headshot of a person showing happy expression, smile, raised cheeks, lip corners pulled up, plain background",
                "a detailed facial photograph of a person showing happy expression, smile, raised cheeks, lip corners pulled up, soft lighting",
                "a candid portrait of a person showing happy expression, smile, raised cheeks, lip corners pulled up, minimal shadows",
                "a passport-style photo of a person showing happy expression, smile, raised cheeks, lip corners pulled up"
            ],
            [
                "a close-up portrait photo of a person showing sad expression, downturned lip corners, inner brows raised, face clearly visible",
                "a studio headshot of a person showing sad expression, downturned lip corners, inner brows raised, plain background",
                "a detailed facial photograph of a person showing sad expression, downturned lip corners, inner brows raised, soft lighting",
                "a candid portrait of a person showing sad expression, downturned lip corners, inner brows raised, minimal shadows",
                "a passport-style photo of a person showing sad expression, downturned lip corners, inner brows raised"
            ],
            [
                "a close-up portrait photo of a person showing angry expression, brows lowered and drawn together, tense lips, face clearly visible",
                "a studio headshot of a person showing angry expression, brows lowered and drawn together, tense lips, plain background",
                "a detailed facial photograph of a person showing angry expression, brows lowered and drawn together, tense lips, soft lighting",
                "a candid portrait of a person showing angry expression, brows lowered and drawn together, tense lips, minimal shadows",
                "a passport-style photo of a person showing angry expression, brows lowered and drawn together, tense lips"
            ],
            [
                "a close-up portrait photo of a person showing neutral expression, relaxed facial muscles, no pronounced expression, face clearly visible",
                "a studio headshot of a person showing neutral expression, relaxed facial muscles, no pronounced expression, plain background",
                "a detailed facial photograph of a person showing neutral expression, relaxed facial muscles, no pronounced expression, soft lighting",
                "a candid portrait of a person showing neutral expression, relaxed facial muscles, no pronounced expression, minimal shadows",
                "a passport-style photo of a person showing neutral expression, relaxed facial muscles, no pronounced expression"
            ]
        ]
    ]

    task_text_features = []
    for task_prompts in PROMPTS:
        for class_prompts in task_prompts:
            tokens = tokenizer(class_prompts).to(text_model.device)
            text_f = text_model(text=tokens, normalize=False)
            class_feat = text_f.mean(dim=0)
            class_feat = F.normalize(class_feat, dim=-1) if normalize else class_feat
            task_text_features.append(class_feat)
    text_features = torch.stack(task_text_features, dim=0)  # [9+2+7, hidden]
    return text_features

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
    total_samples = [0, 0, 0]

    all_true_labels = [[] for _ in range(3)]
    all_pred_labels = [[] for _ in range(3)]

    iterator = tqdm(dataloader) if use_tqdm else dataloader
    with torch.no_grad():
        for i, (images, labels) in enumerate(iterator):
            images = images.to(device)
            labels = labels.to(device)

            logits = model.forward(images)  # atteso: list/tuple di 3 tensori [B, C_task]

            for task_idx, task_logits in enumerate(logits):
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

            if not use_tqdm and i % 30 == 0:
                print(f"{i}/{len(dataloader)} batches processed", end='\r')

    top1_accuracy = [total_correct_top1[i] / max(1, total_samples[i]) for i in range(3)]
    top2_accuracy = [total_correct_top2[i] / max(1, total_samples[i]) for i in range(3)]
    return top1_accuracy, top2_accuracy, all_true_labels, all_pred_labels

def main(model_type, dataset_path, batch_size, output_path, use_tqdm, num_prompt, model_ckpt_path, vpt_ckpt_path, text_ckpt):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(output_path, exist_ok=True)

    # Supporta "x,y,z" per *_single
    vpt_arg = vpt_ckpt_path
    if isinstance(vpt_arg, str) and ('_single' in model_type):
        vpt_arg = [p.strip() for p in vpt_arg.split(',')]

    model, image_processor, tokenizer = load_model(
        model_type=model_type,
        num_prompt=num_prompt,
        ckpt_path=model_ckpt_path,
        vpt_ckpt=vpt_arg,
        device=device
    )
    model.to(device)

    # Carica o ricostruisce le text-features
    text_features = None
    try:
        if text_ckpt is not None and os.path.isfile(text_ckpt):
            text_features = torch.load(text_ckpt, map_location=device)
    except Exception as e:
        print(f"Warning: failed to load text features from '{text_ckpt}': {e}")

    if text_features is None:
        if hasattr(model, "text_model"):
            print("Building text features on-the-fly...")
            text_features = get_text_features(model.text_model, tokenizer, normalize=True).to(device)
        else:
            raise RuntimeError("No text features provided and model has no text_model to build them.")

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

    top1_acc, top2_acc, all_true_labels, all_pred_labels = validate(model, dataloader, device, use_tqdm)

    print_accuracy_table(top1_acc, top2_acc)

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

def argparse_args():
    parser = ArgumentParser()
    parser.add_argument('--model_type', type=str, default='PECoreBase',
                        choices=['PECoreBase', "Siglip2Base", "PECoreVPT", "Siglip2VPT",
                                 "PECoreSoftCPT", "Siglip2SoftCPT", "PECoreVPT_single", "Siglip2VPT_single"],
                        help='Type of model to use.')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for DataLoader.')
    parser.add_argument('--output_path', type=str, default='output', help='Path to save outputs like confusion matrices.')
    parser.add_argument('--no_tqdm', action='store_true', help='Disable tqdm progress bar.')
    parser.add_argument('--num_prompt', type=int, default=0, help='Number of prompt tokens to use (only for VPT models).')
    parser.add_argument('--model_ckpt_path', type=str, default=None, help='Path to the model checkpoint (only for VPT and SoftCPT models).')
    parser.add_argument('--vpt_ckpt_path', type=str, default=None,
                        help='Path to the VPT checkpoint (only for VPT models). For *_single, pass 3 paths separated by commas.')
    parser.add_argument('--text_ckpt', type=str, default='./text_features.pt', help='Path to the precomputed text features checkpoint.')
    return parser.parse_args()

if __name__ == "__main__":
    args = argparse_args()
    main(
        model_type=args.model_type,
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        output_path=args.output_path,
        use_tqdm=not args.no_tqdm,
        num_prompt=args.num_prompt,
        model_ckpt_path=args.model_ckpt_path,
        vpt_ckpt_path=args.vpt_ckpt_path,
        text_ckpt=args.text_ckpt
    )
