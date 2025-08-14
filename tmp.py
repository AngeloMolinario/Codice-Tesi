from core.vision_encoder.config import PE_VISION_CONFIG
from core.vision_encoder import transforms
from wrappers.PerceptionEncoder.pe import PECore_Vision
from PIL import Image
import torch
import os
from dataset.dataset import BaseDataset, MultiDataset, TaskBalanceDataset
from wrappers.PerceptionEncoder.pe import PECore
from wrappers.promptopt.prompt_learner import CustomModel
from training import training_functions
from torch.utils.data import DataLoader
from training.loss import *
from core.vision_encoder import transforms
from training.training_functions import *
from utils.metric import MultitaskTracker
from utils.configuration import Config
import numpy as np
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import seaborn as sns
from collections import defaultdict

# ---------------------------------------------------------
# Funzione di analisi con i 4 grafici richiesti
# ---------------------------------------------------------
def analyze_age_errors(all_preds_list, all_labels_list, all_probs_list, class_names, task_names, accuracies, output_dir):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    os.makedirs(output_dir, exist_ok=True)

    preds = all_preds_list[0].numpy()
    labels = all_labels_list[0].numpy()
    probs = all_probs_list[0].numpy()
    errors = preds - labels
    num_classes = len(class_names)

    # 1 - Distribuzione di probabilità per ogni classe (salvata separatamente)
    prob_dir = os.path.join(output_dir, "Prob_distribution_per_class")
    os.makedirs(prob_dir, exist_ok=True)
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() > 0:
            plt.figure(figsize=(8, 5))
            mean_probs = probs[mask].mean(axis=0)
            norm_counts = mean_probs / mean_probs.sum()
            plt.bar(class_names, norm_counts)
            plt.xticks(rotation=45)
            plt.xlabel("Classi")
            plt.ylabel("Frazione di campioni (normalizzata)")
            plt.title(f"Distribuzione di probabilità normalizzata - {class_names[c]}")
            plt.tight_layout()
            plt.savefig(os.path.join(prob_dir, f"class_{c}_{class_names[c]}.png"))
            plt.close()

    # 2 - Accuracy per singolo task
    plt.figure(figsize=(8, 5))
    plt.bar(task_names, accuracies)
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.01, f"{v:.2f}", ha='center')
    plt.ylabel("Accuracy")
    plt.title("Accuracy per task")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "2_accuracy_per_task.png"))
    plt.close()

    # 3 - Distribuzione errori con valori numerici
    offset_matrix = np.zeros((num_classes, 2*num_classes-1))
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() > 0:
            offsets = preds[mask] - labels[mask]
            for off in range(-(num_classes-1), num_classes):
                offset_matrix[c, off + num_classes - 1] = np.mean(offsets == off)
    plt.figure(figsize=(12, 6))
    sns.heatmap(offset_matrix, annot=True, fmt=".2f",
                xticklabels=range(-(num_classes-1), num_classes),
                yticklabels=class_names, cmap="Blues")
    plt.xlabel("Offset (Pred - Reale)")
    plt.ylabel("Classe Reale")
    plt.title("Distribuzione normalizzata degli errori (Task 0 - Age)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "3_error_distribution_with_prob.png"))
    plt.close()

    # 4 - Probabilità medie di scelta
    prob_matrix = np.zeros((num_classes, num_classes))
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() > 0:
            prob_matrix[c] = probs[mask].mean(axis=0)
    plt.figure(figsize=(10, 8))
    sns.heatmap(prob_matrix, annot=True, fmt=".2f",
                xticklabels=class_names, yticklabels=class_names, cmap="YlOrRd")
    plt.xlabel("Classe Predetta")
    plt.ylabel("Classe Reale")
    plt.title("Probabilità media di scelta (Task 0 - Age)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "4_avg_prob_matrix.png"))
    plt.close()

    print(f"[INFO] Grafici salvati in: {output_dir}")

# ---------------------------------------------------------
# Funzione di validazione aggiornata
# ---------------------------------------------------------
def multitask_val_fn(model, dataloader, loss_fn, device, task_weight, config, text_features=None):
    model.eval()
    compute_text_features = text_features is None or config.NUM_TEXT_CNTX > 0

    iterator = tqdm(dataloader) if config.USE_TQDM else dataloader

    logit_split = [len(c) for c in config.CLASSES]
    num_task = len(loss_fn)

    with torch.no_grad():
        task_weight = torch.softmax(task_weight, dim=0)

    losses_sums = torch.zeros(num_task+1, device=device)

    preds_per_task = [[] for _ in range(num_task)]
    labels_per_task = [[] for _ in range(num_task)]
    probs_per_task  = [[] for _ in range(num_task)]

    num_batch = 0
    with torch.inference_mode():
        for image, label in iterator:
            num_batch += 1
            image = image.to(device, non_blocking=True)
            labels = label.to(device, non_blocking=True)

            if compute_text_features:
                text_features = model.get_text_features(normalize=True)

            image_features = model.get_image_features(image, normalize=True)
            logits = model.logit_scale.exp() * (image_features @ text_features.T)
            logits_by_task = torch.split(logits, logit_split, dim=1)

            total_loss = 0.0

            for i in range(len(loss_fn)):
                loss_i, pred_i = loss_fn[i](logits_by_task[i], labels[:, i], return_predicted_label=True)
                losses_sums[i] += loss_i.detach()

                valid_mask = labels[:, i] != -1
                if valid_mask.any():
                    preds_per_task[i].append(pred_i[valid_mask].detach())
                    labels_per_task[i].append(labels[valid_mask, i].detach())
                    probs_per_task[i].append(torch.softmax(logits_by_task[i][valid_mask], dim=1).detach())

                total_loss = total_loss + task_weight[i] * loss_i

            losses_sums[-1] += total_loss.detach()

            if not config.USE_TQDM and (num_batch + 1) % 100 == 0:
                print(f"Processed {num_batch + 1}/{len(iterator)}", end='\r')

    accuracies = []
    all_preds_list, all_labels_list, all_probs_list = [], [], []
    for i in range(num_task):
        if preds_per_task[i]:
            all_preds  = torch.cat(preds_per_task[i],  dim=0)
            all_labels = torch.cat(labels_per_task[i], dim=0)
            all_probs  = torch.cat(probs_per_task[i], dim=0)
            acc = (all_preds == all_labels).float().mean().item()
        else:
            all_preds  = torch.empty(0, dtype=torch.long, device=device)
            all_labels = torch.empty(0, dtype=torch.long, device=device)
            all_probs  = torch.empty((0, logit_split[i]), dtype=torch.float32, device=device)
            acc = float('nan')

        accuracies.append(acc)
        all_preds_list.append(all_preds.cpu())
        all_labels_list.append(all_labels.cpu())
        all_probs_list.append(all_probs.cpu())

    mean_loss = [(losses_sums[i]/num_batch).item() for i in range(num_task+1)]

    return mean_loss, accuracies, all_preds_list, all_labels_list, all_probs_list

# ---------------------------------------------------------
# Configurazione e caricamento modello
# ---------------------------------------------------------
config = Config("/user/amolinario/Codice-Tesi/config/test.json")

output_dir = config.OUTPUT_DIR
final_dir = "../TEST_SCPT10_age/t2"

vision_ckpt = os.path.join(output_dir, "ckpt", "best_accuracy_model.pt")

if config.TUNING.lower() == 'softcpt':
    print("Inizializzazione del modello SoftCPT")
    base_model = PECore.from_config("PE-Core-B16-224", pretrained=True, num_prompt=0)
    model_ = CustomModel(
        n_ctx=config.NUM_TEXT_CNTX,
        tasknames=config.TASK_NAMES,
        classnames=config.CLASSES,
        model=base_model,
        tokenizer=transforms.get_text_tokenizer(base_model.text_model.context_length)
    ).to("cuda")
else:
    print("Inizializzazione del modello VPT")
    model_ = PECore.from_config("PE-Core-B16-224", pretrained=True, num_prompt=10).to("cuda")

checkpoint = torch.load(vision_ckpt, map_location="cuda")
missing, unexpected = model_.load_state_dict(checkpoint, strict=False)
print("Chiavi mancanti nel modello:", missing)
print("Chiavi inattese nel checkpoint:", unexpected)

text_features = None
if config.TUNING.lower() != 'softcpt':
    TEXT_CLASSES_PROMPT = config.TEXT_CLASSES_PROMPT
    tokenizer = transforms.get_text_tokenizer(model_.text_model.context_length)
    all_text_features = []
    for task_prompts in TEXT_CLASSES_PROMPT:
        text = tokenizer(task_prompts).to("cuda")
        task_text_features = model_.get_text_features(text=text, normalize=True)
        all_text_features.append(task_text_features)
    text_features = torch.cat(all_text_features, dim=0)
    model_.text_model = None
    print(f"Loaded text features into the model {text_features.shape}")
else:
    text_features = model_.get_text_features(normalize=True)
    print(f"Loaded text features into the model {text_features.shape}")

# ---------------------------------------------------------
# Dataset e dataloader
# ---------------------------------------------------------
test_dataset = MultiDataset(
    dataset_names=['UTKFace',"FairFace","LFW","MiviaGender","RAF-DB"],#['CelebA_HQ', 'FairFace', 'LFW', 'MiviaGender', 'RAF-DB', 'UTKFace', 'TestDataset'],
    transform=transforms.get_image_transform(224),
    split="test",
    datasets_root="/user/amolinario/processed_datasets/datasets_with_standard_labels/",
    all_datasets=False,
    verbose=True
)
test_dataset.get_class_weights(0)
test_dataset.get_class_weights(1)
test_dataset.get_class_weights(2)

dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2, prefetch_factor=2)

# ---------------------------------------------------------
# Loss e pesi dei task
# ---------------------------------------------------------
loss_fn = [
    MaskedLoss(OrdinalLoss(num_classes=9), -1),
    MaskedLoss(CrossEntropyLoss(), -1),
    MaskedLoss(CrossEntropyLoss(), -1)
]
task_weight = torch.ones(3, device="cuda")

# ---------------------------------------------------------
# Validazione e analisi
# ---------------------------------------------------------
losses, accuracies, all_preds_list, all_labels_list, all_probs_list = multitask_val_fn(
    model=model_,
    dataloader=dataloader,
    loss_fn=loss_fn,
    device="cuda",
    task_weight=task_weight,
    config=config,
    text_features=text_features
)

analyze_age_errors(
    all_preds_list,
    all_labels_list,
    all_probs_list,
    class_names=config.CLASSES[0],
    task_names=config.TASK_NAMES,
    accuracies=accuracies,
    output_dir=final_dir
)

# ---------------------------------------------------------
# Salvataggio metriche
# ---------------------------------------------------------
print(f"Losses:")
print(f"\tAge: {losses[0]:.6f}")
print(f"\tGender: {losses[1]:.6f}")
print(f"\tEmotion: {losses[2]:.6f}")
print(f"\tMultitask: {losses[-1]:.6f}")

print(f"Accuracy:")
print(f"\tAge: {accuracies[0]:.6f}")
print(f"\tGender: {accuracies[1]:.6f}")
print(f"\tEmotion: {accuracies[2]:.6f}")
print(f"\tMean: {sum(accuracies)/3:.6f}")

with open(os.path.join(final_dir, "results.txt"), "w") as f:
    f.write(f"Losses:\n")
    f.write(f"\tAge: {losses[0]:.6f}\n")
    f.write(f"\tGender: {losses[1]:.6f}\n")
    f.write(f"\tEmotion: {losses[2]:.6f}\n")
    f.write(f"\tMultitask: {losses[-1]:.6f}\n")
    f.write(f"Accuracy:\n")
    f.write(f"\tAge: {accuracies[0]:.6f}\n")
    f.write(f"\tGender: {accuracies[1]:.6f}\n")
    f.write(f"\tEmotion: {accuracies[2]:.6f}\n")
    f.write(f"\tMean: {sum(accuracies)/3:.6f}\n")

# ---------------------------------------------------------
# Matrici di confusione per ogni task
# ---------------------------------------------------------
task_names = ["Age", "Gender", "Emotion"]
for i, (preds, labels) in enumerate(zip(all_preds_list, all_labels_list)):
    if preds.numel() == 0 or labels.numel() == 0:
        print(f"Nessun dato valido per la matrice di confusione del task {task_names[i]}")
        continue
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title(f"Confusion Matrix - {task_names[i]}")
    plt.savefig(f"{final_dir}/confusion_matrix_{task_names[i].lower()}.png")
    plt.close(fig)
    print(f"Salvata matrice di confusione per il task {task_names[i]}")
