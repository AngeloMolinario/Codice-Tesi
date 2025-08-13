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
                    preds_per_task[i].append(pred_i[valid_mask].detach())  # SU GPU
                    labels_per_task[i].append(labels[valid_mask, i].detach())  # SU GPU

                total_loss = total_loss + task_weight[i] * loss_i

            losses_sums[-1] += total_loss.detach()  

            if not config.USE_TQDM and (num_batch + 1)%100 == 0:
                print(f"Processed {num_batch + 1}/{len(iterator)}", end='\r')

    accuracies = []
    all_preds_list, all_labels_list = [], []
    for i in range(num_task):
        if preds_per_task[i]:
            all_preds  = torch.cat(preds_per_task[i],  dim=0)
            all_labels = torch.cat(labels_per_task[i], dim=0)
            acc = (all_preds == all_labels).float().mean().item()
        else:
            all_preds  = torch.empty(0, dtype=torch.long, device=device)
            all_labels = torch.empty(0, dtype=torch.long, device=device)
            acc = float('nan')

        accuracies.append(acc)
        # Porta su CPU prima di restituire
        all_preds_list.append(all_preds.cpu())
        all_labels_list.append(all_labels.cpu())

    mean_loss = [(losses_sums[i]/num_batch).item() for i in range(num_task+1)]

    return mean_loss, accuracies, all_preds_list, all_labels_list



config = Config("/user/amolinario/Codice-Tesi/config/test.json")

output_dir ="../TRAIN_AGE/EXP2_vpt_argmax" #config.OUTPUT_DIR
final_dir = "../TEST_VPT50_age"

vision_ckpt = os.path.join(output_dir, "ckpt", "best_model.pt")  # Modifica il percorso se necessario




model_ = PECore.from_config("PE-Core-B16-224", pretrained=True, num_prompt=50).to("cuda")



checkpoint = torch.load(vision_ckpt, map_location="cuda")
missing, unexpected = model_.load_state_dict(checkpoint, strict=False)
print("Chiavi mancanti nel modello:", missing)
print("Chiavi inattese nel checkpoint:", unexpected)



TEXT_CLASSES_PROMPT = config.TEXT_CLASSES_PROMPT

tokenizer = transforms.get_text_tokenizer(model_.text_model.context_length)

# Process text prompts by task to maintain task structure
all_text_features = []
for task_prompts in TEXT_CLASSES_PROMPT:
    text = tokenizer(task_prompts).to("cuda")
    task_text_features = model_.get_text_features(text=text, normalize=True)
    all_text_features.append(task_text_features)
text_features = torch.cat(all_text_features, dim=0)

print(f"Loaded text features into the model {text_features.shape}")

test_dataset = MultiDataset(
    dataset_names=["FairFace"],
    transform=transforms.get_image_transform(224),
    split="test",
    datasets_root="/user/amolinario/processed_datasets/datasets_with_standard_labels/",
    all_datasets=False,
    verbose=True
)
test_dataset.get_class_weights(0)
test_dataset.get_class_weights(1)
test_dataset.get_class_weights(2)
model_.text_model = None
dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2, prefetch_factor=2)


os.makedirs(final_dir, exist_ok=True)

from training.loss import OrdinalLoss, CrossEntropyLoss, MaskedLoss

loss_fn = [
    MaskedLoss(OrdinalLoss(num_classes=9), -1),
    MaskedLoss(CrossEntropyLoss(), -1),
    MaskedLoss(CrossEntropyLoss(), -1)
]
task_weight = torch.ones(3, device="cuda")

# Run validation and print accuracies
losses, accuracies, all_preds_list, all_labels_list = multitask_val_fn(
    model=model_,
    dataloader=dataloader,
    loss_fn=loss_fn,
    device="cuda",
    task_weight=task_weight,
    config=config,
    text_features=text_features
)

print(f"Losses:")
print(f"\tAge: {losses[0]:.6f}")
print(f"\tGender: {losses[1]:.6f}")
print(f"\tEmotion: {losses[2]:.6f}")
print(f"\tMultitask: {losses[-1]:.6f}")

print(f"Accuracy:")
print(f"\tAge: {accuracies[0]:.6f}")
print(f"\tGender: {accuracies[1]:.6f}")
print(f"\tEmotion: {accuracies[2]:.6f}")
print(f"\tMean: {np.nanmean(accuracies):.6f}")


# Crea e salva le matrici di confusione
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