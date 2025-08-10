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



CLASSES =[
        ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"],
        ["male", "female"],
        ["surprise", "fear", "disgust", "happy", "sad", "angry", "neutral"]
    ]

def get_prediction(age, gender, emotion):
    age_logit = age
    gender_logit = gender
    emotion_logit = emotion

    age_prob = torch.softmax(age_logit, dim=1)
    age_pred = torch.round(torch.sum(age_prob * torch.arange(9, device=age_prob.device).unsqueeze(0), dim=1))
    gender_pred = torch.argmax(gender_logit, dim=1)
    emotion_pred = torch.argmax(emotion_logit, dim=1)
    return age_pred, gender_pred, emotion_pred


def load_model(text_features_path, vpt_files):

    vision_cfg = PE_VISION_CONFIG["PE-Core-B16-224"]
    num_prompt = 40
    num_classes = (9, 2, 7)
    device = "cuda"
    # 2. Istanzia il modello con 3 VPT
    model = PECore_Vision(
        vision_cfg=vision_cfg,
        num_prompt=num_prompt,
        num_classes=num_classes,
        num_vpt=len(vpt_files)
    ).to(device)

    # 3. Carica vision encoder pretrained (pesi di base, senza VPT)
    model.load_pretrained_vision_encoder(name="PE-Core-B16-224")#, checkpoint_path=vision_ckpt)

    # 4. Carica i 3 VPT da tre file diversi
    model.load_vpt_checkpoints(vpt_files)

    # 5. Carica le text features precompute
    model.load_text_features(text_features_path, device=device)

    # 6. Inferenza: ottieni una lista di output, uno per ogni contesto VPT
    model.eval()
    return model, device

output_dir = "./../TRAIN/EXP2/"

vision_ckpt = os.path.join(output_dir, "ckpt", "best_model.pt")  # Modifica il percorso se necessario
text_features_path = os.path.join(output_dir, "ckpt", "vpt_text_features.pt")
vpt_files = [
    os.path.join(output_dir, "ckpt", f"best_model.pt")
]

model, device = load_model( text_features_path, vpt_files)


test_dataset = MultiDataset(
    dataset_names=["UTKFace", "RAF-DB"],#['LFW', 'MiviaGender', 'RAF-DB', 'UTKFace', "FairFace"],
    transform=transforms.get_image_transform(224),
    split="test",
    datasets_root="/user/amolinario/processed_datasets/datasets_with_standard_labels/",
    all_datasets=False,
    verbose=True
)

dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

os.makedirs("../TEST", exist_ok=True)

print(f"TEXT FEATURES {model.text_features.shape}")

# Initialize as empty tensors on GPU
all_age_preds = []
all_age_labels = []
all_gender_preds = []
all_gender_labels = []
all_emotion_preds = []
all_emotion_labels = []

with torch.inference_mode():
    for image, label in tqdm(dataloader):
        image = image.to(device)
        label = label.to(device)
        age, gender, emotion = model(image)
        age_pred, gender_pred, emotion_pred = get_prediction(age, gender, emotion)

        # Estrai i label come tensor
        age_labels = label[:, 0]
        gender_labels = label[:, 1]
        emotion_labels = label[:, 2]

        # Filtra i target -1 (sul device)
        mask_age = age_labels != -1
        mask_gender = gender_labels != -1
        mask_emotion = emotion_labels != -1

        all_age_preds.append(age_pred[mask_age])
        all_age_labels.append(age_labels[mask_age])

        all_gender_preds.append(gender_pred[mask_gender])
        all_gender_labels.append(gender_labels[mask_gender])

        all_emotion_preds.append(emotion_pred[mask_emotion])
        all_emotion_labels.append(emotion_labels[mask_emotion])

# Concatenate all batches (still on GPU)
all_age_preds = torch.cat(all_age_preds)
all_age_labels = torch.cat(all_age_labels)
all_gender_preds = torch.cat(all_gender_preds)
all_gender_labels = torch.cat(all_gender_labels)
all_emotion_preds = torch.cat(all_emotion_preds)
all_emotion_labels = torch.cat(all_emotion_labels)

# Move to CPU and convert to numpy
all_age_preds = all_age_preds.cpu().numpy()
all_age_labels = all_age_labels.cpu().numpy()
all_gender_preds = all_gender_preds.cpu().numpy()
all_gender_labels = all_gender_labels.cpu().numpy()
all_emotion_preds = all_emotion_preds.cpu().numpy()
all_emotion_labels = all_emotion_labels.cpu().numpy()

# Calcola e salva le confusion matrix (come array .npy)
age_cm = confusion_matrix(all_age_labels, all_age_preds, labels=range(9))
gender_cm = confusion_matrix(all_gender_labels, all_gender_preds, labels=range(2))
emotion_cm = confusion_matrix(all_emotion_labels, all_emotion_preds, labels=range(7))

np.save("../TEST/confusion_matrix_age.npy", age_cm)
np.save("../TEST/confusion_matrix_gender.npy", gender_cm)
np.save("../TEST/confusion_matrix_emotion.npy", emotion_cm)

# Salva le confusion matrix come immagini
fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay(age_cm, display_labels=CLASSES[0]).plot(ax=ax, cmap="Blues", colorbar=False)
plt.title("Age Confusion Matrix")
plt.savefig("../TEST/confusion_matrix_age.png")
plt.close()

fig, ax = plt.subplots(figsize=(4, 4))
ConfusionMatrixDisplay(gender_cm, display_labels=CLASSES[1]).plot(ax=ax, cmap="Blues", colorbar=False)
plt.title("Gender Confusion Matrix")
plt.savefig("../TEST/confusion_matrix_gender.png")
plt.close()

fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay(emotion_cm, display_labels=CLASSES[2]).plot(ax=ax, cmap="Blues", colorbar=False)
plt.title("Emotion Confusion Matrix")
plt.savefig("../TEST/confusion_matrix_emotion.png")
plt.close()

# Compute accuracy for each task (only on valid samples)
age_acc = np.mean(np.array(all_age_preds) == np.array(all_age_labels)) if len(all_age_labels) > 0 else float('nan')
gender_acc = np.mean(np.array(all_gender_preds) == np.array(all_gender_labels)) if len(all_gender_labels) > 0 else float('nan')
emotion_acc = np.mean(np.array(all_emotion_preds) == np.array(all_emotion_labels)) if len(all_emotion_labels) > 0 else float('nan')

# Compute mean accuracy (only over tasks with at least one valid sample)
accuracies = [acc for acc in [age_acc, gender_acc, emotion_acc] if not np.isnan(acc)]
mean_acc = np.mean(accuracies) if accuracies else float('nan')

# Print and save results
print(f"Age accuracy: {age_acc:.4f}")
print(f"Gender accuracy: {gender_acc:.4f}")
print(f"Emotion accuracy: {emotion_acc:.4f}")
print(f"Mean accuracy: {mean_acc:.4f}")

with open("../TEST/accuracies.txt", "w") as f:
    f.write(f"Age accuracy: {age_acc:.4f}\n")
    f.write(f"Gender accuracy: {gender_acc:.4f}\n")
    f.write(f"Emotion accuracy: {emotion_acc:.4f}\n")
    f.write(f"Mean accuracy: {mean_acc:.4f}\n")
