import os
from torch.utils.data import DataLoader

from dataset.dataset import BaseDataset, MultiDataset
from wrappers.PerceptionEncoder.pe import PECore
from wrappers.promptopt.prompt_learner import CustomModel
from training import training_functions
from training.loss import *
from core.vision_encoder import transforms
from training.training_functions import *
# This file contains the implementation for age classification testing using a PECore model.

BATCH_SIZE = 58
NUM_WORKERS = 8
EPOCHS = 3
LR = 0.0001
TASK = 'age'
MODEL_TYPE = 'VPT'
CLASSES = ['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70+']
TEXT_CLASSES_PROMPT = [
    "A photo of a person between 0 and 2 years old",
    "A photo of a person between 3 and 9 years old",
    "A photo of a person between 10 and 19 years old",
    "A photo of a person between 20 and 29 years old",
    "A photo of a person between 30 and 39 years old",
    "A photo of a person between 40 and 49 years old",
    "A photo of a person between 50 and 59 years old",
    "A photo of a person between 60 and 69 years old",
    "A photo of a person with more than 70 years old"
    ]

NUM_VISUAL_PROMPTS = 5
DATASET_NAMES = ['UTKFace']
DATASET_ROOT = "./datasets_with_standard_labels"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TRAIN_SPLIT = 0.8
VAL_SPLIT = 1-TRAIN_SPLIT

model = get_model(MODEL_TYPE, num_prompt=NUM_VISUAL_PROMPTS).to(DEVICE)
tokenizer = transforms.get_text_tokenizer(model.text_model.context_length)
img_transform = transforms.get_image_transform(model.image_size)


loss_fn = get_task_loss_fn(TASK, num_classes=len(CLASSES))

optimizer = None
if MODEL_TYPE == 'VPT':
    params = []
    total_trainable_params = 0
    for name, param in model.named_parameters():
        
        if "prompt_learner" in name:
            param.requires_grad = True
            params += [param]
            total_trainable_params += param.numel()
            print(f"Parameter: {name}, shape: {param.shape}, numel: {param.numel()}")
        elif "logit_scale" in name:
            param.requires_grad = True
            params += [param]
            total_trainable_params += param.numel()
            print(f"Parameter: {name}, shape: {param.shape}, numel: {param.numel()}")
        else:
            param.requires_grad = False
#        print(f"Parameter: {name}, requires_grad: {param.requires_grad}")

    optimizer = torch.optim.AdamW(params, lr=LR)
    print(f"Trainable parameter objects: {len(params)}")
    print(f"Total trainable parameter values: {total_trainable_params}")



dataset = get_datasets(DATASET_NAMES, transforms=img_transform, split="train", dataset_root=DATASET_ROOT)
testing_dataset = get_datasets(['UTKFace'], transforms=img_transform, split="test", dataset_root=DATASET_ROOT)

data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True if DEVICE=='cuda' else False)
test_loader = DataLoader(testing_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True if DEVICE=='cuda' else False)

text_features = None
if MODEL_TYPE !=  "SoftCPT":
    print("Encoding text features for age classification...")
    texts = tokenizer(TEXT_CLASSES_PROMPT).to(DEVICE)
    text_features = model.get_text_features(texts, normalize=True)
    print(f"Encoded text features shape: {text_features.shape}")

epoch_train_fn = get_training_step_fn(TASK)
epoch_val_fn   = get_validation_step_fn(TASK)

training_losses = []
training_accuracies = []
validation_losses = []
validation_accuracies = []

for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}")    
    epoch_loss, epoch_accuracy = epoch_train_fn(model, optimizer, data_loader, loss_fn, TASK, DEVICE, text_features, use_tqdm=True)
    training_losses.append(epoch_loss.item())
    training_accuracies.append(epoch_accuracy)   
    print(f"Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")    
    print("Performing validation...")
    epoch_loss, epoch_accuracy = epoch_val_fn(model, test_loader, loss_fn, TASK, DEVICE, text_features, use_tqdm=True)
    validation_losses.append(epoch_loss.item())
    validation_accuracies.append(epoch_accuracy)
    print(f"Validation Loss ORDINAL: {epoch_loss:.4f}, Validation Accuracy: {epoch_accuracy:.4f}")
    epoch_loss, epoch_accuracy = epoch_val_fn(model, test_loader, CrossEntropyLoss(), TASK, DEVICE, text_features, use_tqdm=True)
    print(f"Validation Loss CE: {epoch_loss:.4f}, Validation Accuracy: {epoch_accuracy:.4f}")
