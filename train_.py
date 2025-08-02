import os
import json
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataset.dataset import BaseDataset, MultiDataset
from wrappers.PerceptionEncoder.pe import PECore
from wrappers.promptopt.prompt_learner import CustomModel
from training import training_functions
from training.loss import *
from core.vision_encoder import transforms
from training.training_functions import *
from utils.metric import TrainingMetrics
# This file contains the implementation for age classification testing using a PECore model.


# TODO: Add a json or yaml configuration file to store these parameters
USE_TQDM = False
BATCH_SIZE = 80
NUM_WORKERS = 6
EPOCHS = 25
LR = 0.00001
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

NUM_VISUAL_PROMPTS = 10
DATASET_NAMES = ["FairFace"]
DATASET_ROOT = "../datasets_with_standard_labels"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")
# TODO: add the training validation splitting logic
TRAIN_SPLIT = 0.8
VAL_SPLIT = 1-TRAIN_SPLIT

model = get_model(MODEL_TYPE, num_prompt=NUM_VISUAL_PROMPTS).to(DEVICE)
if torch.__version__[0] == '2':
    print("Compiling model with torch.compile...")
    model = torch.compile(model, mode="max-autotune")
    print("Model compiled successfully.")
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

data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, persistent_workers=True, pin_memory=True if DEVICE=='cuda' else False)
test_loader = DataLoader(testing_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, persistent_workers=True, pin_memory=True if DEVICE=='cuda' else False)

text_features = None
if MODEL_TYPE !=  "SoftCPT":
    print("Encoding text features for age classification...")
    texts = tokenizer(TEXT_CLASSES_PROMPT).to(DEVICE)
    with torch.no_grad():
        text_features = model.get_text_features(texts, normalize=True)
    print(f"Encoded text features shape: {text_features.shape}")

epoch_train_fn = get_training_step_fn(TASK)
epoch_val_fn   = get_validation_step_fn(TASK)

training_losses = []
training_accuracies = []
validation_ordinal_losses = []
validation_ordinal_accuracies = []
validation_ce_losses = []
validation_ce_accuracies = []

metrics_tracker = TrainingMetrics(output_dir='output/metrics', class_names=CLASSES)

epoch_loss, epoch_accuracy, all_preds, all_labels = epoch_val_fn(model, test_loader, loss_fn, TASK, DEVICE, text_features, use_tqdm=USE_TQDM)
print(f"Validation Loss ORDINAL BEFORE TRAINING: {epoch_loss:.4f}, Validation Accuracy: {epoch_accuracy:.4f}")
metrics_tracker.update_predictions(torch.cat(all_preds), torch.cat(all_labels))
metrics_tracker.plot_confusion_matrix(epoch='initial_ORDINAL')
metrics_tracker.reset_predictions()

epoch_loss, epoch_accuracy, all_preds, all_labels = epoch_val_fn(model, test_loader, CrossEntropyLoss(), TASK, DEVICE, text_features, use_tqdm=USE_TQDM)
print(f"Validation Loss CE BEFORE TRAINING: {epoch_loss:.4f}, Validation Accuracy: {epoch_accuracy:.4f}")

metrics_tracker.update_predictions(torch.cat(all_preds), torch.cat(all_labels))
metrics_tracker.plot_confusion_matrix(epoch='initial_CE')
metrics_tracker.reset_predictions()

patience = 5
epochs_no_improve = 0
best_val_loss = float('inf')
early_stop = False

for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}")    
    epoch_loss, epoch_accuracy = epoch_train_fn(model, optimizer, data_loader, loss_fn, TASK, DEVICE, text_features, use_tqdm=USE_TQDM)
    training_losses.append(epoch_loss.item())
    training_accuracies.append(epoch_accuracy)   
    print(f"Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")


    ######
    print(f"Training loss type: {type(epoch_loss)} - Accuracy type: {type(epoch_accuracy)}")
    ######

    print("Performing validation...")
    epoch_loss, epoch_accuracy, all_preds, all_labels = epoch_val_fn(model, test_loader, loss_fn, TASK, DEVICE, text_features, use_tqdm=USE_TQDM)
    validation_ordinal_losses.append(epoch_loss.item())
    validation_ordinal_accuracies.append(epoch_accuracy)
    print(f"Validation Loss ORDINAL: {epoch_loss:.4f}, Validation Accuracy: {epoch_accuracy:.4f}")

    metrics_tracker.update_predictions(torch.cat(all_preds), torch.cat(all_labels))
    metrics_tracker.plot_confusion_matrix(epoch=f"epoch_{epoch+1}_ORDINAL")
    metrics_tracker.reset_predictions()
    metrics_tracker.plot_metrics()

    epoch_loss, epoch_accuracy, all_preds, all_labels = epoch_val_fn(model, test_loader, CrossEntropyLoss(), TASK, DEVICE, text_features, use_tqdm=USE_TQDM)
    validation_ce_losses.append(epoch_loss.item())
    validation_ce_accuracies.append(epoch_accuracy)
    print(f"Validation Loss CE: {epoch_loss:.4f}, Validation Accuracy: {epoch_accuracy:.4f}")
    metrics_tracker.update_predictions(torch.cat(all_preds), torch.cat(all_labels))
    metrics_tracker.plot_confusion_matrix(epoch=f"epoch_{epoch+1}_CE")
    metrics_tracker.reset_predictions()
    metrics_tracker.plot_metrics()


    # Early stopping check
    if patience is not None:
        # If is None, we do not apply early stopping logic
        if epoch_loss.item() < best_val_loss:
            best_val_loss = epoch_loss.item()
            epochs_no_improve = 0
            # Save the best model weights
            os.makedirs('output/ckpt', exist_ok=True)
            torch.save(model.state_dict(), 'output/ckpt/best_model.pth')
            print("Validation loss improved, saving model.")
        else:
            epochs_no_improve += 1
            print(f"Validation loss did not improve for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement.")
            early_stop = True

        if early_stop:
            break

    # Save model state dict
    torch.save(model.state_dict(), 'output/ckpt/latest_model.pth')

    # Save training state in a JSON file
    training_state = {
        'epoch': epoch + 1,
        'best_val_loss': best_val_loss,
        'epochs_no_improve': epochs_no_improve,
        'training_losses': training_losses,
        'training_accuracies': training_accuracies,
        'validation_ordinal_losses': validation_ordinal_losses,
        'validation_ordinal_accuracies': validation_ordinal_accuracies,
        'validation_ce_losses': validation_ce_losses,
        'validation_ce_accuracies': validation_ce_accuracies
    }
    with open('output/ckpt/training_state.json', 'w') as f:
        json.dump(training_state, f, indent=4)

# Plotting and saving the training curves
print("Plotting and saving training curves...")
os.makedirs('output/plot', exist_ok=True)

# Plot Losses
plt.figure(figsize=(10, 6))
plt.plot(training_losses, label='Training Loss')
plt.plot(validation_ordinal_losses, label='Validation Ordinal Loss')
plt.plot(validation_ce_losses, label='Validation CE Loss')
plt.title('Losses vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('output/plot/losses_curve.png')
plt.close()

# Plot Accuracies
plt.figure(figsize=(10, 6))
plt.plot(training_accuracies, label='Training Accuracy')
plt.plot(validation_ordinal_accuracies, label='Validation Ordinal Accuracy')
plt.plot(validation_ce_accuracies, label='Validation CE Accuracy')
plt.title('Accuracy vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('output/plot/accuracies_curve.png')
plt.close()

print("Training curves saved in 'output/plot'.")

