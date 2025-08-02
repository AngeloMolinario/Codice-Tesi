import os
import json
import torch
import shutil
from torch.utils.data import random_split
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
from utils.configuration import Config



# Load configuration from JSON file
configuration_path = 'config/PECore_VPT_age.json'
config = Config(configuration_path)
# Set a generator
generator = torch.Generator().manual_seed(config.SEED)


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")
print(f"Loaded configuration: {config}")

os.makedirs(config.OUTPUT_DIR, exist_ok=True)
shutil.copy2(configuration_path, f'{config.OUTPUT_DIR}/training_configuration.json')

model = get_model(config).to(DEVICE)
if torch.__version__[0] == '2':
    print("Compiling model with torch.compile...")
    model = torch.compile(model, mode="max-autotune")
    print("Model compiled successfully.")
tokenizer = transforms.get_text_tokenizer(model.text_model.context_length)
img_transform = transforms.get_image_transform(model.image_size)


loss_fn = get_task_loss_fn(config.TASK, num_classes=len(config.CLASSES))

optimizer = None
params = []
total_trainable_params = 0
for name, param in model.named_parameters():
    if any(trainable_param in name for trainable_param in config.NAMED_TRAINABLE_PARAMETERS):
        param.requires_grad = True
        params += [param]
        total_trainable_params += param.numel()
        print(f"Parameter: {name}, shape: {param.shape}, numel: {param.numel()}")        
    else:
        param.requires_grad = False

optimizer = torch.optim.AdamW(params, lr=config.LR)
print(f"Trainable parameter objects: {len(params)}")
print(f"Total trainable parameter values: {total_trainable_params}")



dataset = get_datasets(config.DATASET_NAMES, transforms=img_transform, split="train", dataset_root=config.DATASET_ROOT)
train_size = int(config.TRAIN_SPLIT * len(dataset))
val_size = len(dataset) - train_size

training_dataset, validation_dataset = random_split(dataset, [train_size, val_size], generator=generator)

training_loader = DataLoader(training_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, persistent_workers=True, pin_memory=True if DEVICE=='cuda' else False)
validation_loader = DataLoader(validation_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, persistent_workers=True, pin_memory=True if DEVICE=='cuda' else False)

text_features = None
if config.MODEL_TYPE !=  "SoftCPT":
    print(f"Encoding text features for {config.TASK} classification...")
    texts = tokenizer(config.TEXT_CLASSES_PROMPT).to(DEVICE)
    with torch.no_grad():
        text_features = model.get_text_features(texts, normalize=True)
    print(f"Encoded text features shape: {text_features.shape}")

epoch_train_fn = get_training_step_fn(config.TASK)
epoch_val_fn   = get_validation_step_fn(config.TASK)

training_losses = []
training_accuracies = []
validation_ordinal_losses = []
validation_ordinal_accuracies = []
validation_ce_losses = []
validation_ce_accuracies = []

metrics_tracker = TrainingMetrics(output_dir=f'{config.OUTPUT_DIR}/metrics', class_names=config.CLASSES)

epoch_loss, epoch_accuracy, all_preds, all_labels = epoch_val_fn(model, validation_loader, loss_fn, config.TASK, DEVICE, text_features, use_tqdm=config.USE_TQDM)
print(f"Validation Loss ORDINAL BEFORE TRAINING: {epoch_loss:.4f}, Validation Accuracy: {epoch_accuracy:.4f}")
metrics_tracker.update_predictions(torch.cat(all_preds), torch.cat(all_labels))
metrics_tracker.plot_confusion_matrix(epoch='initial_ORDINAL')
metrics_tracker.reset_predictions()
if config.TASK == 'age':
    epoch_loss, epoch_accuracy, all_preds, all_labels = epoch_val_fn(model, validation_loader, CrossEntropyLoss(), config.TASK, DEVICE, text_features, use_tqdm=config.USE_TQDM)
    print(f"Validation Loss CE BEFORE TRAINING: {epoch_loss:.4f}, Validation Accuracy: {epoch_accuracy:.4f}")

    metrics_tracker.update_predictions(torch.cat(all_preds), torch.cat(all_labels))
    metrics_tracker.plot_confusion_matrix(epoch='initial_CE')
    metrics_tracker.reset_predictions()

patience = config.EARLY_STOPPING_PATIENCE
epochs_no_improve = 0
best_val_loss = float('inf')
early_stop = False

for epoch in range(config.EPOCHS):
    print(f"Epoch {epoch+1}/{config.EPOCHS}")    
    ########### TRAINING STEP ###########
    epoch_loss, epoch_accuracy = epoch_train_fn(model, optimizer, training_loader, loss_fn, config.TASK, DEVICE, text_features, use_tqdm=config.USE_TQDM)
    training_losses.append(epoch_loss.item())
    training_accuracies.append(epoch_accuracy)   
    print(f"Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")


    ############ VALIDATION STEP ###########
    print("Performing validation...")

    if config.TASK == 'age':
        # Also validate with CrossEntropyLoss for age classification
        epoch_loss, epoch_accuracy, all_preds, all_labels = epoch_val_fn(model, validation_loader, CrossEntropyLoss(), config.TASK, DEVICE, text_features, use_tqdm=config.USE_TQDM)
        print(f"Validation Loss CE: {epoch_loss:.4f}, Validation Accuracy: {epoch_accuracy:.4f}")

        # Store validation losses and accuracies
        validation_ce_losses.append(epoch_loss.item())
        validation_ce_accuracies.append(epoch_accuracy)    
        metrics_tracker.update_predictions(torch.cat(all_preds), torch.cat(all_labels))
        metrics_tracker.plot_confusion_matrix(epoch=f"epoch_{epoch+1}_CE")
        metrics_tracker.reset_predictions()
        metrics_tracker.plot_metrics()


    epoch_loss, epoch_accuracy, all_preds, all_labels = epoch_val_fn(model, validation_loader, loss_fn, config.TASK, DEVICE, text_features, use_tqdm=config.USE_TQDM)
    print(f"Validation Loss ORDINAL: {epoch_loss:.4f}, Validation Accuracy: {epoch_accuracy:.4f}")
    # Store validation losses and accuracies

    validation_ordinal_losses.append(epoch_loss.item())
    validation_ordinal_accuracies.append(epoch_accuracy)
    metrics_tracker.update_predictions(torch.cat(all_preds), torch.cat(all_labels))
    metrics_tracker.plot_confusion_matrix(epoch=f"epoch_{epoch+1}_ORDINAL")
    metrics_tracker.reset_predictions()
    metrics_tracker.plot_metrics()


    


    # Early stopping check
    if patience is not None:
        # If is None, we do not apply early stopping logic
        if epoch_loss.item() < best_val_loss:
            best_val_loss = epoch_loss.item()
            epochs_no_improve = 0
            # Save the best model weights
            os.makedirs(f'{config.OUTPUT_DIR}/ckpt', exist_ok=True)
            torch.save(model.state_dict(), f'{config.OUTPUT_DIR}/ckpt/best_model.pth')
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
    torch.save(model.state_dict(), f'{config.OUTPUT_DIR}/ckpt/latest_model.pth')

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
    with open(f'{config.OUTPUT_DIR}/ckpt/training_state.json', 'w') as f:
        json.dump(training_state, f, indent=4)

    # Plotting and saving the training curves
    plot_losses(
        training_losses,
        validation_ordinal_losses,
        validation_ce_losses,
        training_accuracies,
        validation_ordinal_accuracies,
        validation_ce_accuracies,
        config.OUTPUT_DIR
    )

plot_losses(
    training_losses,
    validation_ordinal_losses,
    validation_ce_losses,
    training_accuracies,
    validation_ordinal_accuracies,
    validation_ce_accuracies,
    config.OUTPUT_DIR
)