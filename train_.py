import os
import sys
import json
import torch
import shutil
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torchvision.transforms as T

import matplotlib.pyplot as plt

from dataset.dataset import BaseDataset, MultiDataset, TaskBalanceDataset
from wrappers.PerceptionEncoder.pe import PECore
from wrappers.promptopt.prompt_learner import CustomModel
from training import training_functions
from training.loss import *
from core.vision_encoder import transforms
from training.training_functions import *
from utils.metric import TrainingMetrics
from utils.configuration import Config



# Load configuration from JSON file
configuration_path = sys.argv[1] if len(sys.argv) > 1 else "config/PECore_VPT_age.json"
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
    model = torch.compile(model)  # , mode="max-autotune")
    print("Model compiled successfully.")
tokenizer = transforms.get_text_tokenizer(model.text_model.context_length)
img_transform = transforms.get_image_transform(model.image_size)



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

if config.MODEL_TYPE == "softCPT":
    for param in model.get_softCPT_parameters():
        param.requires_grad = True

optimizer = torch.optim.Adam(params, lr=config.LR) if config.MODEL_TYPE != "softCPT" else torch.optim.Adam(model.get_softCPT_parameters(), lr=config.LR)
# Aggiungi scheduler per diminuire LR di 1/6 ad ogni epoca fino a un minimo di 1e-6
scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda epoch: max(8/10, 1e-6/optimizer.param_groups[0]['lr']))
print(f"Trainable parameter objects: {len(params)}")
print(f"Total trainable parameter values: {total_trainable_params}")

training_dataset = validation_dataset = None


# AUGMENTATION TRANSFORMATION FOR DUPLICATED SAMPLES IN THE DATASET
augment_transform = T.Compose([
    T.Resize((model.image_size, model.image_size)),    
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.2),
    T.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.2),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True)
])

training_set = training_dataset = TaskBalanceDataset(
        dataset_names=config.DATASET_NAMES, 
        transform=img_transform, 
        split="train", 
        datasets_root=config.DATASET_ROOT, 
        balance_task=config.BALANCE_TASK,
        verbose=True,
        all_datasets=len(config.DATASET_NAMES) == 0,
        augment_duplicate=augment_transform,  # Use the augmentation transform for duplicated samples
    )

validation_set = validation_dataset = MultiDataset(
        dataset_names=config.DATASET_NAMES, 
        transform=img_transform, 
        split="val", 
        datasets_root=config.DATASET_ROOT, 
        verbose=True,
        all_datasets=len(config.DATASET_NAMES) == 0
    )


print("training_dataset length:", len(training_dataset))
print("validation_dataset length:", len(validation_dataset))

training_loader = DataLoader(training_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, persistent_workers=True, pin_memory=True)
validation_loader = DataLoader(validation_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, persistent_workers=True, pin_memory=True)

text_features = None
if config.MODEL_TYPE !=  "softCPT":
    print(f"Encoding text features for {config.TASK} classification...")
    texts = tokenizer(config.TEXT_CLASSES_PROMPT).to(DEVICE)
    with torch.no_grad():
        text_features = model.get_text_features(texts, normalize=True)
    print(f"Encoded text features shape: {text_features.shape}")

epoch_train_fn = get_training_step_fn(config.TASK)
epoch_val_fn   = get_validation_step_fn(config.TASK)

weights = None
if config.TASK == "multitask":
    weights = []
    weights.append(training_dataset.get_class_weights("age").to(DEVICE))
    weights.append(training_dataset.get_class_weights("gender").to(DEVICE))
    weights.append(training_dataset.get_class_weights("emotion").to(DEVICE))
else:
    weights = training_dataset.get_class_weights(config.TASK).to(DEVICE)

loss_fn = get_task_loss_fn(config, weights=weights)

# Initialize metrics tracker based on task type
if config.TASK == 'multitask':
    task_names = ['age', 'gender', 'emotion']
    metrics_tracker = TrainingMetrics(
        output_dir=f'{config.OUTPUT_DIR}/metrics', 
        class_names=config.CLASSES,  # Should be a list of lists for multitask
        is_multitask=True,
        task_names=task_names
    )
else:
    metrics_tracker = TrainingMetrics(
        output_dir=f'{config.OUTPUT_DIR}/metrics', 
        class_names=config.CLASSES,
        is_multitask=False
    )

patience = config.EARLY_STOPPING_PATIENCE
epochs_no_improve = 0
best_val_loss = float('inf')
early_stop = False


print(f"Loss type: {type(loss_fn)}")

# INITIAL VALIDATION ROUND BEFORE TRAINING
print("Performing initial validation before training...")
initial_epoch_loss, initial_epoch_accuracy, initial_all_preds, initial_all_labels = epoch_val_fn(
    model, validation_loader, loss_fn, config.TASK, DEVICE, text_features, use_tqdm=config.USE_TQDM
)

# Print initial validation metrics
if config.TASK == 'multitask':
    task_labels = ['all', 'age', 'gender', 'emotion']
    for i in range(len(initial_epoch_loss)):
        print(f"Initial Validation) Task {i} ({task_labels[i]}) - Loss: {initial_epoch_loss[i]:.4f}, Accuracy: {initial_epoch_accuracy[i]:.4f}")
else:
    print(f"Initial Validation) Loss: {initial_epoch_loss[0]:.4f}, Accuracy: {initial_epoch_accuracy[0]:.4f}")

# Handle initial confusion matrix based on task type
if config.TASK == 'multitask':
    # For multitask: update predictions for each task separately
    for task_idx in range(len(initial_all_preds)):
        if initial_all_preds[task_idx] and initial_all_labels[task_idx]:  # Check if task has data
            task_preds = torch.cat(initial_all_preds[task_idx])
            task_labels_tensor = torch.cat(initial_all_labels[task_idx])
            metrics_tracker.update_predictions(task_preds, task_labels_tensor, task_idx=task_idx)
    
    # Plot confusion matrices for all tasks with epoch="-1"
    metrics_tracker.plot_confusion_matrix(epoch="-1")
    metrics_tracker.reset_predictions()
else:
    # For single task: handle based on data structure
    if initial_all_preds and initial_all_labels:
        if isinstance(initial_all_preds[0], list):
            # If it's a list of lists, concatenate the first list
            task_preds = torch.cat(initial_all_preds[0]) if initial_all_preds[0] else torch.tensor([])
            task_labels_tensor = torch.cat(initial_all_labels[0]) if initial_all_labels[0] else torch.tensor([])
        else:
            # If it's already tensors, concatenate directly
            task_preds = torch.cat(initial_all_preds)
            task_labels_tensor = torch.cat(initial_all_labels)
        
        if len(task_preds) > 0 and len(task_labels_tensor) > 0:
            metrics_tracker.update_predictions(task_preds, task_labels_tensor)
            
            # Plot confusion matrix with epoch="-1"
            metrics_tracker.plot_confusion_matrix(epoch="-1")
            metrics_tracker.reset_predictions()

print("Initial validation completed. Starting training...")

for epoch in range(config.EPOCHS):
    print(f"Epoch {epoch+1}/{config.EPOCHS}")
    print(f"Current learning rate: {optimizer.param_groups[0]['lr']:.8f}")
    ########### TRAINING STEP ###########
    epoch_loss, epoch_accuracy = epoch_train_fn(model, optimizer, training_loader, loss_fn, config.TASK, DEVICE, text_features, use_tqdm=config.USE_TQDM)
    
    # Update training metrics in the tracker
    metrics_tracker.update_train_metrics(epoch_loss, epoch_accuracy)
    
    # Print training metrics with proper task labels
    if config.TASK == 'multitask':
        task_labels = ['all', 'age', 'gender', 'emotion']
        for i in range(len(epoch_loss)):
            print(f"Training) Task {i} ({task_labels[i]}) - Loss: {epoch_loss[i]:.4f}, Accuracy: {epoch_accuracy[i]:.4f}")
    else:
        print(f"Training) Loss: {epoch_loss[0]:.4f}, Accuracy: {epoch_accuracy[0]:.4f}")
        
    ############ VALIDATION STEP ###########
    print("Performing validation...")

    epoch_loss, epoch_accuracy, all_preds, all_labels = epoch_val_fn(model, validation_loader, loss_fn, config.TASK, DEVICE, text_features, use_tqdm=config.USE_TQDM)
    
    # Update validation metrics in the tracker
    metrics_tracker.update_val_metrics(epoch_loss, epoch_accuracy)
    
    # Print validation metrics with proper task labels
    if config.TASK == 'multitask':
        task_labels = ['all', 'age', 'gender', 'emotion']
        for i in range(len(epoch_loss)):
            print(f"Validation) Task {i} ({task_labels[i]}) - Loss: {epoch_loss[i]:.4f}, Accuracy: {epoch_accuracy[i]:.4f}")
    else:
        print(f"Validation) Loss: {epoch_loss[0]:.4f}, Accuracy: {epoch_accuracy[0]:.4f}")
    
    # Handle confusion matrix based on task type
    if config.TASK == 'multitask':
        # For multitask: update predictions for each task separately
        for task_idx in range(len(all_preds)):
            if all_preds[task_idx] and all_labels[task_idx]:  # Check if task has data
                task_preds = torch.cat(all_preds[task_idx])
                task_labels_tensor = torch.cat(all_labels[task_idx])
                metrics_tracker.update_predictions(task_preds, task_labels_tensor, task_idx=task_idx)
        
        # Plot confusion matrices for all tasks every epoch
        metrics_tracker.plot_confusion_matrix(epoch=f"epoch_{epoch+1}")
        metrics_tracker.reset_predictions()
    else:
        # For single task: handle based on data structure
        if all_preds and all_labels:
            if isinstance(all_preds[0], list):
                # If it's a list of lists, concatenate the first list
                task_preds = torch.cat(all_preds[0]) if all_preds[0] else torch.tensor([])
                task_labels_tensor = torch.cat(all_labels[0]) if all_labels[0] else torch.tensor([])
            else:
                # If it's already tensors, concatenate directly
                task_preds = torch.cat(all_preds)
                task_labels_tensor = torch.cat(all_labels)
            
            if len(task_preds) > 0 and len(task_labels_tensor) > 0:
                metrics_tracker.update_predictions(task_preds, task_labels_tensor)
                
                # Plot confusion matrix every epoch
                metrics_tracker.plot_confusion_matrix(epoch=f"epoch_{epoch+1}")
                metrics_tracker.reset_predictions()

    # Plot training/validation curves every epoch
    metrics_tracker.plot_metrics()
    
    # Early stopping check
    if patience is not None:
        # If is None, we do not apply early stopping logic
        if epoch_loss[0] < best_val_loss:
            best_val_loss = epoch_loss[0]
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

    # Aggiorna il learning rate alla fine dell'epoca (solo se sopra il minimo)
    if (epoch+1)%3 == 0:
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr > 1e-6:
            new_lr = max(current_lr / 6, 1e-6)
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
            print(f"Learning rate updated to: {new_lr:.8f}")
        else:
            print(f"Learning rate at minimum: {current_lr:.8f}")
    
    # Save model state dict
    torch.save(model.state_dict(), f'{config.OUTPUT_DIR}/ckpt/latest_model.pth')

    # Save training state in a JSON file with enhanced metrics tracking
    training_state = {
        'epoch': epoch + 1,
        'best_val_loss': float(best_val_loss),
        'epochs_no_improve': int(epochs_no_improve),
    }
    
    # Save additional metrics based on task type
    if config.TASK == 'multitask':
        # For multitask, save all task metrics
        training_state.update({
            'training_losses_all': [float(x) for x in metrics_tracker.train_losses[0]],
            'training_accuracies_all': [float(x) for x in metrics_tracker.train_accuracies[0]],
            'validation_losses_all': [float(x) for x in metrics_tracker.val_losses[0]],
            'validation_accuracies_all': [float(x) for x in metrics_tracker.val_accuracies[0]],
            'training_losses_age': [float(x) for x in metrics_tracker.train_losses[1]],
            'training_accuracies_age': [float(x) for x in metrics_tracker.train_accuracies[1]],
            'validation_losses_age': [float(x) for x in metrics_tracker.val_losses[1]],
            'validation_accuracies_age': [float(x) for x in metrics_tracker.val_accuracies[1]],
            'training_losses_gender': [float(x) for x in metrics_tracker.train_losses[2]],
            'training_accuracies_gender': [float(x) for x in metrics_tracker.train_accuracies[2]],
            'validation_losses_gender': [float(x) for x in metrics_tracker.val_losses[2]],
            'validation_accuracies_gender': [float(x) for x in metrics_tracker.val_accuracies[2]],
            'training_losses_emotion': [float(x) for x in metrics_tracker.train_losses[3]],
            'training_accuracies_emotion': [float(x) for x in metrics_tracker.train_accuracies[3]],
            'validation_losses_emotion': [float(x) for x in metrics_tracker.val_losses[3]],
            'validation_accuracies_emotion': [float(x) for x in metrics_tracker.val_accuracies[3]],
        })
    else:
        # For single task, use the original approach
        training_state.update({
            'training_losses': [float(x) for x in metrics_tracker.train_losses],
            'training_accuracies': [float(x) for x in metrics_tracker.train_accuracies],
            'validation_losses': [float(x) for x in metrics_tracker.val_losses],
            'validation_accuracies': [float(x) for x in metrics_tracker.val_accuracies],
        })
    
    with open(f'{config.OUTPUT_DIR}/ckpt/training_state.json', 'w') as f:
        json.dump(training_state, f, indent=4)

# Final plots and confusion matrices
print("Training completed. Generating final plots...")
metrics_tracker.plot_metrics()
if config.TASK == 'multitask':
    metrics_tracker.plot_confusion_matrix(epoch='final')
else:
    metrics_tracker.plot_confusion_matrix(epoch='final')

print(f"All training plots and metrics saved in: {config.OUTPUT_DIR}/metrics")