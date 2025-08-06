import os
import sys
import json
import torch
import shutil
from torch.utils.data import random_split
from torch.utils.data import DataLoader
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

if hasattr(config, 'NUM_SAMPLES_PER_CLASS'):
    # Use TaskBalanceDataset for training with task balancing
    balance_task = getattr(config, 'BALANCE_TASK', None)  # Get balance task from config if available
    training_dataset = TaskBalanceDataset(
        dataset_names=config.DATASET_NAMES, 
        transform=img_transform, 
        split="train", 
        datasets_root=config.DATASET_ROOT, 
        verbose=True,
        balance_task=balance_task
    )
    
    # Use validation dataset names if specified, otherwise use training dataset names
    validation_dataset_names = getattr(config, 'VALIDATION_DATASET_NAMES', config.DATASET_NAMES)
    if not validation_dataset_names:  # If empty list, use training dataset names
        validation_dataset_names = config.DATASET_NAMES
    
    # Use MultiDataset for validation with test split
    validation_dataset = MultiDataset(
        dataset_names=validation_dataset_names, 
        transform=img_transform, 
        split="test", 
        datasets_root=config.DATASET_ROOT, 
        verbose=True
    )
else:
    # Use TaskBalanceDataset for training (without specific balancing)
    training_dataset = TaskBalanceDataset(
        dataset_names=config.DATASET_NAMES, 
        transform=img_transform, 
        split="train", 
        datasets_root=config.DATASET_ROOT, 
        verbose=True
    )
    
    # Use validation dataset names if specified, otherwise use training dataset names
    validation_dataset_names = getattr(config, 'VALIDATION_DATASET_NAMES', config.DATASET_NAMES)
    if not validation_dataset_names:  # If empty list, use training dataset names
        validation_dataset_names = config.DATASET_NAMES
    
    # Use MultiDataset for validation with test split  
    validation_dataset = MultiDataset(
        dataset_names=validation_dataset_names, 
        transforms=img_transform, 
        split="test", 
        datasets_root=config.DATASET_ROOT, 
        verbose=True
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

training_losses = []
training_accuracies = []
validation_ordinal_losses = []
validation_ordinal_accuracies = []
validation_ce_losses = []
validation_ce_accuracies = []

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


for epoch in range(config.EPOCHS):
    print(f"Epoch {epoch+1}/{config.EPOCHS}")
    print(f"Current learning rate: {optimizer.param_groups[0]['lr']:.8f}")
    ########### TRAINING STEP ###########
    epoch_loss, epoch_accuracy = epoch_train_fn(model, optimizer, training_loader, loss_fn, config.TASK, DEVICE, text_features, use_tqdm=config.USE_TQDM)
    training_losses.append(epoch_loss[0])
    training_accuracies.append(epoch_accuracy[0])
    
    # Update training metrics in the tracker
    metrics_tracker.update_train_metrics(epoch_loss, epoch_accuracy)
    
    for i in range(len(epoch_loss)):
        print(f"Training) Task {i} - Loss: {epoch_loss[i]:.4f}, Accuracy: {epoch_accuracy[i]:.4f}")    
    '''
    if (epoch+1) % 5 == 0:
        visualize_similarity_representations(
            model=model, 
            dataloader=validation_loader,
            device=DEVICE,
            task_name="age" if config.TASK=='multitask' else config.TASK,  # Specifica direttamente 'age' se multitask
            output_dir=f'{config.OUTPUT_DIR}/tsne_similarity_plots',
            class_names=config.CLASSES,  # Passa solo le classi age
            text_features=text_features,
            max_samples=10000,
            perplexity=30,
            epoch=epoch+1  # Aggiungi il numero di epoca (1-based)

        )
    '''
    ############ VALIDATION STEP ###########
    print("Performing validation...")

    epoch_loss, epoch_accuracy, all_preds, all_labels = epoch_val_fn(model, validation_loader, loss_fn, config.TASK, DEVICE, text_features, use_tqdm=config.USE_TQDM)
    for i in range(len(epoch_loss)):
        print(f"Validation) Task {i} - Loss: {epoch_loss[i]:.4f}, Accuracy: {epoch_accuracy[i]:.4f}")    
    # Store validation losses and accuracies

    validation_ordinal_losses.append(epoch_loss[0])
    validation_ordinal_accuracies.append(epoch_accuracy[0])
    
    # Update validation metrics in the tracker
    metrics_tracker.update_val_metrics(epoch_loss, epoch_accuracy)
    
    # Handle confusion matrix based on task type
    if config.TASK == 'multitask':
        # For multitask: update predictions for each task separately
        for task_idx in range(len(all_preds)):
            if all_preds[task_idx] and all_labels[task_idx]:  # Check if task has data
                task_preds = torch.cat(all_preds[task_idx])
                task_labels = torch.cat(all_labels[task_idx])
                metrics_tracker.update_predictions(task_preds, task_labels, task_idx=task_idx)
        
        # Plot confusion matrices for all tasks
        metrics_tracker.plot_confusion_matrix(epoch=f"epoch_{epoch+1}")
        metrics_tracker.reset_predictions()
    else:
        # For single task: original behavior
        metrics_tracker.update_predictions(torch.cat(all_preds[0]), torch.cat(all_labels[0]))
        metrics_tracker.plot_confusion_matrix(epoch=f"epoch_{epoch+1}")
        metrics_tracker.reset_predictions()
    
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

    # Save training state in a JSON file
    training_state = {
        'epoch': epoch + 1,
        'best_val_loss': float(best_val_loss),  # Assicura che sia float Python
        'epochs_no_improve': int(epochs_no_improve),  # Assicura che sia int Python
        'training_losses': [float(x) for x in training_losses],  # Converti ogni elemento
        'training_accuracies': [float(x) for x in training_accuracies],  # Converti ogni elemento
        'validation_ordinal_losses': [float(x) for x in validation_ordinal_losses],  # Converti ogni elemento
        'validation_ordinal_accuracies': [float(x) for x in validation_ordinal_accuracies],  # Converti ogni elemento
        'validation_ce_losses': [float(x) for x in validation_ce_losses],  # Converti ogni elemento
        'validation_ce_accuracies': [float(x) for x in validation_ce_accuracies]  # Converti ogni elemento
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