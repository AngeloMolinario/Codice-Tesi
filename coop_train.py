import os
import sys
import json
import torch
import shutil
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torchvision.transforms as T
import matplotlib.pyplot as plt
from torchvision import transforms as T
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR

import os
import numpy as np
import matplotlib.pyplot as plt
import numpy
import seaborn as sns
from dataset.dataset import BaseDataset, MultiDataset, TaskBalanceDataset
from wrappers.PerceptionEncoder.pe import PECore
from wrappers.promptopt.prompt_learner import CustomModel
from training.loss import *
from core.vision_encoder import transforms
from training.training_functions import *
from utils.metric import MultitaskTracker
from utils.configuration import Config

from wrappers.SigLip2.SigLip2Model import Siglip2Model
from transformers import AutoConfig, AutoTokenizer
from wrappers.tokenizer import *
from tqdm import tqdm
from training.training_functions import  *


def get_loss_fn(config, weights=None):    
    task = int(config.TASK)
    if weights is None:
        weights = torch.ones(len(config.CLASSES[task]))
    if task == 0:
        return OrdinalAgeLossEMD(num_classes=len(config.CLASSES[0]),
                                class_frequencies=weights,
                                lambda_ordinal=config.EMD_WEIGHT,
                                omega=config.EMD_OMEGA,
                                mu = config.EMD_MU
                                )
    elif task == 1:
        return CrossEntropyLoss(num_classes=len(config.CLASSES[1]), weights=weights)
    elif task == 2:
        return CrossEntropyLoss(num_classes=len(config.CLASSES[2]), weights=weights)
    else:
        raise ValueError(f"Unknown task: {task}")



def single_task_train_fn(model, dataloader, optimizer, loss_fn, device, config, text_features=None, scaler=None):
    ''' Gestisce unicamente il VPT'''
    model.train()    
    iterator = tqdm(dataloader) if config.USE_TQDM else dataloader        
    
    # Accumulatori per loss e metriche
    total_loss_sum = 0.0
    preds_list = []
    labels_list = []
    
    
    if text_features is not None:
        print("Using precomputed text features")
        computed_text_features = text_features.T.contiguous()        
    else:
        computed_text_features = None

    num_batch = 0
    for image, label in iterator:
        num_batch += 1
        
        image = image.to(device, non_blocking=True)
        labels = label[:, config.TASK].to(device, non_blocking=True)
        
        scale = 1.0
        if hasattr(model, 'logit_scale'):
            scale = model.logit_scale.exp()

        bias = 0.0
        if hasattr(model, 'logit_bias'):
            bias = model.logit_bias
        
        # Use mixed precision on CUDA when available
        with autocast(device_type=device):
            image_features = model.get_image_features(image, normalize=True)
            if text_features is None:                
                computed_text_features = model.get_text_features(normalize=True).T

            logits =  scale * (image_features @ computed_text_features) + bias
            loss = loss_fn(logits, labels)
            pred = logits.argmax(dim=1)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        preds_list.append(pred.detach().cpu())
        labels_list.append(labels.detach().cpu())

        total_loss_sum = total_loss_sum + loss.detach()
        
        if not config.USE_TQDM and (num_batch + 1) % 100 == 0:
            print(f"Processed {num_batch + 1}/{len(iterator)}", end='\r')
    
    # Calcola accuracy
    if preds_list:
        all_preds = torch.cat(preds_list)
        all_labels = torch.cat(labels_list)
        accuracy = (all_preds == all_labels).float().mean().item()
    else:
        accuracy = float('nan')
    
    mean_loss = (total_loss_sum / num_batch).item()
    
    return [mean_loss], [accuracy]  # Ritorna liste per compatibilità con il codice esistente

def single_task_val_fn(model, dataloader, loss_fn, device, config, text_features=None):
    model.eval()
    compute_text_features = text_features is None

    iterator = tqdm(dataloader) if config.USE_TQDM else dataloader


    # Accumulatori
    total_loss_sum = 0.0
    preds_list = []
    labels_list = []
    probs_list = []

    num_batch = 0
    if text_features is not None:
        text_features = text_features.T.contiguous()

    with torch.inference_mode():
        for image, label in iterator:
            num_batch += 1

            image = image.to(device, non_blocking=True)
            labels = label[:, config.TASK].to(device, non_blocking=True)

            if compute_text_features:
                text_features = model.get_text_features(normalize=True).T


            scale = 1.0
            if hasattr(model, 'logit_scale'):
                scale = model.logit_scale.exp()

            bias = 0.0
            if hasattr(model, 'logit_bias'):
                bias = model.logit_bias

            # Use mixed precision during validation forward pass on CUDA
            with autocast(device_type=device):

                image_features = model.get_image_features(image, normalize=True)
                logits =  scale *(image_features @ text_features) + bias

                # Calcola probabilità, loss e predizioni
                probs = torch.softmax(logits, dim=1)
                loss = loss_fn(logits, labels)
                pred = logits.argmax(dim=1)

            total_loss_sum += loss.detach()
            preds_list.append(pred.detach())
            labels_list.append(labels.detach())
            probs_list.append(probs.detach())

            if not config.USE_TQDM and (num_batch + 1) % 100 == 0:
                print(f"Processed {num_batch + 1}/{len(iterator)}", end='\r')

    # Calcola accuracy
    if preds_list:
        all_preds = torch.cat(preds_list, dim=0)
        all_labels = torch.cat(labels_list, dim=0)
        all_probs = torch.cat(probs_list, dim=0)
        accuracy = (all_preds == all_labels).float().mean().item()
    else:
        all_preds = torch.empty(0, dtype=torch.long)
        all_labels = torch.empty(0, dtype=torch.long)
        all_probs = torch.empty(0, dtype=torch.float32)
        accuracy = float('nan')

    mean_loss = (total_loss_sum / num_batch).item()

    return [mean_loss], [accuracy], [all_preds.cpu()], [all_labels.cpu()], [all_probs.cpu()]


def plot_prob_evolution(base_dir, class_names, upto_epoch=None):
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    epoch_dirs = sorted([d for d in os.listdir(base_dir) if d.startswith("epoch_")])
    if upto_epoch is not None:
        epoch_dirs = epoch_dirs[:upto_epoch]

    num_classes = len(class_names)
    history = []
    for d in epoch_dirs:
        avg_probs = np.load(os.path.join(base_dir, d, "avg_probs_matrix.npy"))
        history.append(avg_probs)

    history = np.stack(history, axis=0)  # [epochs, num_classes, num_classes]

    # Plot evoluzione della probabilità corretta
    plt.figure(figsize=(10,6))
    for c in range(num_classes):
        plt.plot(range(1, len(history)+1), history[:, c, c], label=f"True {class_names[c]}")
    plt.xlabel("Epoch")
    plt.ylabel("Mean prob of correct class")
    plt.title("Evolution of correct class probability (Age)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "prob_evolution_correct.png"))
    plt.close()


def main():
    # ------------------ REPRODUCIBILITY ------------------
    seed = 2025 
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # -----------------------------------------------------
    # Load configuration from JSON file
    configuration_path = sys.argv[1] if len(sys.argv) > 1 else "config/PECore_VPT_age.json"
    config = Config(configuration_path)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")
    print(f"Loaded configuration: {config}")
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    shutil.copy2(configuration_path, f'{config.OUTPUT_DIR}/training_configuration.json')

    # Get the model
    model = get_model(config).to(DEVICE)

    # Save the vision model right after loading
    os.makedirs(os.path.join(config.OUTPUT_DIR, "ckpt"), exist_ok=True)
    model.save_vision_model(os.path.join(config.OUTPUT_DIR, "ckpt"), filename="vision_ckpt.pt")


    print(f"MODEL LOGIT SCALE {model.logit_scale}")

    # Get the training and validation dataset
    training_set = get_dataset(config=config,
                               split="train",
                               transform=get_image_transform(config, model.image_size),
                               augmentation_transform=None)
    validation_set = get_dataset(config=config,
                                 split="val",
                                 transform=get_image_transform(config, model.image_size),)

    # Create dataloader for training and validation
 
    val_loader = DataLoader(
        dataset=validation_set,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory_device="cuda",
        pin_memory=True
    )

    # Get the loss function
    weights = training_set.get_class_weights(config.TASK, 'default').to(DEVICE)



    train_loader = DataLoader(
        dataset=training_set,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        pin_memory_device="cuda",
        persistent_workers=True
    )

    loss_fn = get_loss_fn(config, weights=weights)

    # Create optimizer
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
    optimizer = torch.optim.AdamW(params, lr=config.LR, foreach=True, weight_decay=0.0)

    # GradScaler for mixed precision
    scaler = GradScaler(device=DEVICE)
    # CosineAnnealingLR scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=config.EPOCHS, eta_min=1e-6)
    lr_history = [optimizer.param_groups[0]['lr']]
    ''' ---------------------- TO REMOVE ONLY FOR DEBUGGING PURPOSES -------------------------'''

    param_to_name = {p: n for n, p in model.named_parameters()}

    # Itera sui param_groups dell'optimizer
    for group_idx, group in enumerate(optimizer.param_groups):
        print(f"\nParam group {group_idx}: lr={group['lr']}, weight_decay={group.get('weight_decay', 0)}")
        total_params_group = 0
        for p in group['params']:
            name = param_to_name.get(p, "<unknown>")
            numel = p.numel()
            total_params_group += numel
            print(f"  {name}: shape={tuple(p.shape)}, numel={numel}")
        print(f"  → Totale parametri in questo gruppo: {total_params_group}")

    ''' -------------------------------  TO HERE --------------------------------------------'''


    print(f"\nLoaded Model: {type(model)}")
    print(f"\nTraining set {type(training_set)} of size: {len(training_set)}")
    print(f"Validation set {type(validation_set)} of size: {len(validation_set)}")
    print(f"\nTraining batch {type(train_loader)} of size: {len(train_loader)}")
    print(f"Validation batch {type(val_loader)} of size: {len(val_loader)}")

    print(f"\nLoss function:  Task {config.TASK}: {type(loss_fn)}")

    # Metrics tracker
    output_dir = config.OUTPUT_DIR
    task_names = [config.TASK_NAMES[config.TASK]]
    class_names = [config.CLASSES[config.TASK]]
    tracker = MultitaskTracker(
        num_tasks=1,
        output_dir=output_dir,
        task_names=task_names,
        class_names=class_names
    )
    print(f"Tracking metrics for task {task_names[0]}")

    # Early stopping
    patience = config.PATIENCE
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_accuracy = 0.0

    # Training loop
    os.makedirs(os.path.join(config.OUTPUT_DIR, "ckpt"), exist_ok=True)
    train_fn = single_task_train_fn
    val_fn = single_task_val_fn

    if config.TUNING == "coop":
        text_features = None
    else:
        text_features = torch.load(config.TEXT_FEATURES_PATH, map_location="cpu").to(DEVICE)
        torch.save(text_features, os.path.join(config.OUTPUT_DIR, "ckpt/text_features.pt"))

        #text_features = torch.split(text_features, (9,2,7))[config.TASK]
        print(f"text_features task shape {text_features.shape}")

    model.save_vision_model(os.path.join(config.OUTPUT_DIR, "ckpt"), filename="vision_ckpt.pt")
        

    for epoch in range(config.EPOCHS):


        ################## TRAINING LOOP ##################
        print(f"\n[Epoch {epoch+1} - TRAIN]", flush=True)
        train_loss, train_acc = train_fn(model, train_loader, optimizer, loss_fn, DEVICE, config, text_features, scaler)
        print(f"  Task '{task_names[0]}': Loss = {train_loss[0]:.4f}, Accuracy = {train_acc[0]:.4f}")

        print(f" New logit scale {model.logit_scale.item()} - exp = {model.logit_scale.exp().item()} - bias {model.logit_bias.item()if hasattr(model, 'logit_bias') else 0.0}")
        tracker.update_loss(0, train_loss[0], train=True)
        tracker.update_accuracy(0, train_acc[0], train=True)


        ################## VALIDATION LOOP ##################
        print(f"\n[Epoch {epoch+1} - VAL]", flush=True)
        val_loss, val_acc, all_preds_list, all_labels_list, all_probs_list = val_fn(model, val_loader, loss_fn, DEVICE, config, model.get_text_features(normalize=True) if text_features is None else text_features)
        print(f"  Task '{task_names[0]}': Loss = {val_loss[0]:.4f}, Accuracy = {val_acc[0]:.4f}")
        tracker.update_loss(0, val_loss[0], train=False)
        tracker.update_accuracy(0, val_acc[0], train=False)
        tracker.update_confusion(0, all_preds_list[0], all_labels_list[0], epoch)

        # Age analysis and save avg_probs matrix to match multitask outputs
        epoch_analysis_dir = os.path.join(config.OUTPUT_DIR, config.TASK_NAMES[config.TASK].replace(" ", "_"), f"epoch_{epoch+1}")
        analyze_age_errors(
            all_preds_list, all_labels_list, all_probs_list,
            class_names=config.CLASSES[config.TASK],
            task_names=task_names,
            accuracies=val_acc,
            output_dir=epoch_analysis_dir
        )
        labels = all_labels_list[0].numpy()
        probs = all_probs_list[0].numpy()
        num_classes = len(config.CLASSES[config.TASK])
        avg_probs_per_class = []
        for c in range(num_classes):
            mask = labels == c
            if mask.sum() > 0:
                avg_probs_per_class.append(probs[mask].mean(axis=0))
            else:
                avg_probs_per_class.append(np.zeros(num_classes))
        avg_probs_per_class = numpy.array(avg_probs_per_class)  # [num_classes, num_classes]
        numpy.save(os.path.join(epoch_analysis_dir, "avg_probs_matrix.npy"), avg_probs_per_class)
        base_dir = os.path.join(config.OUTPUT_DIR, config.TASK_NAMES[config.TASK].replace(" ","_"))
        plot_prob_evolution(base_dir, config.CLASSES[config.TASK])

        tracker.save_confusion_matrices(epoch)
        tracker.plot_losses()
        tracker.plot_accuracy()
        tracker.save()

        if val_loss[0] < best_val_loss or val_acc[0] > best_accuracy:
            if val_loss[0] < best_val_loss:
                best_val_loss = val_loss[0]
                print(f"New best validation loss: {best_val_loss:.4f}. Saving artifacts...")
                if text_features is None:
                    with torch.inference_mode():
                        text_features_to_save = model.get_text_features(normalize=True)
                        torch.save(text_features_to_save, os.path.join(config.OUTPUT_DIR, "ckpt/text_features_bval.pt"))
                if config.NUM_VISUAL_PROMPT > 0:
                    model.save_vpt_token(os.path.join(config.OUTPUT_DIR, "ckpt/vpt_token_bval.pt"))

            if val_acc[0] > best_accuracy:
                best_accuracy = val_acc[0]
                print(f"New best validation accuracy: {best_accuracy:.4f}. Saving artifacts...")
                if text_features is None:
                    with torch.inference_mode():
                        text_features_to_save = model.get_text_features(normalize=True)
                        torch.save(text_features_to_save, os.path.join(config.OUTPUT_DIR, "ckpt/text_features_bacc.pt"))
                if config.NUM_VISUAL_PROMPT > 0:
                    model.save_vpt_token(os.path.join(config.OUTPUT_DIR, "ckpt/vpt_token_bacc.pt"))

            epochs_without_improvement = 0            
        else:
            epochs_without_improvement += 1
            print(f"No improvement in validation loss. Epochs without improvement: {epochs_without_improvement}/{patience}")
        if epochs_without_improvement >= patience:
            print("Early stopping triggered.")
            break

        
        scheduler.step()
        lr_history.append(optimizer.param_groups[0]['lr'])

        # Plot learning rate schedule
        plt.figure()
        plt.plot(range(len(lr_history)), lr_history, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True)
        plt.savefig(os.path.join(config.OUTPUT_DIR, "learning_rate_plot.png"))
        plt.close()

    # Save the full model at the end of training
    torch.save(model.state_dict(), os.path.join(config.OUTPUT_DIR, "full_training_model.pt"))


if __name__ == "__main__":
    main()
