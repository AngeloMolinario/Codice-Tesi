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
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR

import os
import numpy as np
import matplotlib.pyplot as plt
import numpy
import seaborn as sns
from dataset.dataset import BaseDataset, MultiDataset, TaskBalanceDataset
from wrappers.PerceptionEncoder.pe import PECore
from wrappers.promptopt.prompt_learner import CustomModel
from training import training_functions
from training.loss import *
from core.vision_encoder import transforms
from training.training_functions import *
from utils.metric import MultitaskTracker
from utils.configuration import Config

def get_model(config):
    ''' This method look at the configuration file and return the correct model initialized with pretrained weights and the specified attributes'''
    tuning = config.TUNING.lower()
    model_name = config.MODEL.lower()

    if tuning == "softcpt":
        if model_name == "pecore":
            base_model = PECore.from_config("PE-Core-B16-224", pretrained=True, num_prompt=config.NUM_VISUAL_PROMPT)
            model = CustomModel(
                n_ctx=config.NUM_TEXT_CNTX,
                tasknames=config.TASK_NAMES,
                classnames=config.CLASSES,
                model=base_model,
                tokenizer=transforms.get_text_tokenizer(base_model.text_model.context_length)
            )

            return model
        elif model_name == "siglip2":
            raise NotImplementedError(f"Model {model_name} is not implemented for VPT tuning.")
        else:
            raise ValueError(f"Unknown model name: {model_name}")

    elif tuning == "vpt":
        if model_name == "pecore":
            model = PECore.from_config("PE-Core-B16-224", pretrained=True, num_prompt=config.NUM_VISUAL_PROMPT)
            return model
        elif model_name == "siglip2":
            raise NotImplementedError(f"Model {model_name} is not implemented for VPT tuning.")
        else:
            raise ValueError(f"Unknown model name: {model_name}")
    
    elif tuning == "vvpt":
        raise NotImplementedError(f"Model {model_name} is not implemented for VVPT tuning.")
    else:
        raise ValueError(f"Unknown tuning method: {tuning}")

def get_dataset(config, split, transform=None, augmentation_transform=None):

    balance_task = None
    if hasattr(config, "BALANCE_TASK"):
        balance_task = config.BALANCE_TASK
    task = config.TASK
    datasets = config.DATASET_NAMES[task] # Take the dataset names for the specific task
    
    if len(datasets) == 1:
        return BaseDataset(
            root=os.path.join(config.DATASET_ROOT, datasets[0]),
            transform=transform,
            split=split,
            verbose=config.VERBOSE
        )
    else:
        if int(task) == -1:
            # This is the only case in which balancing the dataset may be needed
            if balance_task is not None and split=="train":
                return TaskBalanceDataset(
                    dataset_names=datasets,
                    transform=transform,
                    split=split,
                    datasets_root=config.DATASET_ROOT,
                    all_datasets=len(datasets) == 0, # if the dataset name list is empty load all the dataset in the root folder
                    balance_task=balance_task,
                    augment_duplicate=augmentation_transform,
                    verbose=config.VERBOSE
                )
        # The multitask doesn't need to be balanced, or i'm in a specific task scenario, or i want the validation set
        return MultiDataset(
            dataset_names=datasets,
            transform=transform,
            split=split,
            datasets_root=config.DATASET_ROOT,
            all_datasets=len(datasets) == 0, # if the dataset name list is empty load all the dataset in the root folder
            verbose=config.VERBOSE
        )    

def get_loss_fn(config, weights=None):    
    task = int(config.TASK)
    if weights is None:    
        if task == -1:    
            weights = [
                torch.ones(len(config.CLASSES[0])),
                torch.ones(len(config.CLASSES[1])),
                torch.ones(len(config.CLASSES[2]))
            ]
        else:
            weights = torch.ones(len(config.CLASSES[task]))
    
    if task == -1:
        age_loss = OrdinalPeakedCELoss(num_classes=len(config.CLASSES[0]), weights=weights[0])
        gender_loss = CrossEntropyLoss(weights=weights[1])
        emotion_loss = CrossEntropyLoss(weights=weights[2])

        loss = [
            MaskedLoss(age_loss, -1),
            MaskedLoss(gender_loss, -1),
            MaskedLoss(emotion_loss, -1)
        ]
        return loss
    
    elif task == 0:
        return OrdinalLoss(num_classes=len(config.CLASSES[0]), weights=weights)        
    elif task == 1:
        return CrossEntropyLoss(weights=weights)        
    elif task == 2:
        return CrossEntropyLoss(weights=weights)        
    else:
        raise ValueError(f"Unknown task: {task}")

def get_image_transform(config):
    model_name = config.MODEL.lower()
    if model_name == 'pecore':
        return transforms.get_image_transform(224)
    elif model_name == 'siglip2':
        raise NotImplementedError(f"Image transform for model {model_name} is not implemented.")
    
    raise ValueError(f"Unknown model name: {model_name}")

def get_augmentation_transform(config):
    model_name = config.MODEL.lower()
    tform = [
        T.Resize((224,224)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.05),
        T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8))], p=0.5),
        T.ToTensor()
    ]

    if model_name == 'pecore':
        tform.append(T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True))
    elif model_name == 'siglip2':
        raise NotImplementedError(f"Augmentation transform for model {model_name} is not implemented.")
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    return T.Compose(tform)

def multitask_train_fn(model, dataloader, optimizer, running_mean, loss_fn, device, task_weight, config, text_features=None, scaler=None):
    model.train()
    compute_text_features = text_features is None or config.NUM_TEXT_CNTX > 0
    # If I am training SoftCPT than text_embedding will be None, otherwise they doesn't change during the training

    iterator = tqdm(dataloader) if config.USE_TQDM else dataloader

    logit_split = [len(c) for c in config.CLASSES]
    num_task = len(loss_fn)

    if not torch.is_tensor(task_weight):
        task_weight = torch.as_tensor(task_weight, dtype=torch.float32, device=device)
    else:
        task_weight = task_weight.to(device)

    # accumultaros for loss and metrics tracking
    losses_sums = torch.zeros(num_task+1, device=device)  # +1 for the total loss

    correct = [torch.zeros(1, device=device) for _ in range(num_task)]
    count   = [torch.zeros(1, device=device) for _ in range(num_task)]
    if text_features is not None:
        text_features = text_features.T.contiguous()

    num_batch = 0
    for image, label in iterator:
        num_batch += 1

        image = image.to(device, non_blocking=True)
        labels = label.to(device, non_blocking=True)

        if compute_text_features:
            text_features = model.get_text_features(normalize=True).T.contiguous()        

        with autocast():
            with torch.set_grad_enabled(config.NUM_VISUAL_PROMPT!=0):
                image_features = model.get_image_features(image, normalize=True)

            logits = model.logit_scale.exp() * (image_features @ text_features)

            logits_by_task = torch.split(logits, logit_split, dim=1)  # tuple di view

            total_loss = 0.0

            for i in range(len(loss_fn)):
                loss_i, pred_i = loss_fn[i](logits_by_task[i], labels[:, i], return_predicted_label=True)
                losses_sums[i] += loss_i.detach()

                valid_mask = labels[:, i] != -1
                if valid_mask.any():
                    correct[i] += (pred_i[valid_mask] == labels[valid_mask, i]).sum()
                    count[i]   += valid_mask.sum()

                if running_mean is not None:
                    running_mean.update_by_idx(loss_i.item(), i)

                total_loss = total_loss + task_weight[i] * loss_i

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            optimizer.step()

        losses_sums[-1] += total_loss.detach()
        if not config.USE_TQDM and (num_batch + 1)%100 == 0:
                print(f"Processed {num_batch + 1}/{len(iterator)}", end='\r')

    # Compute the average losses and accuracy
    accuracies = [(correct[i] / count[i]).item() if count[i] > 0 else float('nan')
              for i in range(num_task)]

    mean_loss = [(losses_sums[i]/num_batch).item() for i in range(num_task+1)]

    return mean_loss, accuracies


def multitask_val_fn_(model, dataloader, loss_fn, device, task_weight, config, text_features=None):
    model.eval()
    compute_text_features = text_features is None or config.NUM_TEXT_CNTX > 0

    iterator = tqdm(dataloader) if config.USE_TQDM else dataloader

    logit_split = [len(c) for c in config.CLASSES]
    num_task = len(loss_fn)

    if not torch.is_tensor(task_weight):
        task_weight = torch.as_tensor(task_weight, dtype=torch.float32, device=device)
    else:
        task_weight = task_weight.to(device)

    losses_sums = torch.zeros(num_task+1, device=device)

    preds_per_task = [[] for _ in range(num_task)]
    labels_per_task = [[] for _ in range(num_task)]

    num_batch = 0
    with torch.no_grad():
        for image, label in iterator:
            num_batch += 1

            image = image.to(device, non_blocking=True)
            labels = label.to(device, non_blocking=True)

            if compute_text_features:
                text_features = model.get_text_features(normalize=True)
            
            with autocast():
                image_features = model.get_image_features(image, normalize=True)
                logits = model.logit_scale.exp() * (image_features @ text_features.T)
                logits_by_task = torch.split(logits, logit_split, dim=1)

                total_loss = 0.0

                for i in range(len(loss_fn)):
                    loss_i, pred_i = loss_fn[i](logits_by_task[i], labels[:, i], return_predicted_label=True)
                    losses_sums[i] += loss_i.detach()

                    valid_mask = labels[:, i] != -1
                    if valid_mask.any():
                        preds_per_task[i].append(pred_i[valid_mask].detach().cpu())  # SU GPU
                        labels_per_task[i].append(labels[valid_mask, i].detach().cpu())  # SU GPU

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

def analyze_age_errors(all_preds_list, all_labels_list, all_probs_list, class_names, task_names, accuracies, output_dir):


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

            # --- abilita autocast per mixed precision ---
            from torch.cuda.amp import autocast
            with autocast():
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

def single_task_train_fn(model, dataloader, optimizer, running_mean, loss_fn, device, task_weight, config, text_features=None, scaler=None):
    ''' Gestisce unicamente il VPT'''
    model.train()    
    iterator = tqdm(dataloader) if config.USE_TQDM else dataloader        
    
    # Accumulatori per loss e metriche
    total_loss_sum = 0.0
    preds_list = []
    labels_list = []
    
    
    if text_features is not None:
        text_features = text_features.T.contiguous()
    
    num_batch = 0
    for image, label in iterator:
        num_batch += 1
        
        image = image.to(device, non_blocking=True)
        labels = label[:, config.TASK].to(device, non_blocking=True)
                                        
        scale = model.logit_scale.exp()

        with autocast():
            image_features = model.get_image_features(image, normalize=True)
            logits = scale * (image_features @ text_features)
            loss, pred = loss_fn(logits, labels, return_predicted_label=True)
        
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

def single_task_val_fn(model, dataloader, loss_fn, device, task_weight, config, text_features=None):
    model.eval()
    compute_text_features = text_features is None or config.NUM_TEXT_CNTX > 0
    
    iterator = tqdm(dataloader) if config.USE_TQDM else dataloader
    
    # Per single task, abbiamo solo una loss function    
    
    # Accumulatori
    total_loss_sum = 0.0
    preds_list = []
    labels_list = []
    
    num_batch = 0
    with torch.inference_mode():
        for image, label in iterator:
            num_batch += 1
            
            image = image.to(device, non_blocking=True)
            labels = label[:,config.TASK].to(device, non_blocking=True)
            
            if compute_text_features:
                text_features = model.get_text_features(normalize=True)
            
            image_features = model.get_image_features(image, normalize=True)
            logits = model.logit_scale.exp() * (image_features @ text_features.T)
            
            # Calcola loss e predizioni
            loss, pred = loss_fn(logits, labels, return_predicted_label=True)
            total_loss_sum += loss.detach()
            preds_list.append(pred.detach().cpu())
            labels_list.append(labels.detach().cpu())

            if not config.USE_TQDM and (num_batch + 1) % 100 == 0:
                print(f"Processed {num_batch + 1}/{len(iterator)}", end='\r')
    
    # Calcola accuracy
    if preds_list:
        all_preds = torch.cat(preds_list, dim=0)
        all_labels = torch.cat(labels_list, dim=0)
        accuracy = (all_preds == all_labels).float().mean().item()
    else:
        all_preds = torch.empty(0, dtype=torch.long)
        all_labels = torch.empty(0, dtype=torch.long)
        accuracy = float('nan')
    
    mean_loss = (total_loss_sum / num_batch).item()
    
    return [mean_loss], [accuracy], [all_preds], [all_labels]


def get_step_fn(config, mode="train"):
    if config.TASK == -1:
        return multitask_train_fn if mode=='train' else multitask_val_fn
    else:
        return single_task_train_fn if mode=='train' else single_task_val_fn
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
    # Load configuration from JSON file
    configuration_path = sys.argv[1] if len(sys.argv) > 1 else "config/PECore_VPT_age.json"
    config = Config(configuration_path)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")
    print(f"Loaded configuration: {config}")
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    shutil.copy2(configuration_path, f'{config.OUTPUT_DIR}/training_configuration.json')

    ################## Get the model ############################################################

    model = get_model(config).to(DEVICE)    


    ################## Get the training and validation dataset ##################################
    training_set =  get_dataset(config=config,
                               split="train",
                               transform=get_image_transform(config),
                               augmentation_transform=get_augmentation_transform(config)
                               )
    validation_set = get_dataset(config=config,
                                 split="val",
                                 transform=get_image_transform(config)
                                )
    

    ################# Create dataloader for training and validation ############################

    train_loader = DataLoader(
        dataset=training_set,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        pin_memory_device="cuda",
        persistent_workers=True,
        drop_last=True,
        prefetch_factor=config.PREFETCH_FACTOR
    )


    val_loader = DataLoader(
        dataset=validation_set,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory_device="cuda",
        pin_memory=True,
        drop_last=True,
        prefetch_factor=config.PREFETCH_FACTOR
    )

    ################ Get the loss function #####################################################

    weights = None
    if int(config.TASK) == -1:
        weights = [training_set.get_class_weights(i).to(DEVICE) for i in range(3)]
    else:
        weights = training_set.get_class_weights(int(config.TASK)).to(DEVICE)

    loss_fn = get_loss_fn(config, weights=weights)

    ####################### CREATE OPTIMIZER ###################################################

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
        
    optimizer = torch.optim.AdamW(params, lr=config.LR, foreach=True, weight_decay=0.0)

    # Add GradScaler for mixed precision
    scaler = GradScaler()

    # Add CosineAnnealingLR scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=config.EPOCHS, eta_min=1e-6)

    # Lista per tracciare il learning rate
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
    print(f"\n\nTraining set {type(training_set)} of size: {len(training_set)}")
    print(f"Validation set {type(validation_set)} of size: {len(validation_set)}")
    print(f"\nTraining batch {type(train_loader)} of size: {len(train_loader)}")
    print(f"Validation batch {type(val_loader)} of size: {len(val_loader)}")

    print(f"\n\nLoss function:")
    if config.TASK == -1:
        for i, loss in enumerate(loss_fn):
            if type(loss) is MaskedLoss:
                print(f"  Task {i}: {type(loss.base_loss)} with ignore index {loss.ignore_index}")
            else:
                print(f"  Task {i}: {type(loss)}")
    else:
        print(f"  Task {config.TASK}: {type(loss_fn)}")

    ########################## METRICS TRACKER #####################################################
    tracker = None

    num_tasks = len(config.TASK_NAMES) if config.TASK == -1 else 1
    output_dir = config.OUTPUT_DIR
    task_names = config.TASK_NAMES if config.TASK == -1 else [config.TASK_NAMES[config.TASK]]
    class_names = config.CLASSES

    if config.TASK == -1:        
        tracker = MultitaskTracker(
            num_tasks=num_tasks,
            output_dir=output_dir,
            task_names=task_names,
            class_names=class_names
        )
        print(f"Tracking metrics for multitask: {task_names}")
    else:
        tracker = MultitaskTracker(
            num_tasks=1,
            output_dir=output_dir,
            task_names=task_names,
            class_names=[class_names[config.TASK]]
        )
        print(f"Tracking metrics for task {task_names[0]}")

    ######################## EARLY STOPPING BASED ON LOSS ###################################

    patience = 8
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    best_accuracy = 0.0

    ######################## START TRAINING LOOP ############################################
    os.makedirs(os.path.join(config.OUTPUT_DIR, "ckpt"), exist_ok=True)
    train_fn = get_step_fn(config, mode="train")
    val_fn   = get_step_fn(config, mode="val")

    running_mean = None
    if config.TASK == -1:
        running_mean = RunningMeans(['age', 'gender', 'emotion'], alpha=0.95)
    
    text_features = None
    if config.TUNING.lower() != 'softcpt':
        tokenizer = transforms.get_text_tokenizer(model.text_model.context_length)
        
        # Process text prompts by task to maintain task structure
        all_text_features = []
        for task_prompts in config.TEXT_CLASSES_PROMPT:
            text = tokenizer(task_prompts).to(DEVICE)
            task_text_features = model.get_text_features(text=text, normalize=True)
            all_text_features.append(task_text_features)
        
        if config.TUNING.lower() != "softcpt":
            model.save(
                save_path=os.path.join(config.OUTPUT_DIR, f"ckpt/initial_model.pt"),
                text_features=torch.cat(all_text_features, dim=0),
                text_features_path=os.path.join(config.OUTPUT_DIR, f"ckpt/vpt_text_features.pt"),
            )

        # Concatenate all task text features
        if config.TASK == -1:
            text_features = torch.cat(all_text_features, dim=0)
        else:
            text_features = all_text_features[config.TASK]
        
        print(f"Text prompts by task: {config.TEXT_CLASSES_PROMPT}")
        print(f"Text features shape: {text_features.shape}")
        print(f"Text features shapes per task: {[tf.shape for tf in all_text_features]}")


    #task_weight = torch.tensor(config.TASK_WEIGHTS).to(DEVICE)
    task_weight = training_set.get_task_weights()
    print(f"Task weights: {task_weight}")



    print(f"\n[Epoch 0 - VAL]", flush=True)
    val_loss, val_acc, all_preds_list, all_labels_list, _ = val_fn(model, val_loader, loss_fn, DEVICE, task_weight, config, text_features)
    
    
    for t in range(num_tasks):
        print(f"  Task '{task_names[t]}': Loss = {val_loss[t]:.4f}, Accuracy = {val_acc[t]:.4f}")
        tracker.update_confusion(t, all_preds_list[t], all_labels_list[t], 100)
    if num_tasks > 1:
        print(f"  Total raw Loss: {sum(val_loss[:-1]):.4f}, Total weighted loss {val_loss[-1]:.4f}, Mean Accuracy : {sum(val_acc)/num_tasks:.4f}")
        tracker.update_accuracy(None, sum(val_acc)/num_tasks, train=False, mean=True)

    tracker.save_confusion_matrices(100)
    if config.TASK == -1:        
        tracker = MultitaskTracker(
            num_tasks=num_tasks,
            output_dir=output_dir,
            task_names=task_names,
            class_names=class_names
        )
        print(f"Tracking metrics for multitask: {task_names}")
    else:
        tracker = MultitaskTracker(
            num_tasks=1,
            output_dir=output_dir,
            task_names=task_names,
            class_names=[class_names[config.TASK]]
        )
        print(f"Tracking metrics for task {task_names[0]}")

    weights_history = []
    for epoch in range(config.EPOCHS):

        print(f"\n[Epoch {epoch+1} - TRAIN]", flush=True)

        if running_mean is not None:
            w = []
            for i in range(num_tasks):
                w_i = running_mean.get_by_index(i)
                if w_i is None:
                    w_i=1.0
                w.append(1.0 / max(w_i, 1e-8))
            task_weight = torch.tensor(w, device=DEVICE)
            task_weight = task_weight / task_weight.sum()
            print(f"Task weights (EMA inverse) for epoch {epoch+1}: {task_weight.tolist()}")
        else:
            task_weight = torch.ones(num_tasks, device=DEVICE)

        #task_weight = torch.tensor([1.0, 0.2, 0.8], device=DEVICE)
        weights_history.append(task_weight.cpu().numpy())

        train_loss, train_acc = train_fn(model, train_loader, optimizer, running_mean, loss_fn, DEVICE, task_weight, config, text_features, scaler)
        
        if running_mean is not None:
            running_mean.plot(os.path.join(config.OUTPUT_DIR, "running_mean_train.png"))
        
        for t in range(num_tasks):
            print(f"  Task '{task_names[t]}': Loss = {train_loss[t]:.4f}, Accuracy = {train_acc[t]:.4f}")
            tracker.update_loss(t, train_loss[t], train=True)
            tracker.update_accuracy(t, train_acc[t], train=True)
        if num_tasks > 1:
            print(f"  Total Loss: {sum(train_loss[:-1]):.4f}, Total weighted loss {train_loss[-1]:.4f}, Mean Accuracy : {sum(train_acc)/num_tasks:.4f}")        
            tracker.update_loss(None, sum(train_loss[:-1]), train=True, multitask=True)
        tracker.update_accuracy(None, sum(train_acc)/num_tasks, train=True, mean=True)        

        # Validation
        print(f"\n[Epoch {epoch+1} - VAL]", flush=True)
        val_loss, val_acc, all_preds_list, all_labels_list, all_probs_list = val_fn(model, val_loader, loss_fn, DEVICE, task_weight, config, text_features)


        for t in range(num_tasks):
            print(f"  Task '{task_names[t]}': Loss = {val_loss[t]:.4f}, Accuracy = {val_acc[t]:.4f}")
            tracker.update_loss(t, val_loss[t], train=False)
            tracker.update_accuracy(t, val_acc[t], train=False)
            tracker.update_confusion(t, all_preds_list[t], all_labels_list[t], epoch)

        if config.TASK == -1 or config.TASK == 0:
            epoch_analysis_dir = os.path.join(config.OUTPUT_DIR, "age_analysis", f"epoch_{epoch+1}")
            analyze_age_errors(
                all_preds_list, all_labels_list, all_probs_list,
                class_names=config.CLASSES[0] if config.TASK == -1 else config.CLASSES[config.TASK],
                task_names=task_names,
                accuracies=val_acc,
                output_dir=epoch_analysis_dir
            )
             # ---- nuovo: traccia prob medie per classe ----
            labels = all_labels_list[0].numpy()
            probs  = all_probs_list[0].numpy()
            num_classes = len(config.CLASSES[0] if config.TASK == -1 else config.CLASSES[config.TASK])

            avg_probs_per_class = []
            for c in range(num_classes):
                mask = labels == c
                if mask.sum() > 0:
                    avg_probs_per_class.append(probs[mask].mean(axis=0))
                else:
                    avg_probs_per_class.append(np.zeros(num_classes))

            avg_probs_per_class = numpy.array(avg_probs_per_class)  # [num_classes, num_classes]

            numpy.save(os.path.join(epoch_analysis_dir, "avg_probs_matrix.npy"), avg_probs_per_class)
        if config.TASK == -1 or config.TASK == 0:
            base_dir = os.path.join(config.OUTPUT_DIR, "age_analysis")
            plot_prob_evolution(base_dir, config.CLASSES[0] if config.TASK == -1 else config.CLASSES[config.TASK])


        if num_tasks > 1:
            print(f"  Total raw Loss: {sum(val_loss[:-1]):.4f}, Total weighted loss {val_loss[-1]:.4f}, Mean Accuracy : {sum(val_acc)/num_tasks:.4f}")
            tracker.update_loss(None, sum(val_loss[:-1]), train=False, multitask=True)
            tracker.update_accuracy(None, sum(val_acc)/num_tasks, train=False, mean=True)

        tracker.save_confusion_matrices(epoch)
        tracker.plot_losses()
        tracker.save()        

        if val_loss[-1] < best_val_loss or sum(val_acc)/num_tasks > best_accuracy:
            if val_loss[-1] < best_val_loss:
                best_val_loss = val_loss[-1]
                print(f"New best validation loss: {best_val_loss:.4f}. Saving model...")
                torch.save(model.state_dict(), os.path.join(config.OUTPUT_DIR, f"ckpt/best_model.pt"))
            if sum(val_acc)/num_tasks > best_accuracy:
                best_accuracy = sum(val_acc)/num_tasks
                print(f"New best validation accuracy: {best_accuracy:.4f}. Saving model...")
                torch.save(model.state_dict(), os.path.join(config.OUTPUT_DIR, f"ckpt/best_accuracy_model.pt"))
           
            
            epochs_without_improvement = 0
            
            if config.TUNING.lower() == "softcpt":
                model.save_text_features(
                text_features_path=os.path.join(config.OUTPUT_DIR, f"ckpt/best_soft_cpt_text_features.pt"),
                normalize=True
            )            
        else:
            epochs_without_improvement += 1
            print(f"No improvement in validation loss. Epochs without improvement: {epochs_without_improvement}/{patience}")

        if epochs_without_improvement >= patience:
            print("Early stopping triggered.")
            break

        if config.TUNING.lower() == "softcpt":
            model.save_text_features(
            text_features_path=os.path.join(config.OUTPUT_DIR, f"ckpt/latest_soft_cpt_text_features.pt"),
            normalize=True
        )
        torch.save(model.state_dict(), os.path.join(config.OUTPUT_DIR, f"ckpt/last_model.pt"))

        # Step the scheduler at the end of each epoch
        scheduler.step()
        # Salva il learning rate corrente
        lr_history.append(optimizer.param_groups[0]['lr'])

    # Plot della variazione del learning rate
    plt.figure()
    plt.plot(range(len(lr_history)), lr_history, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True)
    plt.savefig(os.path.join(config.OUTPUT_DIR, "learning_rate_plot.png"))
    plt.close()

    weights_history = np.array(weights_history)
    plt.figure(figsize=(8,4))
    for i, name in enumerate(task_names):
        plt.plot(weights_history[:,i], label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Task Weight (EMA inverse)')
    plt.title('Task Weights per Epoch')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_DIR, 'task_weights_per_epoch.png'))
    plt.close()

main()