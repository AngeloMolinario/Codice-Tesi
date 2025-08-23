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
from wrappers.SigLip2.SigLip2Model import Siglip2Model
from transformers import AutoConfig
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
            model = Siglip2Model(
                config = AutoConfig.from_pretrained("google/siglip2-base-patch16-224", cache_dir="./hf_models"),
                num_prompts=config.NUM_VISUAL_PROMPT
            )
            model.load_model(path="./hf_models/model.pth", map_location="cpu")
            return model
        
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
    # Solo multitask
    age_loss = CrossEntropyMAELoss(
        bin_centers=torch.tensor([1.5, 6.5, 14.5, 24.5, 34.5, 44.5, 54.5, 64.5, 80.0]).to('cuda'),
        class_weights=weights[0].to('cuda'),
        alpha=0.3,
        scale=15.0
    ).to('cuda')
    gender_loss = CrossEntropyLoss(weights=weights[1])
    emotion_loss = CrossEntropyLoss(weights=weights[2])
    loss = [
        MaskedLoss(age_loss, -1),
        MaskedLoss(gender_loss, -1),
        MaskedLoss(emotion_loss, -1)
    ]
    return loss

def get_image_transform(config):
    model_name = config.MODEL.lower()
    if model_name == 'pecore':
        return transforms.get_image_transform(224)
    elif model_name == 'siglip2':
        return T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
        ])

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

            #logits = model.logit_scale.exp() * (image_features @ text_features)
            logits = (image_features @ text_features)
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
                print(f"Processed {num_batch + 1}/{len(iterator)}", end='\r', flush=True)

    # Compute the average losses and accuracy
    accuracies = [(correct[i] / count[i]).item() if count[i] > 0 else float('nan')
              for i in range(num_task)]

    mean_loss = [(losses_sums[i]/num_batch).item() for i in range(num_task+1)]

    return mean_loss, accuracies

def analyze_age_errors(all_preds_list, all_labels_list, all_probs_list, class_names, task_names, accuracies, output_dir):
    """
    Genera i grafici di analisi errori per l'età (task 0) e salva anche un file .npy
    con tutti i dati necessari per ricreare i grafici.

    File salvato: {output_dir}/analysis_data.npy
    Contenuti principali salvati:
      - class_names, task_names, accuracies
      - per-class:
          n_samples, mean_probs, argmax_counts, argmax_norm_counts
      - offset_matrix (+ ticks)
      - prob_matrix
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    os.makedirs(output_dir, exist_ok=True)

    preds = all_preds_list[0].numpy()
    labels = all_labels_list[0].numpy()
    probs = all_probs_list[0].numpy()
    num_classes = len(class_names)

    # Helper per annotare i valori sopra le barre e bloccare y in [0, 1]
    def _annotate_bars_and_fix_ylim(ax, bars):
        ax.set_ylim(0, 1)
        for rect in bars:
            h = rect.get_height()
            # Posiziona l'etichetta poco sopra la barra ma entro il limite 1.0
            y = min(h + 0.02, 0.98)
            ax.text(rect.get_x() + rect.get_width() / 2.0, y, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=9)

    # --- 1 - Distribuzione di probabilità per ogni classe (salvata separatamente)
    prob_dir = os.path.join(output_dir, "Prob_distribution_per_class")
    os.makedirs(prob_dir, exist_ok=True)

    # Dati da salvare
    per_class_data = []
    n_samples_per_class = np.zeros(num_classes, dtype=int)

    for c in range(num_classes):
        mask = labels == c
        n_c = int(mask.sum())
        n_samples_per_class[c] = n_c

        mean_probs = None
        norm_counts = None

        if n_c > 0:
            mean_probs = probs[mask].mean(axis=0)  # [num_classes]
            denom = mean_probs.sum()
            norm_counts = (mean_probs / denom) if denom > 0 else np.zeros_like(mean_probs)

            plt.figure(figsize=(8, 5))
            ax = plt.gca()
            bars = ax.bar(class_names, norm_counts)
            plt.xticks(rotation=45)
            plt.xlabel("Classi")
            plt.ylabel("Frazione di campioni (normalizzata)")
            plt.title(f"Distribuzione di probabilità normalizzata - {class_names[c]}")
            _annotate_bars_and_fix_ylim(ax, bars)
            plt.tight_layout()
            plt.savefig(os.path.join(prob_dir, f"class_{c}_{class_names[c]}.png"))
            plt.close()

        per_class_data.append({
            "class_index": c,
            "class_name": class_names[c],
            "n_samples": n_c,
            "mean_probs": None if mean_probs is None else mean_probs.astype(np.float32),
            "norm_counts": None if norm_counts is None else norm_counts.astype(np.float32),
        })

    # --- 1b - Distribuzione dell'argmax per ogni classe reale
    argmax_dir = os.path.join(output_dir, "Argmax_distribution_per_class")
    os.makedirs(argmax_dir, exist_ok=True)

    for c in range(num_classes):
        mask = labels == c
        if mask.sum() > 0:
            pred_counts = np.bincount(preds[mask], minlength=num_classes)
            norm_pred_counts = pred_counts / pred_counts.sum() if pred_counts.sum() > 0 else np.zeros(num_classes, dtype=float)

            plt.figure(figsize=(8, 5))
            ax = plt.gca()
            bars = ax.bar(class_names, norm_pred_counts)
            plt.xticks(rotation=45)
            plt.xlabel("Classi Predette (Argmax)")
            plt.ylabel("Frazione di campioni")
            plt.title(f"Distribuzione argmax predetto - {class_names[c]}")
            _annotate_bars_and_fix_ylim(ax, bars)
            plt.tight_layout()
            plt.savefig(os.path.join(argmax_dir, f"class_{c}_{class_names[c]}.png"))
            plt.close()

            # arricchisci il blocco per-class con conteggi argmax
            per_class_data[c]["argmax_counts"] = pred_counts.astype(np.int32)
            per_class_data[c]["argmax_norm_counts"] = norm_pred_counts.astype(np.float32)
        else:
            per_class_data[c]["argmax_counts"] = None
            per_class_data[c]["argmax_norm_counts"] = None

    # --- 2 - Accuracy per singolo task
    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    bars = ax.bar(task_names, accuracies)
    # Annotazioni sulle barre con y in [0,1]
    ax.set_ylim(0, 1)
    for i, rect in enumerate(bars):
        v = rect.get_height()
        y = min(v + 0.02, 0.98)
        ax.text(rect.get_x() + rect.get_width() / 2.0, y, f"{v:.2f}", ha='center', va='bottom')
    plt.ylabel("Accuracy")
    plt.title("Accuracy per task")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "2_accuracy_per_task.png"))
    plt.close()

    # --- 3 - Distribuzione errori con valori numerici
    offset_ticks = np.arange(-(num_classes - 1), num_classes)
    offset_matrix = np.zeros((num_classes, 2 * num_classes - 1), dtype=np.float32)
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() > 0:
            offsets = preds[mask] - labels[mask]
            for j, off in enumerate(offset_ticks):
                offset_matrix[c, j] = np.mean(offsets == off)

    plt.figure(figsize=(12, 6))
    sns.heatmap(offset_matrix, annot=True, fmt=".2f",
                xticklabels=offset_ticks,
                yticklabels=class_names, cmap="Blues")
    plt.xlabel("Offset (Pred - Reale)")
    plt.ylabel("Classe Reale")
    plt.title("Distribuzione normalizzata degli errori (Task 0 - Age)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "3_error_distribution_with_prob.png"))
    plt.close()

    # --- 4 - Probabilità medie di scelta
    prob_matrix = np.zeros((num_classes, num_classes), dtype=np.float32)
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

    # ===== Salvataggio dati necessari per ricreare i grafici =====
    data_to_save = {
        "class_names": list(class_names),
        "task_names": list(task_names),
        "accuracies": np.array(accuracies, dtype=np.float32),
        "num_classes": int(num_classes),
        "n_samples_per_class": n_samples_per_class,
        "per_class": per_class_data,                 # lista di dict (vedi sopra)
        "offset_matrix": offset_matrix,              # [C, 2C-1]
        "offset_ticks": offset_ticks,                # [-C+1, ..., C-1]
        "prob_matrix": prob_matrix,                  # [C, C]
        # opzionali utili per ricostruzioni complete:
        "preds": preds.astype(np.int32),
        "labels": labels.astype(np.int32),
        # evitiamo di salvare 'probs' complete se sono enormi; decommenta se vuoi raw probs:
        "probs": probs.astype(np.float32),
    }
    np.save(os.path.join(output_dir, "analysis_data.npy"), data_to_save, allow_pickle=True)

    print(f"[INFO] Grafici salvati in: {output_dir}")
    print(f"[INFO] Dati per i grafici salvati in: {os.path.join(output_dir, 'analysis_data.npy')}")


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
            with autocast():
                image_features = model.get_image_features(image, normalize=True)
                #logits = model.logit_scale.exp() * (image_features @ text_features.T)
                logits = (image_features @ text_features.T)
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
                print(f"Processed {num_batch + 1}/{len(iterator)}", end='\r', flush=True)

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
        persistent_workers=True,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=config.PREFETCH_FACTOR
    )

    ################ Get the loss function #####################################################

    weights = [training_set.get_class_weights_effective(i).to(DEVICE) for i in range(3)]
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
    for i, loss in enumerate(loss_fn):
        if type(loss) is MaskedLoss:
            print(f"  Task {i}: {type(loss.base_loss)} with ignore index {loss.ignore_index}")



    ########################## METRICS TRACKER #####################################################
    tracker = None

    num_tasks = len(config.TASK_NAMES) if config.TASK == -1 else 1
    output_dir = config.OUTPUT_DIR
    task_names = config.TASK_NAMES
    class_names = config.CLASSES
    tracker = MultitaskTracker(
        num_tasks=num_tasks,
        output_dir=output_dir,
        task_names=task_names,
        class_names=class_names
    )
    print(f"Tracking metrics for multitask: {task_names}")

    # EARLY STOPPING BASED ON LOSS
    patience = 14
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_accuracy = 0.0

    # TRAINING LOOP
    os.makedirs(os.path.join(config.OUTPUT_DIR, "ckpt"), exist_ok=True)
    train_fn = multitask_train_fn
    val_fn   = multitask_val_fn

    running_mean = RunningMeans(['age', 'gender', 'emotion'], alpha=0.95)

    text_features = None
    if config.TUNING.lower() != 'softcpt':
        tokenizer = transforms.get_text_tokenizer(model.text_model.context_length)
        all_text_features = []
        for task_prompts in config.TEXT_CLASSES_PROMPT:
            text = tokenizer(task_prompts).to(DEVICE)
            task_text_features = model.get_text_features(text=text, normalize=True)
            all_text_features.append(task_text_features)
        model.save(
            save_path=os.path.join(config.OUTPUT_DIR, f"ckpt/initial_model.pt"),
            text_features=torch.cat(all_text_features, dim=0),
            text_features_path=os.path.join(config.OUTPUT_DIR, f"ckpt/vpt_text_features.pt"),
        )
        text_features = torch.cat(all_text_features, dim=0)
        print(f"Text prompts by task: {config.TEXT_CLASSES_PROMPT}")
        print(f"Text features shape: {text_features.shape}")
        print(f"Text features shapes per task: {[tf.shape for tf in all_text_features]}")

    task_weight = training_set.get_task_weights()
    print(f"Task weights: {task_weight}")

    print(f"\n[Epoch 0 - VAL]", flush=True)
    val_loss, val_acc, all_preds_list, all_labels_list, _ = val_fn(model, val_loader, loss_fn, DEVICE, task_weight, config, text_features)
    for t in range(num_tasks):
        print(f"  Task '{task_names[t]}': Loss = {val_loss[t]:.4f}, Accuracy = {val_acc[t]:.4f}")
        tracker.update_confusion(t, all_preds_list[t], all_labels_list[t], 100)
    print(f"  Total raw Loss: {sum(val_loss[:-1]):.4f}, Total weighted loss {val_loss[-1]:.4f}, Mean Accuracy : {sum(val_acc)/num_tasks:.4f}")
    tracker.update_accuracy(None, sum(val_acc)/num_tasks, train=False, mean=True)
    tracker.save_confusion_matrices(100)

    tracker = MultitaskTracker(
        num_tasks=num_tasks,
        output_dir=output_dir,
        task_names=task_names,
        class_names=class_names
    )

    weights_history = []

    print(f"LOGIT SCALE: {model.logit_scale.item()}")

    for epoch in range(config.EPOCHS):
        
        print(f"[Epoch {epoch+1}]", flush=True)
        w = []
        for i in range(num_tasks):
            w_i = running_mean.get_by_index(i)
            if w_i is None:
                w_i=1.0
            w.append(1.0 / max(w_i, 1e-8))
        task_weight = torch.tensor(w, device=DEVICE)
        task_weight = task_weight / task_weight.sum()
        print(f"Task weights (EMA inverse) for epoch {epoch+1}: {task_weight.tolist()}")
        weights_history.append(task_weight.cpu().numpy())

        train_loss, train_acc = train_fn(model, train_loader, optimizer, running_mean, loss_fn, DEVICE, task_weight, config, text_features, scaler)
        running_mean.plot(os.path.join(config.OUTPUT_DIR, "running_mean_train.png"))
        for t in range(num_tasks):
            print(f"  Task '{task_names[t]}': Loss = {train_loss[t]:.4f}, Accuracy = {train_acc[t]:.4f}")
            tracker.update_loss(t, train_loss[t], train=True)
            tracker.update_accuracy(t, train_acc[t], train=True)
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
        epoch_analysis_dir = os.path.join(config.OUTPUT_DIR, "age_analysis", f"epoch_{epoch+1}")
        analyze_age_errors(
            all_preds_list, all_labels_list, all_probs_list,
            class_names=config.CLASSES[0],
            task_names=task_names,
            accuracies=val_acc,
            output_dir=epoch_analysis_dir
        )
        labels = all_labels_list[0].numpy()
        probs  = all_probs_list[0].numpy()
        num_classes = len(config.CLASSES[0])
        avg_probs_per_class = []
        for c in range(num_classes):
            mask = labels == c
            if mask.sum() > 0:
                avg_probs_per_class.append(probs[mask].mean(axis=0))
            else:
                avg_probs_per_class.append(np.zeros(num_classes))
        avg_probs_per_class = numpy.array(avg_probs_per_class)  # [num_classes, num_classes]
        numpy.save(os.path.join(epoch_analysis_dir, "avg_probs_matrix.npy"), avg_probs_per_class)
        base_dir = os.path.join(config.OUTPUT_DIR, "age_analysis")
        plot_prob_evolution(base_dir, config.CLASSES[0])
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
        else:
            epochs_without_improvement += 1
            print(f"No improvement in validation loss. Epochs without improvement: {epochs_without_improvement}/{patience}")
        if epochs_without_improvement >= patience:
            print("Early stopping triggered.")
            break
        torch.save(model.state_dict(), os.path.join(config.OUTPUT_DIR, f"ckpt/last_model.pt"))
        scheduler.step()
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

        _weights_history = np.array(weights_history)
        plt.figure(figsize=(8,4))
        for i, name in enumerate(task_names):
            plt.plot(_weights_history[:,i], label=name)
        plt.xlabel('Epoch')
        plt.ylabel('Task Weight (EMA inverse)')
        plt.title('Task Weights per Epoch')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(config.OUTPUT_DIR, 'task_weights_per_epoch.png'))
        plt.close()


if __name__ == "__main__":
    main()