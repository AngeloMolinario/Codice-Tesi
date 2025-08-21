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
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

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
        weights = torch.ones(len(config.CLASSES[task]))
    if task == 0:
        return HybridOrdinalLossV2(num_classes=len(config.CLASSES[0]),
                                    alpha=0.3,              # CORAL-adapted (leggera regolarizzazione ordinale)
                                    beta=0.7,               # CE soft-target "peaked" (componente principale)
                                    gamma=0.02,              # EMD opzionale (0.02 se vuoi un filo più di ordinalità)
                                    eta=0.30,               # mix hard/soft: mantiene picco netto sulla classe corretta
                                    lambda_dist=1.609,      # ln(5): vicino diretto ≈ 20% del centro nel kernel
                                    support_radius=1,       # azzera target oltre ±2 classi (niente picchi lontani)
                                    temperature=None,
                                    class_weights=weights).to('cuda')
        return HybridOrdinalLoss(num_classes=len(config.CLASSES[0]), class_weights=weights)
        return OrdinalConcentratedLoss(num_classes=len(config.CLASSES[0]), weights=weights,
                 alpha=2.5,
                 ce_weight=2.5,
                 w_far=10.0,      # Aumentiamo il peso per forzare probabilità zero lontano
                 w_conc=10.0,     # Peso per il nuovo termine di concentrazione
                 w_emd=1.5,
                 eps=1e-8)
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

def analyze_age_errors(all_preds_list, all_labels_list, all_probs_list, class_names, task_names, accuracies, output_dir):
    """
    Genera i grafici di analisi errori per l'etÃ  (task 0) e salva anche un file .npy
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

    # --- 1 - Distribuzione di probabilitÃ  per ogni classe (salvata separatamente)
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
            plt.title(f"Distribuzione di probabilitÃ  normalizzata - {class_names[c]}")
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

    # --- 4 - ProbabilitÃ  medie di scelta
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
    plt.title("ProbabilitÃ  media di scelta (Task 0 - Age)")
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

        # Use mixed precision on CUDA when available
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
    probs_list = []

    num_batch = 0
    with torch.inference_mode():
        for image, label in iterator:
            num_batch += 1

            image = image.to(device, non_blocking=True)
            labels = label[:, config.TASK].to(device, non_blocking=True)

            if compute_text_features:
                text_features = model.get_text_features(normalize=True)

            # Use mixed precision during validation forward pass on CUDA
            with autocast():
                image_features = model.get_image_features(image, normalize=True)
                logits = model.logit_scale.exp() * (image_features @ text_features.T)

                # Calcola probabilità, loss e predizioni
                probs = torch.softmax(logits, dim=1)
                loss, pred = loss_fn(logits, labels, return_predicted_label=True)
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

    # Get the training and validation dataset
    training_set = get_dataset(config=config,
                               split="train",
                               transform=get_image_transform(config),
                               augmentation_transform=get_augmentation_transform(config))
    validation_set = get_dataset(config=config,
                                 split="val",
                                 transform=get_image_transform(config))

    # Create dataloader for training and validation
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

    # Get the loss function
    weights = training_set.get_class_weights(int(config.TASK)).to(DEVICE)
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
    optimizer = torch.optim.AdamW(params, lr=config.LR, foreach=True, weight_decay=1e-4)

    # GradScaler for mixed precision
    scaler = GradScaler()
    # CosineAnnealingLR scheduler
    # 1) Warm-up lineare: da 0 -> lr iniziale, in warmup_steps step
    warmup = LinearLR(optimizer, start_factor=0.0, end_factor=1.0, total_iters=5)

    # 2) Cosine decay: dal lr corrente fino a ~0, per i restanti step
    cosine = CosineAnnealingLR(optimizer, T_max=10 - 5, eta_min=1e-8)

    # 3) Catena: warmup poi cosine
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[5])

    lr_history = [optimizer.param_groups[0]['lr']]

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
    patience = 8
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_accuracy = 0.0

    # Training loop
    os.makedirs(os.path.join(config.OUTPUT_DIR, "ckpt"), exist_ok=True)
    train_fn = single_task_train_fn
    val_fn = single_task_val_fn

    text_features = None
    if config.TUNING.lower() != 'softcpt':
        tokenizer = transforms.get_text_tokenizer(model.text_model.context_length)
        text = tokenizer(config.TEXT_CLASSES_PROMPT[config.TASK]).to(DEVICE)
        text_features = model.get_text_features(text=text, normalize=True)
        model.save(
            save_path=os.path.join(config.OUTPUT_DIR, f"ckpt/initial_model.pt"),
            text_features=text_features,
            text_features_path=os.path.join(config.OUTPUT_DIR, f"ckpt/vpt_text_features.pt"),
        )
        print(f"Text prompts: {config.TEXT_CLASSES_PROMPT[config.TASK]}")
        print(f"Text features shape: {text_features.shape}")

    print(f"\n[Epoch 0 - VAL]", flush=True)
    val_loss, val_acc, all_preds_list, all_labels_list, all_probs_list = val_fn(model, val_loader, loss_fn, DEVICE, None, config, text_features)
    print(f"  Task '{task_names[0]}': Loss = {val_loss[0]:.4f}, Accuracy = {val_acc[0]:.4f}")
    tracker.update_confusion(0, all_preds_list[0], all_labels_list[0], 100)
    tracker.save_confusion_matrices(100)
    tracker.save()

    weights_history = []
    alpha_max = 4.5
    warmup_epochs = 6 # porta la CE in primo piano nelle prime epoche
    def compute_alpha(e):
        if e < warmup_epochs:
            return alpha_max * (e + 1) / warmup_epochs
        return alpha_max

    alpha_history = []

    for epoch in range(config.EPOCHS):
        alpha_t = compute_alpha(epoch)
        if hasattr(loss_fn, "set_alpha"):
            loss_fn.set_alpha(float(alpha_t))
        alpha_history.append(float(alpha_t))
        print(f"[Epoch {epoch+1}] alpha={alpha_t:.3f}", flush=True)
        weights_history.append(1.0)  # Only one task
        train_loss, train_acc = train_fn(model, train_loader, optimizer, None, loss_fn, DEVICE, None, config, text_features, scaler)
        print(f"  Task '{task_names[0]}': Loss = {train_loss[0]:.4f}, Accuracy = {train_acc[0]:.4f}")
        tracker.update_loss(0, train_loss[0], train=True)
        tracker.update_accuracy(0, train_acc[0], train=True)

        # Validation
        print(f"\n[Epoch {epoch+1} - VAL]", flush=True)
        val_loss, val_acc, all_preds_list, all_labels_list, all_probs_list = val_fn(model, val_loader, loss_fn, DEVICE, None, config, text_features)
        print(f"  Task '{task_names[0]}': Loss = {val_loss[0]:.4f}, Accuracy = {val_acc[0]:.4f}")
        tracker.update_loss(0, val_loss[0], train=False)
        tracker.update_accuracy(0, val_acc[0], train=False)
        tracker.update_confusion(0, all_preds_list[0], all_labels_list[0], epoch)

        # Age analysis and save avg_probs matrix to match multitask outputs
        epoch_analysis_dir = os.path.join(config.OUTPUT_DIR, "age_analysis", f"epoch_{epoch+1}")
        analyze_age_errors(
            all_preds_list, all_labels_list, all_probs_list,
            class_names=config.CLASSES[0],
            task_names=task_names,
            accuracies=val_acc,
            output_dir=epoch_analysis_dir
        )
        labels = all_labels_list[0].numpy()
        probs = all_probs_list[0].numpy()
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

        tracker.save_confusion_matrices(epoch)
        tracker.plot_losses()
        tracker.save()

        if val_loss[0] < best_val_loss or val_acc[0] > best_accuracy:
            if val_loss[0] < best_val_loss:
                best_val_loss = val_loss[0]
                print(f"New best validation loss: {best_val_loss:.4f}. Saving model...")
                torch.save(model.state_dict(), os.path.join(config.OUTPUT_DIR, f"ckpt/best_model.pt"))
            if val_acc[0] > best_accuracy:
                best_accuracy = val_acc[0]
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


if __name__ == "__main__":
    main()