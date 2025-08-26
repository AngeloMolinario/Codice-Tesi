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
from wrappers.SigLip2.SigLip2Model import Siglip2Model
from wrappers.tokenizer import PETokenizer, SigLip2Tokenizer
from transformers import AutoConfig, AutoTokenizer
from training.loss import *
from core.vision_encoder import transforms
from utils.metric import MultitaskTracker
from utils.configuration import Config
from utils.running_mean import RunningMeans
from tqdm import tqdm
from training.training_functions import *


def get_loss_fn(config, weights=None):    
    '''
        Get a list of loss functions with the following position:
        0. Age loss
        1. Gender loss
        2. Emotion loss
    '''
    age_loss = OrdinalAgeLossEMD(num_classes=len(config.CLASSES[0]), class_frequencies=weights[0], lambda_ordinal=1.0)

    gender_loss = CrossEntropyLoss(num_classes=len(config.CLASSES[1]), weights=weights[1])
    emotion_loss = CrossEntropyLoss(num_classes=len(config.CLASSES[2]), weights=weights[2])
    loss = [
        MaskedLoss(age_loss, -1),
        MaskedLoss(gender_loss, -1),
        MaskedLoss(emotion_loss, -1)
    ]
    return loss

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
        tform.append(T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True))
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    return T.Compose(tform)

def multitask_train_fn(model, dataloader, optimizer, running_mean, loss_fn, device, task_weight, config, text_features=None, scaler=None):
    model.train()
    compute_text_features = text_features is None or config.NUM_TEXT_CNTX > 0
    # If I am training the SoftCPT model than text_embedding will be None, otherwise they doesn't change during the training

    iterator = tqdm(dataloader) if config.USE_TQDM else dataloader

    logit_split = [len(c) for c in config.CLASSES] # (9, 2, 7)
    num_task = len(loss_fn)

    if not torch.is_tensor(task_weight):
        task_weight = torch.as_tensor(task_weight, dtype=torch.float32, device=device)
    else:
        task_weight = task_weight.to(device)

    # accumultaros for loss and metrics tracking
    losses_sums = torch.zeros(num_task+1, device=device)  # +1 for the multitask total loss

    # NOTE: Not all the task are labeled so to compute the correct accuracy for the task we need to count the number of sample labeles
    correct = [torch.zeros(1, device=device) for _ in range(num_task)] # Correct predictions for each task
    count   = [torch.zeros(1, device=device) for _ in range(num_task)] # Total predictions for each task   

    # If text_features are passed than we need to transpose them and make them contiguous so to speed up the row matrix multiplication
    if text_features is not None:
        text_features = text_features.T.contiguous()

    num_batch = 0 # Needed to compute the final mean value
    for image, label in iterator:
        num_batch += 1

        image = image.to(device, non_blocking=True)
        labels = label.to(device, non_blocking=True)

        if compute_text_features:
            text_features = model.get_text_features(normalize=True).T.contiguous()        

        scale = 1.0
        if hasattr(model, 'logit_scale'):
            scale = model.logit_scale.exp()
        bias = 0.0
        if hasattr(model, 'logit_bias'):
            bias = model.logit_bias

        # Use mixed precision on CUDA when available
        with autocast(device_type=device):
            # If the number of Visual prompt is 0 than we don't need the gradient for the Visual model
            with torch.set_grad_enabled(config.NUM_VISUAL_PROMPT!=0):
                image_features = model.get_image_features(image, normalize=True)

            logits = scale * (image_features @ text_features) + bias
            logits_by_task = torch.split(logits, logit_split, dim=1)

            total_loss = 0.0

            for i in range(len(loss_fn)):
                loss_i = loss_fn[i](logits_by_task[i], labels[:, i])
                pred_i = logits_by_task[i].argmax(dim=1)
                losses_sums[i] += loss_i.detach()

                valid_mask = labels[:, i] != -1
                if valid_mask.any():
                    correct[i] += (pred_i[valid_mask] == labels[valid_mask, i]).sum()
                    count[i]   += valid_mask.sum()

                if running_mean is not None:
                    # Update the running mean for balancing the weights of the multitask loss
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
    accuracies = [(correct[i] / count[i]).item() if count[i] > 0 else float('nan') for i in range(num_task)]

    mean_loss = [(losses_sums[i]/num_batch).item() for i in range(num_task+1)]

    return mean_loss, accuracies


# ---------------------------------------------------------
# Funzione di validazione
# ---------------------------------------------------------
def multitask_val_fn(model, dataloader, loss_fn, device, task_weight, config, text_features=None):
    model.eval()
    compute_text_features = False #text_features is None or config.NUM_TEXT_CNTX > 0


    if text_features is not None:
        text_features = text_features.T.contiguous()


    iterator = tqdm(dataloader) if config.USE_TQDM else dataloader

    logit_split = [len(c) for c in config.CLASSES]
    num_task = len(loss_fn)

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
                logits = scale * (image_features @ text_features) + bias
                logits_by_task = torch.split(logits, logit_split, dim=1)

                total_loss = 0.0

                for i in range(len(loss_fn)):
                    loss_i = loss_fn[i](logits_by_task[i], labels[:, i])
                    pred_i = logits_by_task[i].argmax(dim=1)
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
    
    #############################################################################################
    ##                            Load the given configuration file                            ##
    #############################################################################################

    configuration_path = sys.argv[1] if len(sys.argv) > 1 else "config/PECore_VPT_age.json"
    config = Config(configuration_path)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")
    print(f"Loaded configuration: {config}")
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    shutil.copy2(configuration_path, f'{config.OUTPUT_DIR}/training_configuration.json')

    #############################################################################################
    ##                               Load correct model                                        ##
    #############################################################################################

    model = get_model(config).to(DEVICE)    

    #############################################################################################
    ##                         Dataset and Dataloade building                                  ##
    #############################################################################################

    training_set =  get_dataset(config=config,
                               split="train",
                               transform=get_image_transform(config),
                               augmentation_transform=get_augmentation_transform(config)
                               )
    validation_set = get_dataset(config=config,
                                 split="val",
                                 transform=get_image_transform(config)
                                )

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

    #############################################################################################
    ##               Loss function and class weights configuration                             ##
    #############################################################################################


    # Gender task is more or less balanced, the age and emotion task are unbalanced, in particular
    # we use inverse_sqrt as parameters to get the invecerse rooted weights to give more weight to 
    # rare classes and they are normalized so that the max weight is 1.0 and the other are a fraction of it
    weights = [
        training_set.get_class_weights(0, "inverse_sqrt").to(DEVICE),   
        training_set.get_class_weights(1, "default").to(DEVICE),        
        training_set.get_class_weights(2, "inverse_sqrt").to(DEVICE),
        ]
    
    loss_fn = get_loss_fn(config, weights=weights)

    #############################################################################################
    ##                        Optimizer creation and configuration                             ##
    #############################################################################################

    optimizer = None
    params = []                 # Lista dei parametri apprendibili
    total_trainable_params = 0
    for name, param in model.named_parameters():
        # if the name defined in the config file are a substring of the parameter's name that 
        if any(trainable_param in name for trainable_param in config.NAMED_TRAINABLE_PARAMETERS):
            param.requires_grad = True
            params += [param]
            total_trainable_params += param.numel()
            print(f"Parameter: {name}, shape: {param.shape}, numel: {param.numel()}")        
        else:
            param.requires_grad = False
    optimizer = torch.optim.AdamW(params, lr=config.LR, foreach=True, weight_decay=0.0)

    # Add GradScaler for mixed precision
    scaler = GradScaler(device=DEVICE)
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
        tokenizer = get_tokenizer(config)
        
        task_text_features = []
        for task_prompts in config.TEXT_CLASSES_PROMPT:
            for classes in task_prompts:
                text = tokenizer(classes).to(DEVICE)
                classes_features = F.normalize(model.get_text_features(text=text, normalize=False).mean(dim=0), dim=-1)
                task_text_features.append(classes_features)


        model.save(
            save_path=os.path.join(config.OUTPUT_DIR, f"ckpt/initial_model.pt"),
            text_features=torch.stack(task_text_features, dim=0),
            text_features_path=os.path.join(config.OUTPUT_DIR, f"ckpt/vpt_text_features.pt"),
        )
        text_features = torch.stack(task_text_features, dim=0)
        print(f"Text features shape: {text_features.shape}")


    data_task_weight = training_set.get_task_weights()
    print(f"Task weights: {data_task_weight}")

    print(f"\n[Epoch 0 - VAL]", flush=True)
    val_loss, val_acc, all_preds_list, all_labels_list, _ = val_fn(model, val_loader, loss_fn, DEVICE, torch.ones(3).to(DEVICE), config, text_features)
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


    for epoch in range(config.EPOCHS):
        
        print(f"[Epoch {epoch+1}]", flush=True)
        
        # TASK WEIGHTS COMPUTATION
        w = []
        for i in range(num_tasks):
            w_i = running_mean.get_by_index(i)
            if w_i is None:
                w_i=1.0
            w.append((1.0 / max(w_i, 1e-8))* data_task_weight[i])
        max_raw = max(w)
        task_weight = torch.tensor([r / max(max_raw, 1e-8) for r in w], device=DEVICE)


        print(f"Task weights (EMA inverse) for epoch {epoch+1}: {task_weight.tolist()}")
        weights_history.append(task_weight.cpu().numpy())

        train_loss, train_acc = train_fn(
            model,
            train_loader,
            optimizer,
            running_mean,
            loss_fn,
            DEVICE,
            task_weight,
            config,
            text_features if config.TUNING.lower()!="softcpt" else None,
            scaler)
        
        running_mean.plot(os.path.join(config.OUTPUT_DIR, "running_mean_train.png"))
        for t in range(num_tasks):
            print(f"  Task '{task_names[t]}': Loss = {train_loss[t]:.4f}, Accuracy = {train_acc[t]:.4f}")
            tracker.update_loss(t, train_loss[t], train=True)
            tracker.update_accuracy(t, train_acc[t], train=True)
        print(f"  Total Loss: {sum(train_loss[:-1]):.4f}, Total weighted loss {train_loss[-1]:.4f}, Mean Accuracy : {sum(train_acc)/num_tasks:.4f}")        

        if hasattr(model, 'logit_scale'):
            print(f"Logit scale: {model.logit_scale.item()} - exp({model.logit_scale.exp().item()})")
        if hasattr(model, 'logit_bias'):
            print(f"Logit bias: {model.logit_bias.item()}")
            
        tracker.update_loss(None, sum(train_loss[:-1]), train=True, multitask=True)
        tracker.update_accuracy(None, sum(train_acc)/num_tasks, train=True, mean=True)        

        # Validation
        print(f"\n[Epoch {epoch+1} - VAL]", flush=True)

        if config.TUNING.lower() == 'softcpt':
            text_features = model.get_text_features(normalize=True)

        val_loss, val_acc, all_preds_list, all_labels_list, all_probs_list = val_fn(
            model,
            val_loader,
            loss_fn,
            DEVICE,
            task_weight,
            config,
            text_features
        )

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