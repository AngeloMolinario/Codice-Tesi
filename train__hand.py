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
            base_model = PECore.from_config("PE-Core-B16-224", pretrained=True, num_prompt=0)
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
        age_loss = OrdinalLoss(num_classes=len(config.CLASSES[0]), weights=weights[0])
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
        T.RandomVerticalFlip(p=0.2),
        T.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.2),
        T.ToTensor()
    ]

    if model_name == 'pecore':
        tform.append(T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True))
    elif model_name == 'siglip2':
        raise NotImplementedError(f"Augmentation transform for model {model_name} is not implemented.")
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    return T.Compose(tform)

def multitask_train_fn(model, dataloader, optimizer, running_mean, loss_fn, device, task_weight, config, text_features=None):
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

    # Needed to compute the accuracy score
    preds_per_task = [[] for _ in range(num_task)] # For each task I will store the predicted labels
    labels_per_task = [[] for _ in range(num_task)] # For each task I will store the true labels

    scale = model.logit_scale.exp()
    if text_features is not None:
        text_features = text_features.T.contiguous()

    num_batch = 0
    for image, label in iterator:
        num_batch += 1

        image = image.to(device, non_blocking=True)
        labels = label.to(device, non_blocking=True)


        if compute_text_features:
            text_features = model.get_text_features(normalize=True).T.contiguous()        

        with torch.set_grad_enabled(config.NUM_VISUAL_PROMPT!=0):
            image_features = model.get_image_features(image, normalize=True)

        logits = scale * (image_features @ text_features)

        logits_by_task = torch.split(logits, logit_split, dim=1)  # tuple di view

        total_loss = 0.0

        for i in range(len(loss_fn)):
            loss_i, pred_i = loss_fn[i](logits_by_task[i], labels[:, i], return_predicted_label=True)
            losses_sums[i] += loss_i.detach()

            valid_mask = labels[:, i] != -1
            if valid_mask.any():
                preds_per_task[i].append(pred_i[valid_mask].detach().cpu())
                labels_per_task[i].append(labels[valid_mask, i].detach().cpu())
            
            

            scaled_loss = loss_i
            if running_mean is not None:
                prev_mean = running_mean.get_by_index(i)
                scale = 1.0 if (prev_mean is None) else max(prev_mean, 1e-8)
                scaled_loss = loss_i / scale
                running_mean.update_by_idx(loss_i.item(), i)
            else:
                scaled_loss = loss_i

            total_loss = total_loss + task_weight[i] * scaled_loss
        
            
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

        losses_sums[-1] += total_loss.detach()  
        if not config.USE_TQDM and (num_batch + 1)%100 == 0:
                print(f"Processed {num_batch + 1}/{len(iterator)}", end='\r')

    # Compute the average losses and accuracy
    accuracies = []
    for i in range(num_task):
        if preds_per_task[i]:  # se abbiamo dati validi
            all_preds  = torch.cat(preds_per_task[i])
            all_labels = torch.cat(labels_per_task[i])
            acc = (all_preds == all_labels).float().mean().item()
        else:
            acc = float('nan')  # nessun campione valido per il task
        accuracies.append(acc)

    mean_loss = [(losses_sums[i]/num_batch).item() for i in range(num_task+1)]

    return mean_loss, accuracies



def multitask_val_fn(model, dataloader, loss_fn, device, task_weight, config, text_features=None):
    model.eval()
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

    # Needed to compute the accuracy score
    preds_per_task = [[] for _ in range(num_task)] # For each task I will store the predicted labels
    labels_per_task = [[] for _ in range(num_task)] # For each task I will store the true labels

    num_batch = 0
    with torch.inference_mode():
        for image, label in iterator:
            num_batch += 1

            image = image.to(device, non_blocking=True)
            labels = label.to(device, non_blocking=True)


            if compute_text_features:
                text_features = model.get_text_features(normalize=True)
            
            image_features = model.get_image_features(image, normalize=True)

            logits = model.logit_scale.exp() * (image_features @ text_features.T)

            logits_by_task = torch.split(logits, logit_split, dim=1)  # tuple di view

            total_loss = 0.0

            for i in range(len(loss_fn)):
                loss_i, pred_i = loss_fn[i](logits_by_task[i], labels[:, i], return_predicted_label=True)
                losses_sums[i] += loss_i.detach()

                valid_mask = labels[:, i] != -1
                if valid_mask.any():
                    preds_per_task[i].append(pred_i[valid_mask].detach().cpu())
                    labels_per_task[i].append(labels[valid_mask, i].detach().cpu())

                total_loss = total_loss + task_weight[i] * loss_i

            losses_sums[-1] += total_loss.detach()  

            if not config.USE_TQDM and (num_batch + 1)%100 == 0:
                print(f"Processed {num_batch + 1}/{len(iterator)}", end='\r')

    # Compute the average losses and accuracy
    accuracies = []
    all_preds_list, all_labels_list = [], []
    for i in range(num_task):
        if preds_per_task[i]:
            all_preds  = torch.cat(preds_per_task[i],  dim=0)
            all_labels = torch.cat(labels_per_task[i], dim=0)
            acc = (all_preds == all_labels).float().mean().item()
        else:
            all_preds  = torch.empty(0, dtype=torch.long)
            all_labels = torch.empty(0, dtype=torch.long)
            acc = float('nan')

        accuracies.append(acc)
        all_preds_list.append(all_preds)
        all_labels_list.append(all_labels)

    mean_loss = [(losses_sums[i]/num_batch).item() for i in range(num_task+1)]

    return mean_loss, accuracies, all_preds_list, all_labels_list

def single_task_train_fn(model, dataloader, optimizer, running_mean, loss_fn, device, task_weight, config, text_features=None):
    ''' Gestisce unicamente il VPT'''
    model.train()    
    
    iterator = tqdm(dataloader) if config.USE_TQDM else dataloader        
    
    # Accumulatori per loss e metriche
    total_loss_sum = 0.0
    preds_list = []
    labels_list = []
    
    scale = model.logit_scale.exp()
    if text_features is not None:
        text_features = text_features.T.contiguous()
    
    num_batch = 0
    for image, label in iterator:
        num_batch += 1
        
        image = image.to(device, non_blocking=True)
        labels = label[:, config.TASK].to(device, non_blocking=True)
                                        
                
        image_features = model.get_image_features(image, normalize=True)
        
        logits = scale * (image_features @ text_features)
        
        # Calcola loss e predizioni
        loss, pred = loss_fn(logits, labels, return_predicted_label=True)
                
        
        # Backpropagation
        optimizer.zero_grad(set_to_none=True)
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
            labels = label.to(device, non_blocking=True)
            
            # Per single task, labels è 1D invece di 2D
            if labels.dim() > 1:
                labels = labels[:, config.TASK]
            
            if compute_text_features:
                text_features = model.get_text_features(normalize=True)
            
            image_features = model.get_image_features(image, normalize=True)
            logits = model.logit_scale.exp() * (image_features @ text_features.T)
            
            # Calcola loss e predizioni
            loss, pred = loss_fn(logits, labels, return_predicted_label=True)
            total_loss_sum += loss.detach()
            
            # Filtra campioni validi
            valid_mask = labels != -1
            if valid_mask.any():
                preds_list.append(pred[valid_mask].detach().cpu())
                labels_list.append(labels[valid_mask].detach().cpu())
            
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
        prefetch_factor=4,
        drop_last=True
)
    

    val_loader = DataLoader(
        dataset=validation_set,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        pin_memory_device="cuda",
        persistent_workers=True,
        prefetch_factor=4,
        drop_last=True,
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
    '''
    for i, loss in enumerate(loss_fn):
        if type(loss) is MaskedLoss:
            print(f"  Task {i}: {type(loss.base_loss)} with ignore index {loss.ignore_index}")
        else:
            print(f"  Task {i}: {type(loss)}")
    '''

    ########################## METRICS TRACKER #####################################################
    tracker = None

    num_tasks = len(config.TASK_NAMES) if config.TASK == -1 else 1
    output_dir = config.OUTPUT_DIR
    task_names = config.TASK_NAMES
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
            task_names=[task_names[config.TASK]],
            class_names=[class_names[config.TASK]]
        )
        print(f"Tracking metrics for task {task_names[config.TASK]}")

    ######################## EARLY STOPPING BASED ON LOSS ###################################

    patience = 5
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    ######################## START TRAINING LOOP ############################################
    os.makedirs(os.path.join(config.OUTPUT_DIR, "ckpt"), exist_ok=True)
    train_fn = get_step_fn(config, mode="train")
    val_fn   = get_step_fn(config, mode="val")

    running_mean = None
    if config.TASK == -1:
        running_mean = RunningMeans(['age', 'gender', 'emotion'], alpha=0.90)
    
    text_features = None
    if config.TUNING.lower() != 'softcpt':
        tokenizer = transforms.get_text_tokenizer(model.text_model.context_length)
        
        # Process text prompts by task to maintain task structure
        all_text_features = []
        for task_prompts in config.TEXT_CLASSES_PROMPT:
            text = tokenizer(task_prompts).to(DEVICE)
            task_text_features = model.get_text_features(text=text, normalize=True)
            all_text_features.append(task_text_features)
        
        # Concatenate all task text features
        if config.TASK == -1:
            text_features = torch.cat(all_text_features, dim=0)
        else:
            text_features = all_text_features[config.TASK]
        
        print(f"Text prompts by task: {config.TEXT_CLASSES_PROMPT}")
        print(f"Text features shape: {text_features.shape}")
        print(f"Text features shapes per task: {[tf.shape for tf in all_text_features]}")

    task_weight = training_set.get_task_weights().to(DEVICE)


    val_loss, val_acc, all_preds_list, all_labels_list = val_fn(model, val_loader, loss_fn, DEVICE, task_weight, config, text_features)
    

    for t in range(num_tasks):
        print(f"  Task '{task_names[t]}': Loss = {val_loss[t]:.4f}, Accuracy = {val_acc[t]:.4f}")
    if num_tasks > 1:
        print(f"  Total Loss: {val_loss[-1]:.4f}, Mean Accuracy : {sum(val_acc)/num_tasks:.4f}")


    for epoch in range(config.EPOCHS):

        train_loss, train_acc = train_fn(model, train_loader, optimizer, running_mean, loss_fn, DEVICE, task_weight, config, text_features)
        
        if running_mean is not None:
            running_mean.plot(os.path.join(config.OUTPUT_DIR, "running_mean_train.png"))
        
        print(f"\n[Epoch {epoch+1} - TRAIN]")
        for t in range(num_tasks):
            print(f"  Task '{task_names[t]}': Loss = {train_loss[t]:.4f}, Accuracy = {train_acc[t]:.4f}")
            tracker.update_loss(t, train_loss[t], train=True)
            tracker.update_accuracy(t, train_acc[t], train=True)
        if num_tasks > 1:
            print(f"  Total Loss: {train_loss[-1]:.4f}, Mean Accuracy : {sum(train_acc)/num_tasks:.4f}")        
            tracker.update_loss(None, train_loss[-1], train=True, multitask=True)
        tracker.update_accuracy(None, sum(train_acc)/num_tasks, train=True, mean=True)        

        # Validation
        val_loss, val_acc, all_preds_list, all_labels_list = val_fn(model, val_loader, loss_fn, DEVICE, task_weight, config, text_features)
        
        print(f"\n[Epoch {epoch+1} - VAL]")
        for t in range(num_tasks):
            print(f"  Task '{task_names[t]}': Loss = {val_loss[t]:.4f}, Accuracy = {val_acc[t]:.4f}")
            tracker.update_loss(t, val_loss[t], train=False)
            tracker.update_accuracy(t, val_acc[t], train=False)
            tracker.update_confusion(t, all_preds_list[t], all_labels_list[t], epoch)
        if num_tasks > 1:
            print(f"  Total Loss: {val_loss[-1]:.4f}, Mean Accuracy : {sum(val_acc)/num_tasks:.4f}")
            tracker.update_loss(None, val_loss[-1], train=False, multitask=True)
            tracker.update_accuracy(None, sum(val_acc)/num_tasks, train=False, mean=True)

        tracker.save_confusion_matrices(epoch)
        tracker.plot_losses()
        tracker.save()

        if val_loss[-1] < best_val_loss:
            best_val_loss = val_loss[-1]
            epochs_without_improvement = 0
            print(f"New best validation loss: {best_val_loss:.4f}. Saving model...")
            torch.save(model.state_dict(), os.path.join(config.OUTPUT_DIR, f"ckpt/best_model.pth"))
        else:
            epochs_without_improvement += 1
            print(f"No improvement in validation loss. Epochs without improvement: {epochs_without_improvement}/{patience}")

        if epochs_without_improvement >= patience:
            print("Early stopping triggered.")
            break

        torch.save(model.state_dict(), os.path.join(config.OUTPUT_DIR, f"ckpt/last_model.pth"))


main()