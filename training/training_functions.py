import torch
import os
from tqdm import tqdm
from matplotlib import pyplot as plt

from dataset.dataset import BaseDataset, MultiDataset
from wrappers.PerceptionEncoder.pe import PECore
from wrappers.promptopt.prompt_learner import CustomModel
from .loss import *
from core.vision_encoder import transforms

def _specific_task_train_epoch(model, optmizer, dataloader, losses, task_name, device, text_features=None, use_tqdm=False):
    model.train()
    epoch_loss = torch.tensor(0.0, device=device)
    total_correct = torch.tensor(0, device=device)
    total_samples = torch.tensor(0, device=device)

    # Use tqdm if requested, otherwise use regular enumerate
    iterator = tqdm(enumerate(dataloader), total=len(dataloader), desc="Training") if use_tqdm else enumerate(dataloader)

    for batch_idx, (image, labels) in iterator:
        image = image.to(device)
        labels = labels[task_name].to(device)

        # Forward pass
        if text_features is None:                
            # The only possible case where text_features is None is when we train the softCPT so
            # the get_text_features method don't need the text parameters as the text features
            # are dinamically generated from the init task names and classes names.
            text_features = model.get_text_features(normalize=True)
    
        image_features = model.get_image_features(image, normalize=True)

        logits  = model.logit_scale.exp() * (image_features @ text_features.t())        

        loss, predicted = losses(logits, labels, return_predicted_label=True)

        # Backward pass and optimization
        optmizer.zero_grad()
        loss.backward()
        optmizer.step()

        # Update epoch metrics - mantieni tutto su GPU
        epoch_loss += loss.detach()
        total_correct += (predicted == labels).sum()
        total_samples += labels.size(0)
        
        # Print progress only if not using tqdm
        if not use_tqdm:
            if batch_idx % 250 == 0:
                print(f"Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item()}", end='\r', flush=True)

    # Trasferimento su CPU solo alla fine per il calcolo finale
    epoch_accuracy = (total_correct / total_samples).cpu().item()
    epoch_loss = (epoch_loss / len(dataloader)).cpu()
    
    return epoch_loss, epoch_accuracy
    

def multitask_epoch_train():
    model.train()
    epoch_loss = torch.tensor(0.0, device=device)
    total_correct = 0
    total_samples = 0

    # Use tqdm if requested, otherwise use regular enumerate
    iterator = tqdm(enumerate(dataloader), total=len(dataloader), desc="Training") if use_tqdm else enumerate(dataloader)

    for batch_idx, (image, labels) in iterator:
        image = image.to(device)
        
        age_t = labels['age'].to(device)
        gender_t = labels['gender'].to(device)
        emotion_t = labels['emotion'].to(device)

        # Forward pass
        if text_features is None:                
            # The only possible case where text_features is None is when we train the softCPT so
            # the get_text_features method don't need the text parameters as the text features
            # are dinamically generated from the init task names and classes names.
            text_features = model.get_text_features(normalize=True)
    
        image_features = model.get_image_features(image, normalize=True)

        logits  = model.logit_scale.exp() * (image_features @ text_features.t())        

        loss, predicted = losses(logits, labels, return_predicted_label=True)

        # Backward pass and optimization
        optmizer.zero_grad()
        loss.backward()
        optmizer.step()

        # Update epoch metrics
        epoch_loss += loss.detach().item()
        total_correct += (predicted == labels).sum().item()  # Somma totale delle predizioni corrette
        total_samples += labels.size(0)  # Numero totale di campioni
        
        # Print progress only if not using tqdm
        if not use_tqdm:
            if batch_idx % 250 == 0:
                print(f"Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item()}", end='\r', flush=True)

    epoch_accuracy = total_correct / total_samples  # Accuratezza media su tutto il dataset
    return epoch_loss / len(dataloader), epoch_accuracy

#################### VALIDATION FUNCTIONS ####################
def _specific_task_val_epoch(model, dataloader, losses, task_name, device, text_features=None, use_tqdm=False):
    model.eval()
    epoch_loss = torch.tensor(0.0, device=device)
    total_correct = 0
    total_samples = 0
    all_labels = []
    all_preds = []
    # Use tqdm if requested, otherwise use regular enumerate
    iterator = tqdm(enumerate(dataloader), total=len(dataloader), desc="Validation") if use_tqdm else enumerate(dataloader)
    
    with torch.no_grad():
        for batch_idx, (image, labels) in iterator:
            image = image.to(device)
            labels = labels[task_name].to(device)
            # Forward pass
            if text_features is None:                
                # The only possible case where text_features is None is when we train the softCPT so
                # the get_text_features method don't need the text parameters as the text features
                # are dinamically generated from the init task names and classes names.
                text_features = model.get_text_features(normalize=True)
            
            image_features = model.get_image_features(image, normalize=True)

            logits  = model.logit_scale.exp() * (image_features @ text_features.t())        

            loss, predicted = losses(logits, labels, return_predicted_label=True)

            # Update epoch metrics
            epoch_loss += loss.detach().item()
            total_correct += (predicted == labels).sum().item()  # Somma totale delle predizioni corrette
            total_samples += labels.size(0)  # Numero totale di campioni
            
            all_labels.append(labels.cpu())
            all_preds.append(predicted.cpu())

            # Print progress only if not using tqdm
            if not use_tqdm:
                if batch_idx % 250 == 0:
                    print(f"Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item()}", end='\r', flush=True)            

    epoch_accuracy = total_correct / total_samples  # Accuratezza media su tutto il dataset
    return epoch_loss / len(dataloader), epoch_accuracy, all_preds, all_labels


def multitask_epoch_val():
    raise NotImplementedError("Multitask validation is not implemented yet.")


def get_model(cfg):
    '''
    Returns a model instance based on the specified type and number of prompts
    if type is 'softCPT', it returns a SoftCPT model with the specified number of prompts for the text tuning, with 0 Visual prompts.
    if type is 'VPT', it returns a VPT model with the specified number of prompts for the visual tuning, with 0 Text prompts.
    if type is 'Base', it returns a Base model with no prompts.
    '''
    model = None
    if cfg.MODEL_TYPE == 'softCPT':
        base_model = PECore.from_config("PE-Core-B16-224", pretrained=True, num_prompt=0)

        model = CustomModel(
            n_ctx=cfg.NUM_TEXT_PROMPTS,
            tasknames=cfg.TASK_NAMES,
            classnames=cfg.CLASSES,
            model=base_model,
            tokenizer=transforms.get_text_tokenizer(base_model.text_model.context_length)
        )
    elif cfg.MODEL_TYPE == 'VPT':
        model = PECore.from_config("PE-Core-B16-224", pretrained=True, num_prompt=cfg.NUM_VISUAL_PROMPTS)

    elif cfg.MODEL_TYPE == 'Base':
        model = PECore.from_config("PE-Core-B16-224", pretrained=True, num_prompt=0)
    else:
        raise ValueError(f"Unknown model type: {cfg.MODEL_TYPE}")

    return model

def get_datasets(dataset_names, split='train', transforms=None, dataset_root="../datasets_with_standard_labels"):
    if dataset_names is None:
        raise ValueError("Dataset names must be provided.")
    
    dataset = None

    if len(dataset_names) == 1:
        dataset = BaseDataset(
            root=os.path.join(dataset_root, dataset_names[0]),
            split=split,
            transform=transforms,
        )
    else:
        dataset = MultiDataset(
            dataset_names=dataset_names,
            split=split,
            transform=transforms,
            datasets_root=dataset_root,
            all_datasets = len(dataset_names) == 0
        )    
    
    return dataset


def get_training_step_fn(task):
    if task == 'multitask':
        return multitask_epoch_train
    return _specific_task_train_epoch


def get_validation_step_fn(task):
    if task == 'multitask':
        return multitask_epoch_val
    return _specific_task_val_epoch


def get_task_loss_fn(cfg, class_weights=None):
    if cfg.TASK == 'multitask':
        raise NotImplementedError("Multitask loss function is not implemented yet.")
    if cfg.TASK == 'age':
        print("Using AgeOrdinalLoss for age task. with class weights:", cfg.CLASS_WEIGHTS)
        return WeightedAgeOrdinalLoss(num_classes=len(cfg.CLASSES), weights=class_weights)
        #return AgeOrdinalLoss(num_classes=len(cfg.CLASSES))
    return CrossEntropyLoss()
    


def plot_losses(training_losses, validation_ordinal_losses, validation_ce_losses,
                training_accuracies, validation_ordinal_accuracies, validation_ce_accuracies,
                output_dir):
    print("Plotting and saving training curves...")
    os.makedirs(f'{output_dir}/plot', exist_ok=True)
    
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
    plt.savefig(f'{output_dir}/plot/losses_curve.png')
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
    plt.savefig(f'{output_dir}/plot/accuracies_curve.png')
    plt.close()

    print(f"Training curves saved in '{output_dir}'.")
