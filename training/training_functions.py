import torch
import os
from tqdm import tqdm

from dataset.dataset import BaseDataset, MultiDataset
from wrappers.PerceptionEncoder.pe import PECore
from wrappers.promptopt.prompt_learner import CustomModel
from .loss import *
from core.vision_encoder import transforms

def _specific_task_train_epoch(model, optmizer, dataloader, losses, task_name, device, text_features=None, use_tqdm=False):
    model.train()
    epoch_loss = torch.tensor(0.0, device=device)
    total_correct = 0
    total_samples = 0

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

        # Update epoch metrics
        epoch_loss += loss.detach().item()
        total_correct += (predicted == labels).sum().item()  # Somma totale delle predizioni corrette
        total_samples += labels.size(0)  # Numero totale di campioni
        
        # Print progress only if not using tqdm
        if not use_tqdm:
            if batch_idx % 20 == 0:
                print(f"Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item()}", end='\r', flush=True)


    epoch_accuracy = total_correct / total_samples  # Accuratezza media su tutto il dataset
    return epoch_loss / len(dataloader), epoch_accuracy
    

def multitask_epoch_train():
    raise NotImplementedError("Multitask training is not implemented yet.")

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
                if batch_idx % 25 == 0:
                    print(f"Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item()}", end='\r', flush=True)            

    epoch_accuracy = total_correct / total_samples  # Accuratezza media su tutto il dataset
    return epoch_loss / len(dataloader), epoch_accuracy, all_preds, all_labels


def multitask_epoch_val():
    raise NotImplementedError("Multitask validation is not implemented yet.")


def get_model(model_type: ['softCPT', 'VPT', 'Base'], num_prompt=0, tasknames=None, classnames=None):
    '''
    Returns a model instance based on the specified type and number of prompts
    if type is 'softCPT', it returns a SoftCPT model with the specified number of prompts for the text tuning, with 0 Visual prompts.
    if type is 'VPT', it returns a VPT model with the specified number of prompts for the visual tuning, with 0 Text prompts.
    if type is 'Base', it returns a Base model with no prompts.
    '''
    model = None
    if model_type == 'softCPT':
        base_model = PECore.from_config("PE-Core-B16-224", pretrained=True, num_prompt=0)

        model = CustomModel(
            n_ctx=num_prompt,
            tasknames=tasknames,
            classnames=classnames,
            model=base_model,
            tokenizer=transforms.get_text_tokenizer(base_model.text_model.context_length)
        )
    elif model_type == 'VPT':
        model = PECore.from_config("PE-Core-B16-224", pretrained=True, num_prompt=num_prompt)
    
    elif model_type == 'Base':
        model = PECore.from_config("PE-Core-B16-224", pretrained=True, num_prompt=0)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

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
            datasets_root=dataset_root
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


def get_task_loss_fn(task, num_classes=None):
    if task == 'multitask':
        raise NotImplementedError("Multitask loss function is not implemented yet.")
    if task == 'age':
        return AgeOrdinalLoss(num_classes=num_classes)
    return CrossEntropyLoss()