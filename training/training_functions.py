import torch
import os
from tqdm import tqdm
from matplotlib import pyplot as plt

from dataset.dataset import BaseDataset, MultiDataset, TaskBalanceDataset
from wrappers.PerceptionEncoder.pe import PECore
from wrappers.promptopt.prompt_learner import CustomModel
from .loss import *
from core.vision_encoder import transforms

def _specific_task_train_epoch(model, optimizer, dataloader, losses, task_name, device, text_features=None, use_tqdm=False):
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

        logits = model.logit_scale.exp() * (image_features @ text_features.t())        

        loss, predicted = losses(logits, labels, return_predicted_label=True)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update epoch metrics - mantieni tutto su GPU
        epoch_loss += loss.detach()
        
        # Update accuracy only for valid samples (!= -1)
        valid_mask = (labels != -1)
        if valid_mask.sum() > 0:
            total_correct += (predicted[valid_mask] == labels[valid_mask]).sum()
            total_samples += valid_mask.sum()
        
        # Print progress only if not using tqdm
        if not use_tqdm:
            if batch_idx % 250 == 0:
                print(f"Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item()}", end='\r', flush=True)

    # Trasferimento su CPU solo alla fine per il calcolo finale
    num_batches = len(dataloader)
    
    # Per compatibilità con multitask, restituisci array di lunghezza 4
    # [overall/single_task, unused, unused, unused] per loss
    # [overall, unused, unused, unused] per accuracy
    losses_array = torch.zeros(4)
    losses_array[0] = (epoch_loss / num_batches).cpu()
    
    accuracies_array = torch.zeros(4)
    if total_samples > 0:
        accuracies_array[0] = (total_correct / total_samples).cpu()
    else:
        accuracies_array[0] = 0.0
    
    return losses_array.numpy(), accuracies_array.numpy()
    
def multitask_epoch_train(model, optimizer, dataloader, losses, task_name, device, text_features=None, use_tqdm=False, num_classes=None, alpha=0.99, running_means_file="running_means.json"):
    model.train()
    task_weights = dataloader.dataset.get_task_weights().to(device)
    # Definisci il numero di classi per ogni task se non fornito
    if num_classes is None:
        num_classes = {'age': 9, 'gender': 2, 'emotion': 7}  # Valori di default
    
    # Load running means from file or initialize with default values
    import json
    if os.path.exists(running_means_file):
        try:
            with open(running_means_file, 'r') as f:
                running_means = json.load(f)
        except (json.JSONDecodeError, IOError):
            running_means = {'age': 1.0, 'gender': 1.0, 'emotion': 1.0}
    else:
        running_means = {'age': 1.0, 'gender': 1.0, 'emotion': 1.0}
    
    # Array per le metriche: [multitask/overall, age, gender, emotion]
    task_losses = torch.zeros(4, device=device)  # [multitask, age, gender, emotion]
    task_correct = torch.zeros(4, device=device)  # [overall, age, gender, emotion]
    task_samples = torch.zeros(4, device=device)  # [overall, age, gender, emotion]

    # Use tqdm if requested, otherwise use regular enumerate
    iterator = tqdm(enumerate(dataloader), total=len(dataloader), desc="Training") if use_tqdm else enumerate(dataloader)

    for batch_idx, (image, labels) in iterator:
        image = image.to(device)
        
        age_t = labels['age'].to(device)
        gender_t = labels['gender'].to(device)
        emotion_t = labels['emotion'].to(device)
                   
        text_features = model.get_text_features(normalize=True)
        with torch.no_grad():
            image_features = model.get_image_features(image, normalize=True)

        # Calcola i logits per tutti i task in una volta
        all_logits = model.logit_scale.exp() * (image_features @ text_features.t())        
        # Dividi i logits per ogni task basandosi sul numero di classi
        start_idx = 0
        
        # Age logits: primi num_classes['age'] logits
        age_end_idx = start_idx + num_classes['age']
        age_logits = all_logits[:, start_idx:age_end_idx]
        start_idx = age_end_idx
        
        # Gender logits: successivi num_classes['gender'] logits
        gender_end_idx = start_idx + num_classes['gender']
        gender_logits = all_logits[:, start_idx:gender_end_idx]
        start_idx = gender_end_idx
        
        # Emotion logits: rimanenti num_classes['emotion'] logits
        emotion_end_idx = start_idx + num_classes['emotion']
        emotion_logits = all_logits[:, start_idx:emotion_end_idx]

        age_loss, age_predicted = losses[0](age_logits, age_t, return_predicted_label=True)
        gender_loss, gender_predicted = losses[1](gender_logits, gender_t, return_predicted_label=True)
        emotion_loss, emotion_predicted = losses[2](emotion_logits, emotion_t, return_predicted_label=True)

        # === Update running means ===
        running_means['age'] = alpha * running_means['age'] + (1 - alpha) * age_loss.item()
        running_means['gender'] = alpha * running_means['gender'] + (1 - alpha) * gender_loss.item()
        running_means['emotion'] = alpha * running_means['emotion'] + (1 - alpha) * emotion_loss.item()

        # === Normalize losses ===
        age_loss_scaled = age_loss / (running_means['age'] + 1e-8)
        gender_loss_scaled = gender_loss / (running_means['gender'] + 1e-8)
        emotion_loss_scaled = emotion_loss / (running_means['emotion'] + 1e-8)

        # === Weighted loss ===
        loss = (task_weights[0] * age_loss_scaled +
                task_weights[1] * gender_loss_scaled +
                task_weights[2] * emotion_loss_scaled)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update losses: [multitask, age, gender, emotion]
        task_losses[0] += loss.detach()  # multitask loss (scaled)
        task_losses[1] += age_loss.detach()  # age loss (original)
        task_losses[2] += gender_loss.detach()  # gender loss (original)
        task_losses[3] += emotion_loss.detach()  # emotion loss (original)
        
        # Update accuracies per task (solo per campioni validi, != -1)
        age_valid_mask = (age_t != -1)
        gender_valid_mask = (gender_t != -1)
        emotion_valid_mask = (emotion_t != -1)
        
        if age_valid_mask.sum() > 0:
            age_correct = (age_predicted[age_valid_mask] == age_t[age_valid_mask]).sum()
            task_correct[1] += age_correct  # age accuracy
            task_samples[1] += age_valid_mask.sum()
            
        if gender_valid_mask.sum() > 0:
            gender_correct = (gender_predicted[gender_valid_mask] == gender_t[gender_valid_mask]).sum()
            task_correct[2] += gender_correct  # gender accuracy
            task_samples[2] += gender_valid_mask.sum()
            
        if emotion_valid_mask.sum() > 0:
            emotion_correct = (emotion_predicted[emotion_valid_mask] == emotion_t[emotion_valid_mask]).sum()
            task_correct[3] += emotion_correct  # emotion accuracy
            task_samples[3] += emotion_valid_mask.sum()
        
        # Print progress only if not using tqdm
        if not use_tqdm:
            if batch_idx % 250 == 0:
                print(f"Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item()}", end='\r', flush=True)

    # Save updated running means to file
    try:
        with open(running_means_file, 'w') as f:
            json.dump(running_means, f, indent=2)
    except IOError:
        print(f"Warning: Could not save running means to {running_means_file}")

    # Calcola le metriche finali trasferendo su CPU una sola volta
    num_batches = len(dataloader)
    
    # Loss array: [multitask, age, gender, emotion]
    losses_array = (task_losses / num_batches).cpu().numpy()
    
    # Accuracy array: [overall, age, gender, emotion]
    accuracies_array = torch.zeros(4)
    
    # Calculate per-task accuracies
    for i in range(1, 4):  # age, gender, emotion
        if task_samples[i] > 0:
            accuracies_array[i] = (task_correct[i] / task_samples[i]).cpu()
        else:
            accuracies_array[i] = 0.0
    
    # Calculate overall accuracy (weighted average)
    total_correct = task_correct[1] + task_correct[2] + task_correct[3]
    total_samples = task_samples[1] + task_samples[2] + task_samples[3]
    if total_samples > 0:
        accuracies_array[0] = (total_correct / total_samples).cpu()
    else:
        accuracies_array[0] = 0.0
    
    return losses_array, accuracies_array.numpy()

def old_multitask_epoch_train(model, optimizer, dataloader, losses, task_name, device, text_features=None, use_tqdm=False, num_classes=None):
    model.train()
    task_weights = dataloader.dataset.get_task_weights().to(device)
    # Definisci il numero di classi per ogni task se non fornito
    if num_classes is None:
        num_classes = {'age': 9, 'gender': 2, 'emotion': 7}  # Valori di default
    
    # Array per le metriche: [multitask/overall, age, gender, emotion]
    task_losses = torch.zeros(4, device=device)  # [multitask, age, gender, emotion]
    task_correct = torch.zeros(4, device=device)  # [overall, age, gender, emotion]
    task_samples = torch.zeros(4, device=device)  # [overall, age, gender, emotion]

    # Use tqdm if requested, otherwise use regular enumerate
    iterator = tqdm(enumerate(dataloader), total=len(dataloader), desc="Training") if use_tqdm else enumerate(dataloader)

    for batch_idx, (image, labels) in iterator:
        image = image.to(device)
        
        age_t = labels['age'].to(device)
        gender_t = labels['gender'].to(device)
        emotion_t = labels['emotion'].to(device)
                   
        text_features = model.get_text_features(normalize=True)
        with torch.no_grad():
            image_features = model.get_image_features(image, normalize=True)

        # Calcola i logits per tutti i task in una volta
        all_logits = model.logit_scale.exp() * (image_features @ text_features.t())        
        # Dividi i logits per ogni task basandosi sul numero di classi
        start_idx = 0
        
        # Age logits: primi num_classes['age'] logits
        age_end_idx = start_idx + num_classes['age']
        age_logits = all_logits[:, start_idx:age_end_idx]
        start_idx = age_end_idx
        
        # Gender logits: successivi num_classes['gender'] logits
        gender_end_idx = start_idx + num_classes['gender']
        gender_logits = all_logits[:, start_idx:gender_end_idx]
        start_idx = gender_end_idx
        
        # Emotion logits: rimanenti num_classes['emotion'] logits
        emotion_end_idx = start_idx + num_classes['emotion']
        emotion_logits = all_logits[:, start_idx:emotion_end_idx]

        age_loss, age_predicted = losses[0](age_logits, age_t, return_predicted_label=True)
        gender_loss, gender_predicted = losses[1](gender_logits, gender_t, return_predicted_label=True)
        emotion_loss, emotion_predicted = losses[2](emotion_logits, emotion_t, return_predicted_label=True)

        loss = task_weights[0]*age_loss + task_weights[1]*gender_loss + task_weights[2]*emotion_loss

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update losses: [multitask, age, gender, emotion]
        task_losses[0] += loss.detach()  # multitask loss
        task_losses[1] += age_loss.detach()  # age loss
        task_losses[2] += gender_loss.detach()  # gender loss
        task_losses[3] += emotion_loss.detach()  # emotion loss
        
        # Update accuracies per task (solo per campioni validi, != -1)
        age_valid_mask = (age_t != -1)
        gender_valid_mask = (gender_t != -1)
        emotion_valid_mask = (emotion_t != -1)
        
        if age_valid_mask.sum() > 0:
            age_correct = (age_predicted[age_valid_mask] == age_t[age_valid_mask]).sum()
            task_correct[1] += age_correct  # age accuracy
            task_samples[1] += age_valid_mask.sum()
            
        if gender_valid_mask.sum() > 0:
            gender_correct = (gender_predicted[gender_valid_mask] == gender_t[gender_valid_mask]).sum()
            task_correct[2] += gender_correct  # gender accuracy
            task_samples[2] += gender_valid_mask.sum()
            
        if emotion_valid_mask.sum() > 0:
            emotion_correct = (emotion_predicted[emotion_valid_mask] == emotion_t[emotion_valid_mask]).sum()
            task_correct[3] += emotion_correct  # emotion accuracy
            task_samples[3] += emotion_valid_mask.sum()
        
        # Print progress only if not using tqdm
        if not use_tqdm:
            if batch_idx % 250 == 0:
                print(f"Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item()}", end='\r', flush=True)

    # Calcola le metriche finali trasferendo su CPU una sola volta
    num_batches = len(dataloader)
    
    # Loss array: [multitask, age, gender, emotion]
    losses_array = (task_losses / num_batches).cpu().numpy()
    
    # Accuracy array: [overall, age, gender, emotion]
    accuracies_array = torch.zeros(4)
    
    # Calculate per-task accuracies
    for i in range(1, 4):  # age, gender, emotion
        if task_samples[i] > 0:
            accuracies_array[i] = (task_correct[i] / task_samples[i]).cpu()
        else:
            accuracies_array[i] = 0.0
    
    # Calculate overall accuracy (weighted average)
    total_correct = task_correct[1] + task_correct[2] + task_correct[3]
    total_samples = task_samples[1] + task_samples[2] + task_samples[3]
    if total_samples > 0:
        accuracies_array[0] = (total_correct / total_samples).cpu()
    else:
        accuracies_array[0] = 0.0
    
    return losses_array, accuracies_array.numpy()
    

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


def multitask_epoch_val(model, dataloader, losses, task_name,device, text_features=None, use_tqdm=False, num_classes=None):
    model.eval()  # Changed from model.train() to model.eval()
    task_weights = dataloader.dataset.get_task_weights().to(device)
    # Definisci il numero di classi per ogni task se non fornito
    if num_classes is None:
        num_classes = {'age': 9, 'gender': 2, 'emotion': 7}  # Valori di default
    
    # Array per le metriche: [multitask/overall, age, gender, emotion]
    task_losses = torch.zeros(4, device=device)  # [multitask, age, gender, emotion]
    task_correct = torch.zeros(4, device=device)  # [overall, age, gender, emotion]
    task_samples = torch.zeros(4, device=device)  # [overall, age, gender, emotion]

    # Lists for labels and predictions per task
    all_labels = [[], [], []]  # [age_labels, gender_labels, emotion_labels]
    all_preds = [[], [], []]   # [age_preds, gender_preds, emotion_preds]

    # Use tqdm if requested, otherwise use regular enumerate
    iterator = tqdm(enumerate(dataloader), total=len(dataloader), desc="Validation") if use_tqdm else enumerate(dataloader)
    with torch.no_grad():
        for batch_idx, (image, labels) in iterator:
            image = image.to(device)
            
            age_t = labels['age'].to(device)
            gender_t = labels['gender'].to(device)
            emotion_t = labels['emotion'].to(device)
                    
            text_features = model.get_text_features(normalize=True)

            image_features = model.get_image_features(image, normalize=True)

            # Calcola i logits per tutti i task in una volta
            all_logits = model.logit_scale.exp() * (image_features @ text_features.t())
            
            # Dividi i logits per ogni task basandosi sul numero di classi
            start_idx = 0
            
            # Age logits: primi num_classes['age'] logits
            age_end_idx = start_idx + num_classes['age']
            age_logits = all_logits[:, start_idx:age_end_idx]
            start_idx = age_end_idx
            
            # Gender logits: successivi num_classes['gender'] logits
            gender_end_idx = start_idx + num_classes['gender']
            gender_logits = all_logits[:, start_idx:gender_end_idx]
            start_idx = gender_end_idx
            
            # Emotion logits: rimanenti num_classes['emotion'] logits
            emotion_end_idx = start_idx + num_classes['emotion']
            emotion_logits = all_logits[:, start_idx:emotion_end_idx]

            age_loss = losses[0](age_logits, age_t, return_predicted_label=False)
            gender_loss = losses[1](gender_logits, gender_t, return_predicted_label=False)
            emotion_loss = losses[2](emotion_logits, emotion_t, return_predicted_label=False)


            age_probs = torch.softmax(age_logits, dim=1)
            age_indices = torch.arange(9, device=age_probs.device).float()
            age_pred_continuous = torch.sum(age_probs * age_indices.unsqueeze(0), dim=1)
            age_predicted = torch.round(age_pred_continuous).long()
            gender_predicted = torch.argmax(torch.softmax(gender_logits, dim=1), dim=1)
            emotion_predicted = torch.argmax(torch.softmax(emotion_logits, dim=1), dim=1)


            loss = task_weights[0]*age_loss + task_weights[1]*gender_loss + task_weights[2]*emotion_loss

            # Update losses: [multitask, age, gender, emotion]
            task_losses[0] += loss.detach()  # multitask loss
            task_losses[1] += age_loss.detach()  # age loss
            task_losses[2] += gender_loss.detach()  # gender loss
            task_losses[3] += emotion_loss.detach()  # emotion loss
            
            # Store labels and predictions for each task
            all_labels[0].append(age_t.cpu())      # age labels
            all_labels[1].append(gender_t.cpu())   # gender labels
            all_labels[2].append(emotion_t.cpu())  # emotion labels
            
            all_preds[0].append(age_predicted.cpu())      # age predictions
            all_preds[1].append(gender_predicted.cpu())   # gender predictions
            all_preds[2].append(emotion_predicted.cpu())  # emotion predictions
            
            # Update accuracies per task (solo per campioni validi, != -1)
            age_valid_mask = (age_t != -1)
            gender_valid_mask = (gender_t != -1)
            emotion_valid_mask = (emotion_t != -1)
            
            if age_valid_mask.sum() > 0:
                age_correct = (age_predicted[age_valid_mask] == age_t[age_valid_mask]).sum()
                task_correct[1] += age_correct  # age accuracy
                task_samples[1] += age_valid_mask.sum()
                
            if gender_valid_mask.sum() > 0:
                gender_correct = (gender_predicted[gender_valid_mask] == gender_t[gender_valid_mask]).sum()
                task_correct[2] += gender_correct  # gender accuracy
                task_samples[2] += gender_valid_mask.sum()
                
            if emotion_valid_mask.sum() > 0:
                emotion_correct = (emotion_predicted[emotion_valid_mask] == emotion_t[emotion_valid_mask]).sum()
                task_correct[3] += emotion_correct  # emotion accuracy
                task_samples[3] += emotion_valid_mask.sum()
            
            # Print progress only if not using tqdm
            if not use_tqdm:
                if batch_idx % 250 == 0:
                    print(f"Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item()}", end='\r', flush=True)

    # Calcola le metriche finali trasferendo su CPU una sola volta
    num_batches = len(dataloader)
    
    # Loss array: [multitask, age, gender, emotion]
    losses_array = (task_losses / num_batches).cpu().numpy()
    
    # Accuracy array: [overall, age, gender, emotion]
    accuracies_array = torch.zeros(4)
    
    # Calculate per-task accuracies
    for i in range(1, 4):  # age, gender, emotion
        if task_samples[i] > 0:
            accuracies_array[i] = (task_correct[i] / task_samples[i]).cpu()
        else:
            accuracies_array[i] = 0.0
    
    # Calculate overall accuracy (weighted average)
    total_correct = task_correct[1] + task_correct[2] + task_correct[3]
    total_samples = task_samples[1] + task_samples[2] + task_samples[3]
    if total_samples > 0:
        accuracies_array[0] = (total_correct / total_samples).cpu()
    else:
        accuracies_array[0] = 0.0
    
    return losses_array, accuracies_array.numpy(), all_preds, all_labels


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

def get_datasets(dataset_names, split='train', transforms=None, dataset_root="../datasets_with_standard_labels", config=None, validation_sample=50):
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
        if split == 'train' and hasattr(config, "NUM_SAMPLES_PER_CLASS"):
            # Use TaskBalanceDataset for training with task balancing
            balance_task = getattr(config, 'BALANCE_TASK', None)
            dataset = TaskBalanceDataset(
                dataset_names=dataset_names,
                split=split,
                transform=transforms,
                datasets_root=dataset_root,
                all_datasets=len(dataset_names) == 0,
                balance_task=balance_task
            )
        else:
            # Use regular MultiDataset for validation or when no balancing is needed
            dataset = MultiDataset(
                dataset_names=dataset_names,
                split=split,
                transform=transforms,
                datasets_root=dataset_root,
                all_datasets=len(dataset_names) == 0
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


def get_task_loss_fn(cfg, weights=None):
    if cfg.TASK == 'multitask':
        age_loss = WeightedAgeOrdinalLoss(num_classes=len(cfg.CLASSES[0]), weights=weights[0]) if weights else AgeOrdinalLoss(num_classes=len(cfg.CLASSES[0]))
        gender_loss = CrossEntropyLoss(weights=weights[1]) if weights else CrossEntropyLoss()
        emotion_loss = CrossEntropyLoss(weights=weights[2]) if weights else CrossEntropyLoss()

        age_masked = MaskedLoss(age_loss)
        gender_masked = MaskedLoss(gender_loss)
        emotion_masked = MaskedLoss(emotion_loss)

        return [age_masked, gender_masked, emotion_masked]

    if cfg.TASK == 'age':
        return WeightedAgeOrdinalLoss(num_classes=len(cfg.CLASSES), weights=weights)
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
