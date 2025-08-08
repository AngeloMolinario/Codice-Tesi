import torch
import os
from tqdm import tqdm
from matplotlib import pyplot as plt

from dataset.dataset import BaseDataset, MultiDataset, TaskBalanceDataset
from wrappers.PerceptionEncoder.pe import PECore
from wrappers.promptopt.prompt_learner import CustomModel
from .loss import *
from core.vision_encoder import transforms
from utils.running_mean import RunningMeans
class PrefetchLoader:
    """
    Wrap a DataLoader to asynchronously prefetch batches to the GPU using a
    separate CUDA stream. If the provided device is not CUDA, it simply yields
    from the original loader.
    """

    def __init__(self, loader, device):
        self.loader = loader
        self.device = device

    def __iter__(self):
        if self.device is None or self.device.type != "cuda":
            yield from self.loader
            return

        stream = torch.cuda.Stream()
        first = True
        for data in self.loader:
            with torch.cuda.stream(stream):
                images, labels = data
                images = images.to(self.device, non_blocking=True)

                if isinstance(labels, dict):
                    labels = {k: v.to(self.device, non_blocking=True) for k, v in labels.items()}
                elif isinstance(labels, (list, tuple)):
                    labels = [l.to(self.device, non_blocking=True) for l in labels]
                else:
                    labels = labels.to(self.device, non_blocking=True)

            if not first:
                yield next_images, next_labels
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)
            next_images = images
            next_labels = labels

        yield next_images, next_labels

    def __len__(self):
        return len(self.loader)


##############################################################################################
####            TRAINING AND VALIDATION FOR SINGLE AND MULTITASK                          ####
##############################################################################################
import torch
from tqdm import tqdm

# ==========================
# MULTITASK TRAIN/VAL EPOCH
# ==========================
def run_epoch_multitask(model, dataloader, losses, device, mode="train",
                        num_classes=None,            # es. [9, 2, 7]
                        text_features=None,          # tensor [sum(num_classes), D], già sul device
                        optimizer=None,
                        running_means=None,          # RunningMeans | None
                        task_weights=None,           # lista pesi per task; default [1]*T
                        use_tqdm=False,
                        return_preds=False,
                        use_prefetch=True):
    """
    Esegue un'epoca di training/validation in modalità MULTITASK.

    Parametri:
      - losses: lista di loss callable per task, ciascuna deve restituire (loss, pred) con return_predicted_label=True
      - num_classes: lista num classi per task, ordine coerente con losses e colonne di labels
      - text_features: embedding testuali già concatenati nell'ordine dei task (shape [sum(num_classes), D])
      - running_means: se fornito, usa EMA per scalare le loss (update SOLO in train). Se None, nessuno scaling
      - task_weights: pesi per combinare le loss scalate in total_loss
      - dataloader deve restituire (images, labels) con labels di shape (B, T), -1 = missing
    Ritorna:
      - losses_array: [T+1] -> [total_scaled, raw_task0, raw_task1, ...]
      - accuracies_array: [T+1] -> [overall_on_valids, acc_task0, acc_task1, ...]
      - (opzionale) all_preds, all_labels per task
    """
    is_train = (mode == "train")
    model.train() if is_train else model.eval()

    if num_classes is None:
        num_classes = [9, 2, 7]
    T = len(num_classes)

    if task_weights is None:
        task_weights = [1.0] * T

    # opzionale prefetch su GPU
    loader = dataloader
    if use_prefetch and device is not None and getattr(device, "type", None) == 'cuda':
        try:
            loader = PrefetchLoader(dataloader, device)
        except NameError:
            # se PrefetchLoader non esiste, si usa dataloader direttamente
            loader = dataloader

    # metriche (0 = total scaled; 1..T = raw per task)
    task_losses = torch.zeros(T + 1, device=device)
    task_correct = torch.zeros(T + 1, device=device)
    task_samples = torch.zeros(T + 1, device=device)

    if return_preds:
        all_labels = [[] for _ in range(T)]
        all_preds  = [[] for _ in range(T)]

    iterator = tqdm(loader, desc=f"{mode.capitalize()}") if use_tqdm else loader

    with torch.set_grad_enabled(is_train):
        for images, labels in iterator:
            images = images.to(device, non_blocking=True)  # (B, C, H, W)
            labels = labels.to(device, non_blocking=True)  # (B, T)

            # Image features per batch (puoi togliere no_grad se stai facendo VPT/LoRA visive)
            
            text_features = model.get_text_features(normalize=True)
            with torch.no_grad():
                image_features = model.get_image_features(images, normalize=True)

            # Logits e slicing per task
            all_logits = model.logit_scale.exp() * (image_features @ text_features.t())  # [B, sumC]
            logits_list = torch.split(all_logits, num_classes, dim=1)                     # [ [B,C0], [B,C1], ... ]

            raw_losses, preds_list, targets_list = [], [], []
            for i in range(T):
                targets_i = labels[:, i]
                loss_i, pred_i = losses[i](logits_list[i], targets_i, return_predicted_label=True)
                raw_losses.append(loss_i)
                preds_list.append(pred_i)
                targets_list.append(targets_i)

            # RunningMeans (se presente): update (solo train) e scaling
            if running_means is None:
                scaled_losses = raw_losses
            else:
                if is_train:
                    running_means.update([float(x.detach().item()) for x in raw_losses])
                current_rm = []
                for i in range(T):
                    v = running_means.get_by_index(i)
                    current_rm.append(1.0 if (v is None or v <= 0) else v)
                scaled_losses = [raw / rm for raw, rm in zip(raw_losses, current_rm)]

            total_loss = sum(w * l for w, l in zip(task_weights, scaled_losses))

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                optimizer.step()

            # accumuli: total (scaled) + raw per task
            task_losses[0] += total_loss.detach()
            for i in range(T):
                task_losses[i + 1] += raw_losses[i].detach()

            # accuracy per task (ignora -1)
            for i in range(T):
                mask = (targets_list[i] != -1)
                if mask.any():
                    task_correct[i + 1] += (preds_list[i][mask] == targets_list[i][mask]).sum()
                    task_samples[i + 1] += mask.sum()

            if return_preds:
                for i in range(T):
                    all_labels[i].append(targets_list[i].detach().cpu())
                    all_preds[i].append(preds_list[i].detach().cpu())

    # Finalizzazione metriche
    num_batches = len(dataloader)
    losses_array = (task_losses / max(1, num_batches)).detach().cpu().numpy()

    accuracies_array = torch.zeros(T + 1)
    for i in range(1, T + 1):
        if task_samples[i] > 0:
            accuracies_array[i] = (task_correct[i] / task_samples[i]).detach().cpu()
        else:
            accuracies_array[i] = 0.0
    total_correct = task_correct[1: T + 1].sum()
    total_samples = task_samples[1: T + 1].sum()
    accuracies_array[0] = (total_correct / total_samples).detach().cpu() if total_samples > 0 else 0.0

    if return_preds:
        return losses_array, accuracies_array.numpy(), all_preds, all_labels
    return losses_array, accuracies_array.numpy()


# ==========================
# SINGLE-TASK TRAIN/VAL EPOCH
# ==========================
def run_epoch_single_task(model, dataloader, loss_fn, device, mode="train",
                          task_index=0,               # indice del task singolo (0..T-1)
                          num_classes=None,           # es. [9, 2, 7]
                          text_features=None,         # tensor [sum(num_classes), D]
                          optimizer=None,
                          use_tqdm=False,
                          return_preds=False,
                          use_prefetch=True):
    """
    Esegue un'epoca di training/validation in modalità SINGLE-TASK.

    Parametri:
      - loss_fn: loss callable del task selezionato (deve restituire (loss, pred))
      - task_index: indice del task nelle colonne di labels e in num_classes
      - num_classes: lista num classi dei task totali (usata per lo slicing corretto dei logits)
      - text_features: embedding testuali concatenati per T task (shape [sum(num_classes), D])
      - NESSUN uso di RunningMeans in single-task, per design
    Ritorna:
      - losses_array: [2] -> [total_raw, raw_task]
      - accuracies_array: [2] -> [acc_task, acc_task] (duplicata per compat)
      - (opzionale) all_preds, all_labels
    """
    is_train = (mode == "train")
    model.train() if is_train else model.eval()

    if num_classes is None:
        num_classes = [9, 2, 7]
    T = len(num_classes)

    assert 0 <= task_index < T, f"task_index {task_index} out of range [0, {T-1}]"
    assert text_features is not None, "text_features deve essere passato ed essere già sul device corretto"
    assert text_features.shape[0] == sum(num_classes), (
        f"text_features ha {text_features.shape[0]} righe ma serve {sum(num_classes)} (somma di num_classes)"
    )

    # opzionale prefetch su GPU
    loader = dataloader
    if use_prefetch and device is not None and getattr(device, "type", None) == 'cuda':
        try:
            loader = PrefetchLoader(dataloader, device)
        except NameError:
            loader = dataloader

    # metriche (0 = total (= raw), 1 = raw)
    task_losses = torch.zeros(2, device=device)
    task_correct = torch.zeros(2, device=device)
    task_samples = torch.zeros(2, device=device)

    all_labels, all_preds = ([], []) if return_preds else ([], [])

    iterator = tqdm(loader, desc=f"{mode.capitalize()}") if use_tqdm else loader

    with torch.set_grad_enabled(is_train):
        for images, labels in iterator:
            if not (hasattr(images, 'device') and images.device == device):
                images = images.to(device)
            labels = labels.to(device)  # (B, T)

            with torch.no_grad():
                image_features = model.get_image_features(images, normalize=True)

            all_logits = model.logit_scale.exp() * (image_features @ text_features.t())  # [B, sumC]
            logits_list = torch.split(all_logits, num_classes, dim=1)

            targets = labels[:, task_index]
            loss_raw, pred = loss_fn(logits_list[task_index], targets, return_predicted_label=True)

            total_loss = loss_raw  # nessuna scala/EMA in single-task

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                optimizer.step()

            task_losses[0] += total_loss.detach()
            task_losses[1] += loss_raw.detach()

            mask = (targets != -1)
            if mask.any():
                task_correct[1] += (pred[mask] == targets[mask]).sum()
                task_samples[1] += mask.sum()

            if return_preds:
                all_labels.append(targets.detach().cpu())
                all_preds.append(pred.detach().cpu())

    num_batches = len(dataloader)
    losses_array = (task_losses / max(1, num_batches)).detach().cpu().numpy()

    accuracies_array = torch.zeros(2)
    if task_samples[1] > 0:
        accuracies_array[1] = (task_correct[1] / task_samples[1]).detach().cpu()
    accuracies_array[0] = accuracies_array[1]

    if return_preds:
        return losses_array, accuracies_array.numpy(), all_preds, all_labels
    return losses_array, accuracies_array.numpy()



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
