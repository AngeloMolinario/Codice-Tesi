import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms as T
from transformers import AutoConfig
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np
import os
from matplotlib import pyplot as plt

# Import your custom modules (adjust paths as needed)
from core.vision_encoder.pe import CLIP
from core.vision_encoder.config import *
from core.vision_encoder.transforms import get_image_transform
from dataset.dataset import BaseDataset
from wrappers.PerceptionEncoder.pe import PECore_Vision
from wrappers.SigLip2.SigLip2Model import Siglip2Vision
from wrappers.tokenizer import *

CLASSES = [
    ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"],
    ["male", "female"],
    ["surprise", "fear", "disgust", "happy", "sad", "angry", "neutral"]
]

def compute_prf_metrics(all_true_labels, all_pred_labels):
    """Compute macro Precision/Recall/F1 per task."""
    prf_task = {"precision": [0.0, 0.0, 0.0], "recall": [0.0, 0.0, 0.0], "f1": [0.0, 0.0, 0.0]}
    
    for task_idx in range(3):
        y_true, y_pred = all_true_labels[task_idx], all_pred_labels[task_idx]
        if y_true and y_pred:
            try:
                # Assicurati che i labels siano nell'intervallo corretto
                unique_labels = sorted(set(y_true + y_pred))
                valid_labels = [l for l in unique_labels if 0 <= l < len(CLASSES[task_idx])]
                
                p, r, f1, _ = precision_recall_fscore_support(
                    y_true, y_pred, labels=valid_labels, 
                    average='macro', zero_division=0
                )
                prf_task["precision"][task_idx] = float(p)
                prf_task["recall"][task_idx] = float(r)
                prf_task["f1"][task_idx] = float(f1)
            except Exception as e:
                print(f"Warning: Error computing metrics for task {task_idx}: {e}")
    return prf_task

def format_table(headers, rows):
    """Format table without external dependencies."""
    # Calculate column widths
    col_widths = [len(header) for header in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))
    
    # Create separator line
    separator = "+" + "+".join("-" * (width + 2) for width in col_widths) + "+"
    
    # Format header
    header_row = "|" + "|".join(f" {header:<{col_widths[i]}} " for i, header in enumerate(headers)) + "|"
    
    # Format data rows
    formatted_rows = []
    for row in rows:
        formatted_row = "|" + "|".join(f" {str(cell):<{col_widths[i]}} " for i, cell in enumerate(row)) + "|"
        formatted_rows.append(formatted_row)
    
    # Combine all parts
    table_lines = [separator, header_row, separator] + formatted_rows + [separator]
    return "\n".join(table_lines)

def write_results(output_dir, top1_acc, top2_acc, onebin_acc, prf_task=None):
    """Write results to file."""
    os.makedirs(output_dir, exist_ok=True)
    
    tasks = ['age', 'gender', 'emotion']
    rows = []
    
    for i in range(len(top1_acc)):
        prec = recall = f1 = "N/A"
        if prf_task:
            prec = f"{prf_task['precision'][i]:.4f}"
            recall = f"{prf_task['recall'][i]:.4f}"
            f1 = f"{prf_task['f1'][i]:.4f}"
        
        rows.append([
            f"Task {tasks[i]}",
            f"{top1_acc[i]:.4f}",
            f"{top2_acc[i]:.4f}",
            f"{onebin_acc[i]:.4f}" if i == 0 else "N/A",
            prec, recall, f1
        ])
    
    avg_top1 = sum(top1_acc) / len(top1_acc)
    rows.append(["Average", f"{avg_top1:.4f}", "N/A", "N/A", "N/A", "N/A", "N/A"])
    
    headers = ["Task", "Top-1", "Top-2", "1-Bin (Age)", "Precision", "Recall", "F1"]
    table_str = format_table(headers, rows)
    
    with open(os.path.join(output_dir, "results.txt"), "w") as f:
        f.write(table_str)
    
    print(table_str)
    return table_str

def save_confusion_matrix(y_true, y_pred, class_names, task_name, output_dir):
    """Save confusion matrix as image with numeric annotations."""
    if not y_true or not y_pred:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Assicurati che i labels siano nell'intervallo corretto
    unique_labels = sorted(set(y_true + y_pred))
    valid_labels = [l for l in unique_labels if 0 <= l < len(class_names)]
    
    if not valid_labels:
        print(f"Warning: No valid labels found for {task_name}")
        return
    
    # Crea la confusion matrix solo per i labels validi
    cm = confusion_matrix(y_true, y_pred, labels=valid_labels)
    valid_class_names = [class_names[i] for i in valid_labels]
    
    # Calcola le percentuali per una migliore visualizzazione
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)  # Gestisci divisioni per zero
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cm_normalized, interpolation='nearest', cmap='Blues')
    plt.title(f'Confusion Matrix - {task_name}', fontsize=14, fontweight='bold')
    plt.colorbar(label='Normalized Frequency')
    
    tick_marks = np.arange(len(valid_class_names))
    plt.xticks(tick_marks, valid_class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, valid_class_names)
    
    # Aggiungi annotazioni numeriche
    thresh = cm_normalized.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        # Mostra sia il valore assoluto che la percentuale
        text = f'{cm[i, j]}\n({cm_normalized[i, j]:.2%})'
        plt.text(j, i, text, ha="center", va="center",
                color="white" if cm_normalized[i, j] > thresh else "black",
                fontsize=9, fontweight='bold')
    
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    
    # Salva anche una versione con valori assoluti
    plt.savefig(os.path.join(output_dir, f'confusion_matrix_{task_name.lower()}_normalized.png'), 
                dpi=300, bbox_inches='tight')
    
    # Crea anche una matrice con solo valori assoluti
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title(f'Confusion Matrix (Absolute Values) - {task_name}', fontsize=14, fontweight='bold')
    plt.colorbar(label='Count')
    
    plt.xticks(tick_marks, valid_class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, valid_class_names)
    
    # Aggiungi annotazioni numeriche (solo valori assoluti)
    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, f'{cm[i, j]}', ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=10, fontweight='bold')
    
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'confusion_matrix_{task_name.lower()}_absolute.png'), 
                dpi=300, bbox_inches='tight')
    plt.close('all')  # Chiudi tutte le figure per liberare memoria

def save_class_histograms(y_true, y_pred, class_names, task_name, output_dir):
    """Save classification histograms for each true class showing correct vs incorrect predictions."""
    if not y_true or not y_pred:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Converti in numpy arrays per facilità di manipolazione
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    
    # Assicurati che i labels siano nell'intervallo corretto
    valid_mask = (y_true_arr >= 0) & (y_true_arr < len(class_names)) & \
                 (y_pred_arr >= 0) & (y_pred_arr < len(class_names))
    
    if not np.any(valid_mask):
        print(f"Warning: No valid predictions found for {task_name}")
        return
    
    y_true_valid = y_true_arr[valid_mask]
    y_pred_valid = y_pred_arr[valid_mask]
    
    # Crea istogramma per ogni classe vera
    unique_true_classes = sorted(set(y_true_valid))
    
    for true_class in unique_true_classes:
        if true_class < 0 or true_class >= len(class_names):
            continue
            
        # Seleziona tutti i campioni che appartengono a questa classe vera
        class_mask = y_true_valid == true_class
        if not np.any(class_mask):
            continue
            
        predictions_for_class = y_pred_valid[class_mask]
        
        # Conta le predizioni per ogni classe
        correct_counts = np.zeros(len(class_names))
        incorrect_counts = np.zeros(len(class_names))
        
        for pred in predictions_for_class:
            if pred == true_class:
                correct_counts[pred] += 1
            else:
                incorrect_counts[pred] += 1
        
        # Crea l'istogramma
        plt.figure(figsize=(12, 8))
        
        x = np.arange(len(class_names))
        width = 0.8
        
        # Bar per predizioni corrette (verde) e scorrette (rosso)
        bars_correct = plt.bar(x, correct_counts, width, label='Correct Predictions', 
                              color='green', alpha=0.7, edgecolor='darkgreen', linewidth=1)
        bars_incorrect = plt.bar(x, incorrect_counts, width, bottom=correct_counts, 
                                label='Incorrect Predictions', color='red', alpha=0.7, 
                                edgecolor='darkred', linewidth=1)
        
        # Aggiungi etichette numeriche sui bar
        for i, (correct, incorrect) in enumerate(zip(correct_counts, incorrect_counts)):
            total = correct + incorrect
            if total > 0:
                # Etichetta per predizioni corrette
                if correct > 0:
                    plt.text(i, correct/2, f'{int(correct)}', ha='center', va='center',
                            fontweight='bold', color='white' if correct > total*0.1 else 'black')
                
                # Etichetta per predizioni scorrette
                if incorrect > 0:
                    plt.text(i, correct + incorrect/2, f'{int(incorrect)}', ha='center', va='center',
                            fontweight='bold', color='white' if incorrect > total*0.1 else 'black')
                
                # Etichetta totale sopra il bar
                plt.text(i, total + max(correct_counts + incorrect_counts) * 0.01, 
                        f'{int(total)}', ha='center', va='bottom', fontweight='bold')
        
        plt.xlabel('Predicted Class', fontsize=12)
        plt.ylabel('Number of Samples', fontsize=12)
        plt.title(f'Classification Distribution for True Class: {class_names[true_class]} ({task_name})', 
                 fontsize=14, fontweight='bold')
        
        plt.xticks(x, class_names, rotation=45, ha='right')
        plt.legend(loc='upper right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Salva l'istogramma
        filename = f'class_histogram_{task_name.lower()}_true_{class_names[true_class].replace("/", "_").replace("-", "_")}.png'
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Crea anche un istogramma riassuntivo con tutte le classi
    create_summary_histogram(y_true_valid, y_pred_valid, class_names, task_name, output_dir)

def create_summary_histogram(y_true, y_pred, class_names, task_name, output_dir):
    """Create a summary histogram showing overall classification performance."""
    # Calcola accuracy per classe
    class_accuracies = []
    class_totals = []
    
    for true_class in range(len(class_names)):
        class_mask = y_true == true_class
        if not np.any(class_mask):
            class_accuracies.append(0)
            class_totals.append(0)
            continue
            
        predictions_for_class = y_pred[class_mask]
        correct = np.sum(predictions_for_class == true_class)
        total = len(predictions_for_class)
        
        class_accuracies.append(correct / total if total > 0 else 0)
        class_totals.append(total)
    
    # Crea l'istogramma riassuntivo
    plt.figure(figsize=(12, 8))
    
    x = np.arange(len(class_names))
    colors = ['green' if acc > 0.5 else 'orange' if acc > 0.3 else 'red' for acc in class_accuracies]
    
    bars = plt.bar(x, class_accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    
    # Aggiungi etichette
    for i, (acc, total) in enumerate(zip(class_accuracies, class_totals)):
        if total > 0:
            plt.text(i, acc + 0.01, f'{acc:.3f}\n({total} samples)', 
                    ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.xlabel('True Class', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(f'Per-Class Accuracy Summary - {task_name}', fontsize=14, fontweight='bold')
    plt.xticks(x, class_names, rotation=45, ha='right')
    plt.ylim(0, 1.1)
    plt.grid(axis='y', alpha=0.3)
    
    # Aggiungi linea per accuracy media
    overall_acc = np.sum([acc * total for acc, total in zip(class_accuracies, class_totals)]) / np.sum(class_totals)
    if np.sum(class_totals) > 0:
        plt.axhline(y=overall_acc, color='blue', linestyle='--', linewidth=2, 
                   label=f'Overall Accuracy: {overall_acc:.3f}')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'accuracy_summary_{task_name.lower()}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def discover_checkpoints(ckpt_dir, save_type="bacc"):
    """Discover model checkpoints and features."""
    ckpt_dir = os.path.abspath(ckpt_dir)
    parent_dir = os.path.dirname(ckpt_dir)
    
    # Vision checkpoint
    vision_ckpt = None
    for path in [
        os.path.join(ckpt_dir, "vision_ckpt.pt"),
        os.path.join(parent_dir, "vision_ckpt.pt"),
        os.path.join(parent_dir, "full_training_model.pt")
    ]:
        if os.path.isfile(path):
            vision_ckpt = path
            break
    
    # VPT tokens
    vpt_tokens = []
    if os.path.isdir(ckpt_dir):
        for fn in sorted(os.listdir(ckpt_dir)):
            if fn.startswith("vpt_token") and fn.endswith(".pt") and save_type in fn:
                vpt_tokens.append(os.path.join(ckpt_dir, fn))
    
    # Text features
    text_feats = None
    for path in [
        os.path.join(ckpt_dir, f"text_features_{save_type}.pt"),
        os.path.join(ckpt_dir, "text_features.pt")
    ]:
        if os.path.isfile(path):
            text_feats = path
            break

    logit_scale = None
    for path in [
        os.path.join(ckpt_dir, f"logit_scale_{save_type}.pt"),
        os.path.join(ckpt_dir, "logit_scale.pt")
    ]:
        if os.path.isfile(path):
            logit_scale = path
            break
    
    return vision_ckpt, vpt_tokens, text_feats, logit_scale

def load_text_features(text_path, device):
    """Load text features from checkpoint."""
    if not text_path or not os.path.isfile(text_path):
        return None
    
    try:
        obj = torch.load(text_path, map_location=device)
        if isinstance(obj, dict) and "text_features" in obj:
            return obj["text_features"]
        elif torch.is_tensor(obj):
            return obj
    except Exception as e:
        print(f"Warning: failed to load text features: {e}")
    return None

def load_model(model_type, num_prompt, ckpt_dir, device, pe_vision_config="PE-Core-L14-336", 
               siglip2_repo_id="google/siglip2-base-patch16-224", save_type="bacc"):
    """Load and configure model."""
    vision_ckpt, vpt_tokens, text_path, logit_scale = discover_checkpoints(ckpt_dir, save_type)
    
    if model_type.startswith('PECore'):
        model = PECore_Vision(
            vision_cfg=PE_VISION_CONFIG[pe_vision_config],
            num_prompt=num_prompt if 'VPT' in model_type else 0
        )
        model.load_baseline(vision_ckpt, device)
        image_transform = get_image_transform(model.image_size)
        tokenizer = PETokenizer(32)
        
    elif model_type.startswith('Siglip2'):
        model = Siglip2Vision(
            AutoConfig.from_pretrained(siglip2_repo_id, cache_dir="./hf_models"),
            num_prompt=num_prompt if 'VPT' in model_type else 0
        )
        model.load_baseline(vision_ckpt, device)
        image_transform = T.Compose([
            T.Resize((model.image_size, model.image_size)),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        tokenizer = SigLip2Tokenizer(64)
    else:
        raise NotImplementedError(f"Model type {model_type} not implemented")
    
    # Load VPT tokens if needed
    if 'VPT' in model_type and vpt_tokens:
        num_tokens = 3 if 'single' in model_type else 1
        for token_path in vpt_tokens[:num_tokens]:
            try:
                model.load_VPT_token(token_path, device)
            except Exception as e:
                print(f"Warning: failed to load VPT token: {e}")

    if logit_scale:
        try:
            loaded_scale = torch.load(logit_scale, map_location=device)
            model.logit_scale = torch.nn.Parameter(loaded_scale)
            print(f"Loaded logit scale from checkpoint {loaded_scale}")
        except Exception as e:
            print(f"Warning: failed to load logit scale: {e}")
    
    return model, image_transform, tokenizer, text_path

@torch.inference_mode()
def validate(model, dataloader, device, use_tqdm=True):
    """Optimized validation function with improved accuracy calculation."""
    model.eval()
    
    # GPU counters for efficiency
    total_correct_top1 = torch.zeros(3, device=device, dtype=torch.long)
    total_correct_top2 = torch.zeros(3, device=device, dtype=torch.long)
    total_correct_1bin = torch.zeros(3, device=device, dtype=torch.long)
    total_samples = torch.zeros(3, device=device, dtype=torch.long)
    
    # Accumulate labels on GPU
    all_true_gpu = [[], [], []]
    all_pred_gpu = [[], [], []]
    
    iterator = tqdm(dataloader, desc="Validating") if use_tqdm else dataloader
    
    for batch_idx, (images, labels) in enumerate(iterator):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        with torch.amp.autocast('cuda', enabled=device.type=='cuda'):
            logits = model.forward(images)
        
        for task_idx, task_logits in enumerate(logits):
            if task_logits is None:
                continue
                
            task_labels = labels[:, task_idx]
            valid_mask = task_labels != -1
            if not torch.any(valid_mask):
                continue
            
            valid_labels = task_labels[valid_mask]
            valid_logits = task_logits[valid_mask]
            
            # Calcola top-k predictions in modo sicuro
            k = min(2, valid_logits.size(-1))
            topk_preds = valid_logits.topk(k, dim=-1).indices
            
            # Accumulate predictions
            all_true_gpu[task_idx].append(valid_labels)
            all_pred_gpu[task_idx].append(topk_preds[:, 0])
            
            # Compute accuracies
            top1_correct = (topk_preds[:, 0] == valid_labels)
            total_correct_top1[task_idx] += top1_correct.sum()
            total_samples[task_idx] += valid_labels.numel()
            
            # Top-2 accuracy solo se abbiamo almeno 2 classi
            if k >= 2:
                top2_correct = topk_preds.eq(valid_labels.unsqueeze(1)).any(dim=1)
                total_correct_top2[task_idx] += top2_correct.sum()
            else:
                total_correct_top2[task_idx] += top1_correct.sum()
            
            # 1-bin accuracy for age task only
            if task_idx == 0:
                onebin_correct = (topk_preds[:, 0] - valid_labels).abs() <= 1
                total_correct_1bin[task_idx] += onebin_correct.sum()
    
    # Convert to CPU once with safe division
    denom = total_samples.clamp_min(1).float()
    top1_accuracy = (total_correct_top1.float() / denom).cpu().tolist()
    top2_accuracy = (total_correct_top2.float() / denom).cpu().tolist()
    
    onebin_accuracy = [0.0, 0.0, 0.0]
    if total_samples[0] > 0:
        onebin_accuracy[0] = (total_correct_1bin[0].float() / denom[0]).cpu().item()
    
    # Convert predictions to CPU
    all_true_labels = []
    all_pred_labels = []
    for task_idx in range(3):
        if all_true_gpu[task_idx]:
            all_true_labels.append(torch.cat(all_true_gpu[task_idx]).cpu().tolist())
            all_pred_labels.append(torch.cat(all_pred_gpu[task_idx]).cpu().tolist())
        else:
            all_true_labels.append([])
            all_pred_labels.append([])
    
    # Print validation summary
    print(f"\nValidation Summary:")
    task_names = ["Age", "Gender", "Emotion"]
    for i, name in enumerate(task_names):
        if total_samples[i] > 0:
            print(f"{name}: {total_samples[i].item()} samples, "
                  f"Top-1: {top1_accuracy[i]:.4f}, Top-2: {top2_accuracy[i]:.4f}")
            if i == 0 and onebin_accuracy[0] > 0:
                print(f"  1-bin accuracy: {onebin_accuracy[0]:.4f}")
    
    return top1_accuracy, top2_accuracy, onebin_accuracy, all_true_labels, all_pred_labels

def create_final_summary_table(results_data, output_dir):
    """Create a final summary table with all datasets and their accuracies."""
    if not results_data:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare table data
    headers = ["Dataset", "Age Top-1", "Gender Top-1", "Emotion Top-1", "Average Top-1"]
    rows = []
    
    overall_totals = [0.0, 0.0, 0.0]  # To calculate grand averages
    
    for dataset_name, metrics in results_data.items():
        top1_acc = metrics['top1_acc']
        avg_acc = sum(top1_acc) / len(top1_acc) if top1_acc else 0.0
        
        # Add to overall totals
        for i, acc in enumerate(top1_acc):
            overall_totals[i] += acc
        
        rows.append([
            dataset_name,
            f"{top1_acc[0]:.4f}" if len(top1_acc) > 0 else "N/A",
            f"{top1_acc[1]:.4f}" if len(top1_acc) > 1 else "N/A", 
            f"{top1_acc[2]:.4f}" if len(top1_acc) > 2 else "N/A",
            f"{avg_acc:.4f}"
        ])
    
    # Add grand averages row
    num_datasets = len(results_data)
    if num_datasets > 0:
        grand_avg = sum(overall_totals) / (3 * num_datasets)
        rows.append([
            "GRAND AVERAGE",
            f"{overall_totals[0]/num_datasets:.4f}",
            f"{overall_totals[1]/num_datasets:.4f}",
            f"{overall_totals[2]/num_datasets:.4f}",
            f"{grand_avg:.4f}"
        ])
    
    # Format and save table
    table_str = format_table(headers, rows)
    
    with open(os.path.join(output_dir, "final_summary.txt"), "w") as f:
        f.write("MULTI-DATASET EVALUATION SUMMARY\n")
        f.write("="*50 + "\n\n")
        f.write(table_str)
        f.write("\n\nNotes:\n")
        f.write("- Top-1 accuracies shown for each task\n")
        f.write("- Average Top-1 is the mean across the three tasks\n")
        f.write("- Grand Average is the mean across all datasets\n")
    
    print("\n" + "="*50)
    print("MULTI-DATASET EVALUATION SUMMARY")
    print("="*50)
    print(table_str)
    
    return table_str

def evaluate_multiple_datasets(model, image_processor, device, args):
    """Evaluate on multiple datasets."""
    base_out = args.output_path
    os.makedirs(base_out, exist_ok=True)
    
    results_data = {}  # Store results for final summary
    
    for dataset_path in args.dataset_paths:
        if not os.path.exists(dataset_path):
            print(f"Warning: {dataset_path} does not exist, skipping")
            continue
        
        dataset_name = os.path.basename(os.path.normpath(dataset_path))
        output_dir = os.path.join(base_out, dataset_name)
        
        # Create temporary args for single dataset evaluation
        single_args = type(args)()
        for attr in dir(args):
            if not attr.startswith('_'):
                setattr(single_args, attr, getattr(args, attr))
        single_args.dataset_paths = [dataset_path]
        single_args.output_path = output_dir
        
        print(f"\n=== Evaluating {dataset_name} ===")
        
        # Evaluate single dataset and capture results
        os.makedirs(output_dir, exist_ok=True)
        
        dataset = BaseDataset(dataset_path, transform=image_processor, split="test", verbose=False)
        dataloader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, 
            num_workers=min(8, os.cpu_count() or 0), pin_memory=(device.type=='cuda')
        )
        
        print(f"Evaluating on dataset: {dataset_path}")
        print(f"Dataset size: {len(dataset)} samples")
        
        top1_acc, top2_acc, onebin_acc, all_true, all_pred = validate(
            model, dataloader, device, not args.no_tqdm
        )
        
        # Store results for final summary
        results_data[dataset_name] = {
            'top1_acc': top1_acc,
            'top2_acc': top2_acc,
            'onebin_acc': onebin_acc
        }
        
        # Compute metrics and save results for this dataset
        prf_task = compute_prf_metrics(all_true, all_pred)
        write_results(output_dir, top1_acc, top2_acc, onebin_acc, prf_task)
        
        # Save confusion matrices
        task_names = ["Age", "Gender", "Emotion"]
        cm_dir = os.path.join(output_dir, "confusion_matrices")
        hist_dir = os.path.join(output_dir, "class_histograms")
        for i, (true, pred) in enumerate(zip(all_true, all_pred)):
            if true and pred:
                save_confusion_matrix(true, pred, CLASSES[i], task_names[i], cm_dir)
                save_class_histograms(true, pred, CLASSES[i], task_names[i], hist_dir)

    # Create final summary table
    if results_data:
        create_final_summary_table(results_data, base_out)
    
    print(f"\nCompleted evaluation on {len(results_data)} datasets")

def evaluate_single_dataset(model, image_processor, device, args):
    """Evaluate on single dataset."""
    os.makedirs(args.output_path, exist_ok=True)
    
    dataset_path = args.dataset_paths[0] if args.dataset_paths else args.dataset_path
    dataset = BaseDataset(dataset_path, transform=image_processor, split="test", verbose=False)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, 
        num_workers=min(8, os.cpu_count() or 0), pin_memory=(device.type=='cuda')
    )
    
    print(f"Evaluating on dataset: {dataset_path}")
    print(f"Dataset size: {len(dataset)} samples")
    
    top1_acc, top2_acc, onebin_acc, all_true, all_pred = validate(
        model, dataloader, device, not args.no_tqdm
    )
    
    # Compute metrics and save results
    prf_task = compute_prf_metrics(all_true, all_pred)
    write_results(args.output_path, top1_acc, top2_acc, onebin_acc, prf_task)
    
    # Save confusion matrices
    task_names = ["Age", "Gender", "Emotion"]
    cm_dir = os.path.join(args.output_path, "confusion_matrices")
    hist_dir = os.path.join(args.output_path, "class_histograms")
    for i, (true, pred) in enumerate(zip(all_true, all_pred)):
        if true and pred:
            save_confusion_matrix(true, pred, CLASSES[i], task_names[i], cm_dir)
            save_class_histograms(true, pred, CLASSES[i], task_names[i], hist_dir)

def main(args):
    """Main evaluation function."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    
    # Load model
    model, image_processor, tokenizer, text_path = load_model(
        args.model_type, args.num_prompt, args.ckpt_dir, device,
        args.pe_vision_config, args.siglip2_repo_id, args.save_to_load
    )
    model.to(device)
    
    # Load text features
    text_features = load_text_features(text_path, device)
    if hasattr(model, "text_features") and text_features is not None:
        model.text_features = text_features
    
    print(f"Model loaded on {device} with {sum(p.numel() for p in model.parameters())} parameters")
    if args.output_path is None:
        args.output_path = args.ckpt_dir.replace("TRAIN", "TEST").split("ckpt")[0]
        print(f"Output path not specified, using {args.output_path}")
    
    # Handle single or multiple datasets
    if len(args.dataset_paths) == 1 and not os.path.isdir(args.dataset_paths[0]):
        # Single dataset file
        evaluate_single_dataset(model, image_processor, device, args)
    else:
        # Multiple datasets or directories
        evaluate_multiple_datasets(model, image_processor, device, args)

def parse_args():
    """Parse command line arguments."""
    parser = ArgumentParser()
    parser.add_argument('--model_type', type=str, default='PECoreBase',
                        choices=['PECoreBase', 'Siglip2Base', 'PECoreVPT', 'Siglip2VPT',
                                'PECoreSoftCPT', 'Siglip2SoftCPT', 'PECoreVPT_single', 'Siglip2VPT_single'])
    parser.add_argument('--dataset_paths', type=str, nargs='+', required=True, 
                        help='Dataset path(s) - single path or multiple paths')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--output_path', type=str, default=None, 
                        help='Output path. If None, auto-generated from ckpt_dir (TRAIN->TEST)')
    parser.add_argument('--no_tqdm', action='store_true', help='Disable progress bar')
    parser.add_argument('--num_prompt', type=int, default=0)
    parser.add_argument('--ckpt_dir', type=str, required=True)
    parser.add_argument('--pe_vision_config', type=str, default="PE-Core-L14-336")
    parser.add_argument('--siglip2_repo_id', type=str, default="google/siglip2-large-patch16-384")
    parser.add_argument('--save_to_load', type=str, choices=['bacc', 'bval'], default='bval')
    
    return parser.parse_args()

if __name__ == "__main__":
    main(parse_args())