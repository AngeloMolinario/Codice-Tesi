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

def write_results(output_dir, top1_acc, prf_task=None):
    """Write results to file."""
    os.makedirs(output_dir, exist_ok=True)
    
    tasks = ['age', 'gender', 'emotion']
    rows = []
    
    valid_tasks = []  # Track valid tasks for average calculation
    
    for i in range(len(top1_acc)):
        prec = recall = f1 = "N/A"
        if prf_task:
            prec = f"{prf_task['precision'][i]:.4f}" if prf_task['precision'][i] > 0 else "-"
            recall = f"{prf_task['recall'][i]:.4f}" if prf_task['recall'][i] > 0 else "-"
            f1 = f"{prf_task['f1'][i]:.4f}" if prf_task['f1'][i] > 0 else "-"
        
        if top1_acc[i] >= 0:  # Valid task with data
            top1_str = f"{top1_acc[i]:.4f}"
            valid_tasks.append(i)
        else:
            top1_str = "-"
        
        rows.append([
            f"Task {tasks[i]}",
            top1_str,
            prec, recall, f1
        ])
    
    # Calculate average only for valid tasks
    if valid_tasks:
        valid_accs = [top1_acc[i] for i in valid_tasks]
        avg_top1 = sum(valid_accs) / len(valid_accs)
        avg_str = f"{avg_top1:.4f}"
    else:
        avg_str = "-"
    
    rows.append(["Average", avg_str, "-", "-", "-"])
    
    headers = ["Task", "Top-1", "Precision", "Recall", "F1"]
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
    
    # Crea matrice con solo valori assoluti
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title(f'Confusion Matrix - {task_name}', fontsize=14, fontweight='bold')
    plt.colorbar(label='Count')
    
    tick_marks = np.arange(len(valid_class_names))
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
    plt.savefig(os.path.join(output_dir, f'confusion_matrix_{task_name.lower()}.png'), 
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
    total_samples = torch.zeros(3, device=device, dtype=torch.long)
    
    # Accumulate labels on GPU
    all_true_gpu = [[], [], []]
    all_pred_gpu = [[], [], []]
    
    iterator = tqdm(dataloader, desc="Validating") if use_tqdm else dataloader
    
    for batch_idx, (images, labels) in enumerate(iterator):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
                
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
            
            # Calculate only top-1 predictions using argmax
            preds = torch.argmax(valid_logits, dim=-1)
            
            # Accumulate predictions
            all_true_gpu[task_idx].append(valid_labels)
            all_pred_gpu[task_idx].append(preds)
            
            # Compute top-1 accuracy
            top1_correct = (preds == valid_labels)
            total_correct_top1[task_idx] += top1_correct.sum()
            total_samples[task_idx] += valid_labels.numel()
            
    # Convert to CPU once with safe division
    top1_accuracy = torch.zeros(3, dtype=torch.float)
    for i in range(3):
        if total_samples[i] > 0:
            top1_accuracy[i] = (total_correct_top1[i].float() / total_samples[i].float()).cpu()
        else:
            top1_accuracy[i] = torch.tensor(-1.0)  # Use -1 to indicate no data for this task
    
    top1_accuracy = top1_accuracy.tolist()
    
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
                  f"Top-1: {top1_accuracy[i]:.4f}")
        else:
            print(f"{name}: No valid samples")
    
    return top1_accuracy, all_true_labels, all_pred_labels

def create_final_summary_table(results_data, output_dir):
    """Create a final summary table with all datasets and their accuracies."""
    if not results_data:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare table data
    headers = ["Dataset", "Age Top-1", "Gender Top-1", "Emotion Top-1", "Average Top-1"]
    rows = []
    
    overall_totals = [0.0, 0.0, 0.0]  # To calculate grand averages
    overall_valid_counts = [0, 0, 0]  # To track valid tasks
    
    for dataset_name, metrics in results_data.items():
        top1_acc = metrics['top1_acc']
        
        # Calculate average only for valid tasks (>= 0)
        valid_accs = [acc for acc in top1_acc if acc >= 0]
        avg_acc = sum(valid_accs) / len(valid_accs) if valid_accs else -1.0
        
        # Add to overall totals only for valid tasks
        for i, acc in enumerate(top1_acc):
            if acc >= 0:
                overall_totals[i] += acc
                overall_valid_counts[i] += 1
        
        rows.append([
            dataset_name,
            f"{top1_acc[0]:.4f}" if top1_acc[0] >= 0 else "-",
            f"{top1_acc[1]:.4f}" if top1_acc[1] >= 0 else "-", 
            f"{top1_acc[2]:.4f}" if top1_acc[2] >= 0 else "-",
            f"{avg_acc:.4f}" if avg_acc >= 0 else "-"
        ])
    # Add grand averages row
    grand_avg_values = []
    grand_avg_sum = 0
    valid_task_count = 0
    
    for i in range(3):
        if overall_valid_counts[i] > 0:
            avg = overall_totals[i] / overall_valid_counts[i]
            grand_avg_values.append(f"{avg:.4f}")
            grand_avg_sum += avg
            valid_task_count += 1
        else:
            grand_avg_values.append("-")
    
    grand_avg = grand_avg_sum / valid_task_count if valid_task_count > 0 else -1.0
    
    rows.append([
        "GRAND AVERAGE",
        grand_avg_values[0],
        grand_avg_values[1],
        grand_avg_values[2],
        f"{grand_avg:.4f}" if grand_avg >= 0 else "-"
    ])
    
    # Format and save table
    table_str = format_table(headers, rows)
    
    with open(os.path.join(output_dir, "final_summary.txt"), "w") as f:
        f.write("MULTI-DATASET EVALUATION SUMMARY\n")
        f.write("="*50 + "\n\n")
        f.write(table_str)
        f.write("\n\nNotes:\n")
        f.write("- Top-1 accuracies shown for each task\n")
        f.write("- '-' indicates no valid data for the task\n")
        f.write("- Average Top-1 is the mean across valid tasks only\n")
        f.write("- Grand Average is the mean across all datasets with valid data\n")
    
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
        
        print(f"\n=== Evaluating {dataset_name} ===")
        
        # Evaluate dataset
        os.makedirs(output_dir, exist_ok=True)
        
        dataset = BaseDataset(dataset_path, transform=image_processor, split="test", verbose=False)
        dataloader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, 
            num_workers=3
        )
        
        print(f"Evaluating on dataset: {dataset_path}")
        print(f"Dataset size: {len(dataset)} samples")
        
        top1_acc, all_true, all_pred = validate(
            model, dataloader, device, not args.no_tqdm
        )
        
        # Store results for final summary
        results_data[dataset_name] = {
            'top1_acc': top1_acc
        }
        
        # Compute metrics and save results for this dataset
        prf_task = compute_prf_metrics(all_true, all_pred)
        write_results(output_dir, top1_acc, prf_task)
        
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
    
    # Use the multiple datasets evaluation logic for all cases
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
