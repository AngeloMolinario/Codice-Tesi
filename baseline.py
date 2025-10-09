import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms as T
from transformers import AutoConfig
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np
import os
from matplotlib import pyplot as plt

from core.vision_encoder.pe import CLIP
from core.vision_encoder.config import *
from core.vision_encoder.transforms import get_image_transform
from dataset.dataset import BaseDataset
from wrappers.tokenizer import *

CLASSES = [
    ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"],
    ["male", "female"],
    ["surprise", "fear", "disgust", "happy", "sad", "angry", "neutral"]
]

TEXT_PROMPTS = [
    [
        "a photo of a person between 0 and 2 years old",
        "a photo of a person between 3 and 9 years old",
        "a photo of a person between 10 and 19 years old",
        "a photo of a person between 20 and 29 years old",
        "a photo of a person between 30 and 39 years old",
        "a photo of a person between 40 and 49 years old",
        "a photo of a person between 50 and 59 years old",
        "a photo of a person between 60 and 69 years old",
        "a photo of a person 70 years old or older"
    ],
    [
        "a photo of a male person",
        "a photo of a female person"
    ],
    [
        "a photo of a surprised person",
        "a photo of a fearful person",
        "a photo of a disgusted person",
        "a photo of a happy person",
        "a photo of a sad person",
        "a photo of an angry person",
        "a photo of a neutral person"
    ]
]


def compute_GFLOPs(model, device='cpu'):
    """Compute GFLOPs of the model (placeholder function)."""
    
    model.eval()
    # Configure profiler activities
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device == "cuda" and torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    try:
        total_flops = 0
        dummy_input = torch.randn(1, 3, model.image_size, model.image_size)
        with torch.profiler.profile(activities=activities,record_shapes=True, with_flops=True) as prof:
            with torch.no_grad():
                _ = model(dummy_input.to(device))
    # Sum all FLOPs from profiler events
        for event in prof.events():
            if hasattr(event, 'flops') and event.flops > 0:
                total_flops += event.flops
        
        return total_flops / 1e9
        
    except Exception as e:
        print(f"Profiling failed: {str(e)}. Returning 0.0")
        return 0.0
    

def compute_balanced_accuracy(all_true_labels, all_pred_labels):
    """Compute balanced accuracy per task."""
    balanced_acc = [0.0, 0.0, 0.0]
    
    for task_idx in range(3):
        y_true, y_pred = all_true_labels[task_idx], all_pred_labels[task_idx]
        if y_true and y_pred:
            try:
                # Assicurati che i labels siano nell'intervallo corretto
                unique_labels = sorted(set(y_true + y_pred))
                valid_labels = [l for l in unique_labels if 0 <= l < len(CLASSES[task_idx])]
                
                if valid_labels:
                    balanced_acc[task_idx] = balanced_accuracy_score(y_true, y_pred)
                else:
                    balanced_acc[task_idx] = 0.0
            except Exception as e:
                print(f"Warning: Error computing balanced accuracy for task {task_idx}: {e}")
    
    return balanced_acc

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

def write_results(output_dir, top1_acc, balanced_acc=None, c_index=None):
    """Write results to file."""
    os.makedirs(output_dir, exist_ok=True)

    tasks = ['age', 'gender', 'emotion']
    rows = []

    valid_tasks = []  # Track valid tasks for average calculation

    for i in range(len(top1_acc)):
        # Use balanced accuracy instead of precision, recall, F1
        if balanced_acc is not None and len(balanced_acc) > i:
            bacc_str = f"{balanced_acc[i]:.4f}" if balanced_acc[i] > 0 else "-"
        else:
            bacc_str = "N/A"

        # Add C-index
        if c_index is not None and len(c_index) > i:
            c_index_str = f"{c_index[i]:.4f}" if c_index[i] > 0 else "-"
        else:
            c_index_str = "N/A"

        if top1_acc[i] >= 0:  # Valid task with data
            top1_str = f"{top1_acc[i]:.4f}"
            valid_tasks.append(i)
        else:
            top1_str = "-"

        rows.append([
            f"Task {tasks[i]}",
            top1_str,
            bacc_str,
            c_index_str
        ])

    # Calculate average only for valid tasks
    if valid_tasks:
        valid_accs = [top1_acc[i] for i in valid_tasks]
        avg_top1 = sum(valid_accs) / len(valid_accs)
        avg_str = f"{avg_top1:.4f}"

        # Calculate average balanced accuracy for valid tasks
        if balanced_acc is not None:
            valid_bacc = [balanced_acc[i] for i in valid_tasks if balanced_acc[i] > 0]
            avg_bacc = sum(valid_bacc) / len(valid_bacc) if valid_bacc else 0.0
            avg_bacc_str = f"{avg_bacc:.4f}" if avg_bacc > 0 else "-"
        else:
            avg_bacc_str = "N/A"

        # Calculate average C-index for valid tasks
        if c_index is not None:
            valid_c_index = [c_index[i] for i in valid_tasks if c_index[i] > 0]
            avg_c_index = sum(valid_c_index) / len(valid_c_index) if valid_c_index else 0.0
            avg_c_index_str = f"{avg_c_index:.4f}" if avg_c_index > 0 else "-"
        else:
            avg_c_index_str = "N/A"
    else:
        avg_str = "-"
        avg_bacc_str = "-"
        avg_c_index_str = "-"

    rows.append(["Average", avg_str, avg_bacc_str, avg_c_index_str])

    headers = ["Task", "Top-1", "Balanced Accuracy", "C-Index"]
    table_str = format_table(headers, rows)

    with open(os.path.join(output_dir, "results.txt"), "w") as f:
        f.write(table_str)

    print(table_str)
    return table_str


# TODO: modify this for baseline
def write_model_info(output_dir, params_count, GFLOPs):
    """Write model information including parameters count and GFLOPs to file."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare rows for the model info table
    rows = []
    
    # Add parameter counts
    rows.append(["Logit Scale", f"{params_count['logit_scale']:,}"])
    rows.append(["Vision Transformer", f"{params_count['vision_transformer']:,}"])
    rows.append(["Text Features", f"{params_count['text_features']:,}"])
    rows.append(["VPT Tokens", f"{params_count['vpt_tokens']:,}"])
    rows.append(["Total Trainable", f"{params_count['total_trainable']:,}"])
    rows.append(["Total All", f"{params_count['total_all']:,}"])
    
    # Add GFLOPs
    rows.append(["GFLOPs", f"{GFLOPs:.2f}"])
    
    headers = ["Component", "Count/Value"]
    table_str = format_table(headers, rows)
    
    # Save to file
    with open(os.path.join(output_dir, "model_info.txt"), "w") as f:
        f.write("MODEL INFORMATION\n")
        f.write("="*30 + "\n\n")
        f.write(table_str)
        f.write("\n\nNotes:\n")
        f.write("- Parameter counts include all parameters in each component\n")
        f.write("- Total Trainable excludes text_features (registered buffer)\n")
        f.write("- GFLOPs computed with single forward pass\n")
    
    print("\nModel Information:")
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
        
def load_text_features(text_path, device):
    """Load text features from checkpoint."""
    if not text_path or not os.path.isfile(text_path):
        return None
    
    try:
        obj = torch.load(text_path, map_location=device)
        if isinstance(obj, dict) and "text_features" in obj:
            # If saved as state_dict than load the tensor
            return obj["text_features"]
        elif torch.is_tensor(obj):
            return obj
    except Exception as e:
        print(f"Warning: failed to load text features: {e}")
    return None

def predict(model, images, text_features):
    """Predict logits for each task."""
    with torch.no_grad():
        image_features = model.encode_image(images)

        logits = model.logit_scale.exp() * (image_features @ text_features.T)

    return torch.split(logits, [9, 2, 7], dim=-1)

@torch.inference_mode()
def validate(model, dataloader, text_features, device, use_tqdm=True):
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
                
        logits = predict(model, images, text_features)
        
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

def evaluate_multiple_datasets(model, image_processor, text_features, device, args):
    """Evaluate on multiple datasets."""
    base_out = args.output_path
    os.makedirs(base_out, exist_ok=True)
    
    results_data = {}  # Store results for final summary
    '''
    GFLOPs = compute_GFLOPs(model, device)
    params_count = model.get_parameters_count()
    write_model_info(base_out, params_count, GFLOPs)
    '''

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
            num_workers=8
        )
        
        print(f"Evaluating on dataset: {dataset_path}")
        print(f"Dataset size: {len(dataset)} samples")
        
        top1_acc, all_true, all_pred = validate(
            model, dataloader, text_features, device, not args.no_tqdm
        )
        
        # Store results for final summary
        results_data[dataset_name] = {
            'top1_acc': top1_acc
        }

        # Compute metrics and save results for this dataset
        balanced_acc = compute_balanced_accuracy(all_true, all_pred)
        c_index = None#compute_c_index_per_task(all_true, all_pred)
        write_results(output_dir, top1_acc, balanced_acc, c_index)

        
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
    
    model = CLIP.from_config("PE-Core-L14-336", pretrained=True)
    image_processor = get_image_transform(336)
    tokenizer = PETokenizer(32)

    model.to(device)

    # Load text features
    text_features = []
    for text in TEXT_PROMPTS:
        for prompt in text:
            text_features.append(tokenizer(prompt))

    text_features = torch.cat(text_features).to(device)
    print(f"Text features shape (before encoding): {text_features.shape}")
    text_features = model.encode_text(text_features)
    print(f"Text features shape: {text_features.shape}")


    print(f"Model loaded on {device} with {sum(p.numel() for p in model.parameters())} parameters")
    if args.output_path is None:
        args.output_path = args.ckpt_dir.replace("TRAIN", "TEST").split("ckpt")[0]
        print(f"Output path not specified, using {args.output_path}")
    
    # Use the multiple datasets evaluation logic for all cases
    evaluate_multiple_datasets(model, image_processor, text_features, device, args)

def parse_args():
    """Parse command line arguments."""
    parser = ArgumentParser()
    parser.add_argument('--model_type', type=str, default='PECore',
                        choices=['PECore', 'Siglip2'])
    parser.add_argument('--dataset_paths', type=str, nargs='+', required=True, 
                        help='Dataset path(s) - single path or multiple paths')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--output_path', type=str, default=None, 
                        help='Output path. If None, auto-generated from ckpt_dir (TRAIN->TEST)')
    parser.add_argument('--no_tqdm', action='store_true', help='Disable progress bar')
    
    return parser.parse_args()

if __name__ == "__main__":
    main(parse_args())
