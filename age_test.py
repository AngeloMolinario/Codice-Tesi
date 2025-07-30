import os
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import json
import argparse

import matplotlib.pyplot as plt

from dataset.dataset import BaseDataset
from wrappers.PerceptionEncoder.pe import PECore
from core.vision_encoder.transforms import *

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Age Classification Testing')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of workers for data loading (default: 0)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for data loading (default: 128)')
    parser.add_argument('--dataset_name', type=str, default='UTKFace',
                        help='Dataset name (default: UTKFace)')
    parser.add_argument('--num_prompt', type=int, default=0,
                        help='Number of context prompt tokens to be prepended to the image patches (default: 0)')
    
    return parser.parse_args()

def create_model(num_prompt=0):
    """
    Create and return the PEcore model for age classification.
    This function should be implemented to return your specific model.
    """
    # TODO: Implement model creation
    model = PECore.from_config("PE-Core-B16-224", pretrained=True, num_prompt=num_prompt)

    return model

def create_dataset(dataset_name="UTKFace", transform=None):
    """
    Create and return the BaseDataset for age classification testing.
    This function should be implemented to return your specific dataset.
    """
    # TODO: Implement dataset creation
    dataset = BaseDataset(
        root=f"./datasets_with_standard_labels/{dataset_name}",  # Update this path
        transform=transform,  # Add your transforms if needed
        split="test"
    )
    return dataset

def create_output_directories():
    """Create output directories with incremental numbering if they don't exist."""
    base_name = "MODEL_TEST/AGE"
    
    # Find the next available directory number
    counter = 0
    while True:
        if counter == 0:
            base_dir = base_name
        else:
            base_dir = f"{base_name}{counter}"
        
        if not os.path.exists(base_dir):
            break
        counter += 1
    
    # Create the directories
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(os.path.join(base_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "results"), exist_ok=True)
    
    print(f"Created output directory: {base_dir}")
    return base_dir

def get_age_text_prompts():
    """
    Return text prompts for age classification based on the AGE_GROUP enum.
    """
    return [
        "A photo of a person 0-2 years old",       # Class 0
        "A photo of a person 3-9 years old",       # Class 1
        "A photo of a person 10-19 years old",     # Class 2
        "A photo of a person 20-29 years old",     # Class 3
        "A photo of a person 30-39 years old",     # Class 4
        "A photo of a person 40-49 years old",     # Class 5
        "A photo of a person 50-59 years old",     # Class 6
        "A photo of a person 60-69 years old",     # Class 7
        "A photo of a person more than 70 years old"  # Class 8 (id=75)
    ]

def get_age_categories():
    """
    Return age category labels based on the AGE_GROUP enum.
    """
    return [
        "0-2",           # Class 0
        "3-9",           # Class 1
        "10-19",         # Class 2
        "20-29",         # Class 3
        "30-39",         # Class 4
        "40-49",         # Class 5
        "50-59",         # Class 6
        "60-69",         # Class 7
        "more than 70"   # Class 8
    ]

def tokenize_text_prompts(text_prompts, tokenizer):
    """
    Tokenize the text prompts for the model.
    You'll need to implement this based on your tokenizer.
    """
    # TODO: Implement text tokenization based on your model's tokenizer
    # This should return tokenized text that can be fed to the model
    return tokenizer(text_prompts)
    

def process_model_output(image_features, text_features, logit_scale):
    """
    Process the model output to compute predictions.
    Computes cosine similarity, applies scale factor and sigmoid to get probabilities.
    """
    # Compute cosine similarity between image and text features
    # image_features: [batch_size, feature_dim]
    # text_features: [num_classes, feature_dim]
    
    # Compute cosine similarity
    cosine_sim = torch.matmul(image_features, text_features.t())
    
    # Apply scale factor
    scaled_logits = cosine_sim * logit_scale
    
    # Apply sigmoid to get probabilities
    probabilities = torch.sigmoid(scaled_logits)
    
    # Get predictions (argmax along the text dimension)
    predictions = torch.argmax(probabilities, dim=1)
    
    return predictions, probabilities

def get_model_info(model):
    """Extract model structure and configuration information."""
    model_info = {
        'model_class': model.__class__.__name__,
        'model_type': type(model).__name__,
        'model_module': model.__class__.__module__,
    }
    
    # Try to get specific PE-Core configuration
    try:
        if hasattr(model, 'config'):
            model_info['config'] = str(model.config)
        if hasattr(model, 'image_size'):
            model_info['image_size'] = model.image_size
        if hasattr(model, 'num_prompt'):
            model_info['num_prompt'] = model.num_prompt
        if hasattr(model, 'vision_encoder'):
            model_info['vision_encoder'] = type(model.vision_encoder).__name__
        if hasattr(model, 'text_model'):
            model_info['text_model'] = type(model.text_model).__name__
        if hasattr(model, 'logit_scale'):
            model_info['logit_scale'] = float(model.logit_scale.item()) if hasattr(model.logit_scale, 'item') else str(model.logit_scale)
    except Exception as e:
        model_info['error_getting_config'] = str(e)
    
    return model_info

def test_age_classification(num_prompt=0, num_workers=0, batch_size=128, dataset_name="UTKFace"):
    """Test the age classification model and generate comprehensive metrics."""
    
    # Create output directories
    output_dir = create_output_directories()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Using {num_workers} workers for data loading")
    print(f"Batch size: {batch_size}")
    
    # Initialize model and dataset
    model = create_model(num_prompt=num_prompt)
    model = model.to(device)  # Move model to device
    
    # Get model information for reporting
    model_info = get_model_info(model)
    
    # Save model structure information immediately
    model_structure_text = f"""
MODEL STRUCTURE AND INITIALIZATION REPORT
{'='*50}

Model Configuration:
- Model Class: {model_info.get('model_class', 'Unknown')}
- Model Type: {model_info.get('model_type', 'Unknown')}
- Model Module: {model_info.get('model_module', 'Unknown')}
- Image Size: {model_info.get('image_size', 'Unknown')}
- Number of Prompts: {model_info.get('num_prompt', 'Unknown')}
- Vision Encoder: {model_info.get('vision_encoder', 'Unknown')}
- Text Model: {model_info.get('text_model', 'Unknown')}
- Logit Scale: {model_info.get('logit_scale', 'Unknown')}

Training Configuration:
- Dataset: {dataset_name}
- Batch Size: {batch_size}
- Number of Workers: {num_workers}

Full Model Configuration:
{model_info.get('config', 'Configuration not available')}

Device Used: {device}

Model Parameters:
- Total Parameters: {sum(p.numel() for p in model.parameters()):,}
- Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}

Model Architecture Summary:
{str(model)}
"""
    
    with open(os.path.join(output_dir, "results", "model_structure.txt"), 'w') as f:
        f.write(model_structure_text)
    
    print("Model structure saved to model_structure.txt")
    
    image_transform = get_image_transform(model.image_size)  # Adjust based on your model
    tokenizer = get_text_tokenizer(model.text_model.context_length)
    dataset = create_dataset(dataset_name=dataset_name, transform=image_transform)

    # Create data loader with configurable num_workers
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    print(f"Dataset loaded: {len(dataset)} samples")
    print(f"Number of batches: {len(dataloader)}")
    
    # Set model to evaluation mode
    model.eval()
    
    # Get age text prompts and categories
    age_text_prompts = get_age_text_prompts()
    age_categories = get_age_categories()
    
    # Tokenize text prompts (you'll need to implement this based on your tokenizer)
    tokenized_texts = tokenize_text_prompts(age_text_prompts, tokenizer)  # Adjust based on your model
    
    # Encode text prompts once at the beginning
    with torch.no_grad():
        # Move tokenized texts to device if they're tensors
        if torch.is_tensor(tokenized_texts):
            tokenized_texts = tokenized_texts.to(device)
        text_features = model.encode_text(tokenized_texts, normalize=True)
    
    all_predictions = []
    all_true_labels = []
    all_probabilities = []
    
    print("Starting age classification testing...")
    print(f"Age categories: {age_categories}")
    print(f"Text prompts: {age_text_prompts}")
    
    # Perform inference
    with torch.no_grad():
        logit_scale = model.logit_scale
        for batch_idx, (images, labels) in enumerate(dataloader):
            # Move data to device
            images = images.to(device)
            # Extract age group labels directly from the label dictionary (already grouped)
            age_labels = labels['age'].to(device)
            
            # Get image features
            image_features = model.encode_image(images, normalize=True)
            
            # Process model output to get predictions
            predictions, probabilities = process_model_output(image_features, text_features, logit_scale)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_true_labels.extend(age_labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            
            if batch_idx % 10 == 0:
                print(f"Processed batch {batch_idx}/{len(dataloader)}")
                # Debug: Print some age group labels for the first batch
                if batch_idx == 0:
                    print(f"Sample age groups (first 10):")
                    for i in range(min(10, len(age_labels))):
                        group_id = age_labels[i].item()
                        if 0 <= group_id < len(age_categories):
                            print(f"  Group {group_id} ({age_categories[group_id]})")
                        else:
                            print(f"  Group {group_id} (Unknown/Missing)")
    
    # Convert to numpy arrays
    y_true = np.array(all_true_labels)
    y_pred = np.array(all_predictions)
    y_prob = np.array(all_probabilities)
    
    # Filter out missing values (-1) for evaluation
    valid_mask = y_true != -1
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]
    y_prob_valid = y_prob[valid_mask]
    
    print(f"\nFiltered out {np.sum(~valid_mask)} samples with missing age labels")
    print(f"Evaluating on {len(y_true_valid)} valid samples")
    
    # Get unique classes present in the valid data
    unique_classes = np.unique(y_true_valid)
    num_classes = len(unique_classes)
    
    # Calculate metrics on valid data only
    accuracy = accuracy_score(y_true_valid, y_pred_valid)
    precision = precision_score(y_true_valid, y_pred_valid, average='weighted', zero_division=0)
    recall = recall_score(y_true_valid, y_pred_valid, average='weighted', zero_division=0)
    f1 = f1_score(y_true_valid, y_pred_valid, average='weighted', zero_division=0)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_true_valid, y_pred_valid, labels=unique_classes)
    
    # Print metrics
    print("\n" + "="*50)
    print("AGE CLASSIFICATION TEST RESULTS")
    print("="*50)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    
    # Debug: Print unique classes and available categories
    print(f"\nUnique classes found in data: {unique_classes}")
    print(f"Available age categories: {len(age_categories)} categories (indices 0-{len(age_categories)-1})")
    print(f"Age categories: {age_categories}")
    
    # Create safe target names for classification report
    safe_target_names = []
    for class_id in unique_classes:
        if 0 <= class_id < len(age_categories):
            safe_target_names.append(age_categories[class_id])
        else:
            safe_target_names.append(f"Unknown_Class_{class_id}")
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_true_valid, y_pred_valid, target_names=safe_target_names, zero_division=0))
    
    # Save metrics to file
    metrics = {
        'model_info': model_info,
        'training_config': {
            'dataset_name': dataset_name,
            'batch_size': batch_size,
            'num_workers': num_workers,
            'device': str(device)
        },
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist(),
        'total_samples': len(y_true),
        'valid_samples': len(y_true_valid),
        'missing_samples': int(np.sum(~valid_mask)),
        'correct_predictions': int(np.sum(y_true_valid == y_pred_valid)),
        'age_categories': safe_target_names,
        'text_prompts': [age_text_prompts[i] if 0 <= i < len(age_text_prompts) else f"No prompt for class {i}" for i in unique_classes],
        'unique_classes': unique_classes.tolist(),
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
    }
    
    with open(os.path.join(output_dir, "results", "age_metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Plot and save confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=safe_target_names, 
                yticklabels=safe_target_names)
    plt.title('Age Classification Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plots", "confusion_matrix.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot metrics bar chart
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    metrics_values = [accuracy, precision, recall, f1]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics_names, metrics_values, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
    plt.title('Age Classification Metrics')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, value in zip(bars, metrics_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plots", "metrics_bar_chart.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot prediction distribution (using valid data only)
    unique_pred, counts = np.unique(y_pred_valid, return_counts=True)
    
    plt.figure(figsize=(12, 8))
    pred_labels = [age_categories[i] if 0 <= i < len(age_categories) else f"Unknown_{i}" for i in unique_pred]
    plt.pie(counts, labels=pred_labels, autopct='%1.1f%%', startangle=90)
    plt.title('Distribution of Age Predictions')
    plt.savefig(os.path.join(output_dir, "plots", "prediction_distribution.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot per-class accuracy (using valid data only)
    per_class_accuracy = []
    for class_label in unique_classes:
        class_mask = y_true_valid == class_label
        if np.sum(class_mask) > 0:  # Avoid division by zero
            class_accuracy = accuracy_score(y_true_valid[class_mask], y_pred_valid[class_mask])
            per_class_accuracy.append(class_accuracy)
        else:
            per_class_accuracy.append(0.0)
    
    plt.figure(figsize=(14, 6))
    bars = plt.bar(safe_target_names, per_class_accuracy, 
                   color='lightblue', edgecolor='navy', alpha=0.7)
    plt.title('Per-Class Accuracy for Age Classification')
    plt.ylabel('Accuracy')
    plt.xlabel('Age Category')
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, value in zip(bars, per_class_accuracy):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plots", "per_class_accuracy.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot confidence distribution (using valid data only)
    max_probs = np.max(y_prob_valid, axis=1)
    plt.figure(figsize=(10, 6))
    plt.hist(max_probs, bins=50, alpha=0.7, color='lightblue', edgecolor='navy')
    plt.title('Distribution of Prediction Confidence')
    plt.xlabel('Max Probability')
    plt.ylabel('Frequency')
    plt.axvline(np.mean(max_probs), color='red', linestyle='--', label=f'Mean: {np.mean(max_probs):.3f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plots", "confidence_distribution.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save detailed results with model information
    results_text = f"""
AGE CLASSIFICATION TEST RESULTS
{'='*50}

MODEL INFORMATION:
- Model Class: {model_info.get('model_class', 'Unknown')}
- Model Type: {model_info.get('model_type', 'Unknown')}
- Number of Prompts: {model_info.get('num_prompt', 'Unknown')}
- Image Size: {model_info.get('image_size', 'Unknown')}
- Vision Encoder: {model_info.get('vision_encoder', 'Unknown')}
- Text Model: {model_info.get('text_model', 'Unknown')}
- Device Used: {device}
- Total Parameters: {sum(p.numel() for p in model.parameters()):,}

Text Prompts Used:
"""
    for i, class_id in enumerate(unique_classes):
        if 0 <= class_id < len(age_categories):
            results_text += f"- Class {class_id} ({age_categories[class_id]}): '{age_text_prompts[class_id]}'\n"
        else:
            results_text += f"- Class {class_id} (Unknown): 'No prompt available'\n"
    
    results_text += f"""

Model Performance Metrics:
- Accuracy: {accuracy:.4f}
- Precision: {precision:.4f}
- Recall: {recall:.4f}
- F1-Score: {f1:.4f}

Dataset Information:
- Total samples: {len(y_true)}
- Valid samples: {len(y_true_valid)}
- Missing samples: {np.sum(~valid_mask)}
- Correct predictions: {np.sum(y_true_valid == y_pred_valid)}
- Incorrect predictions: {np.sum(y_true_valid != y_pred_valid)}
- Number of classes: {num_classes}
- Classes present: {unique_classes.tolist()}

Confidence Statistics:
- Mean confidence: {np.mean(max_probs):.4f}
- Std confidence: {np.std(max_probs):.4f}
- Min confidence: {np.min(max_probs):.4f}
- Max confidence: {np.max(max_probs):.4f}

Confusion Matrix:
{cm}

Class Distribution (True Labels):
"""
    
    for class_id in unique_classes:
        count = np.sum(y_true_valid == class_id)
        label = age_categories[class_id] if 0 <= class_id < len(age_categories) else f"Unknown_{class_id}"
        results_text += f"- {label}: {count}\n"
    
    results_text += "\nClass Distribution (Predictions):\n"
    for class_id in unique_classes:
        count = np.sum(y_pred_valid == class_id)
        label = age_categories[class_id] if 0 <= class_id < len(age_categories) else f"Unknown_{class_id}"
        results_text += f"- {label}: {count}\n"
    
    results_text += f"\nPer-Class Accuracy:\n"
    for i, class_id in enumerate(unique_classes):
        label = age_categories[class_id] if 0 <= class_id < len(age_categories) else f"Unknown_{class_id}"
        results_text += f"- {label}: {per_class_accuracy[i]:.4f}\n"
    
    results_text += f"\nDetailed Classification Report:\n"
    results_text += classification_report(y_true_valid, y_pred_valid, target_names=safe_target_names, zero_division=0)
    
    with open(os.path.join(output_dir, "results", "detailed_results.txt"), 'w') as f:
        f.write(results_text)
    
    print(f"\nResults saved to: {output_dir}")
    print("- Plots saved in: plots/")
    print("- Metrics saved in: results/")
    
    return metrics

if __name__ == "__main__":
    args = parse_args()
    test_age_classification(
        num_prompt=args.num_prompt,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        dataset_name=args.dataset_name
    )