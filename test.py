from dataset.dataset import MultiDataset
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score
from wrappers.promptopt.prompt_learner import CustomModel
from core.vision_encoder import transforms
from wrappers.PerceptionEncoder.pe import PECore
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

TASK_NAMES = ["Age group estimation", "Gender recognition", "Emotion classification"]
CLASSES = [
    ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"],
    ["male", "female"],
    ["Surprise", "Fear", "Disgust", "Happy", "Sad", "Angry", "Neutral"]
]
PROMPT_GEN_TYPE = "lin"

def load_model(ckpt, cntx):
    """Load the trained model from checkpoint."""
    base_model = PECore.from_config("PE-Core-B16-224", pretrained=True, num_prompt=0)

    model = CustomModel(
        n_ctx=int(cntx),
        tasknames=TASK_NAMES,
        classnames=CLASSES,
        model=base_model,
        tokenizer=transforms.get_text_tokenizer(base_model.text_model.context_length)
    )
    
    # Load checkpoint
    checkpoint = torch.load(ckpt, map_location='cpu')
    
    # Handle torch.compile() prefix (_orig_mod.) in state_dict keys
    if any(key.startswith('_orig_mod.') for key in checkpoint.keys()):
        # Remove _orig_mod. prefix from all keys
        new_state_dict = {}
        for key, value in checkpoint.items():
            new_key = key.replace('_orig_mod.', '')
            new_state_dict[new_key] = value
        checkpoint = new_state_dict
    
    # Load state dict with strict=False to handle any remaining mismatches
    model.load_state_dict(checkpoint, strict=False)
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    return model


def predict(model, dataloader):
    """
    Make predictions on the dataset and return confusion matrices and accuracy scores.
    Only considers non-missing labels (label != -1).
    
    Returns:
        tuple: (confusion_age, confusion_emotion, confusion_gender, age_acc, emotion_acc, gender_acc)
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Storage for predictions and true labels (only non-missing values)
    age_predictions = []
    age_true_labels = []
    emotion_predictions = []
    emotion_true_labels = []
    gender_predictions = []
    gender_true_labels = []

    with torch.no_grad():
        text_features = model.get_text_features(normalize=True)

        for batch in tqdm(dataloader, desc="Making predictions"):
            images, labels = batch
            images = images.to(device)
            
            # Get model predictions
            image_features = model.get_image_features(images, normalize=True)

            logits = model.logit_scale.exp() * (image_features @ text_features.t())

            # Split logits for each task
            age_logits = logits[:, :len(CLASSES[0])]
            gender_logits = logits[:, len(CLASSES[0]):len(CLASSES[0]) + len(CLASSES[1])]
            emotion_logits = logits[:, len(CLASSES[0]) + len(CLASSES[1]):]
            
            # Get predictions for each task
            # Age: Ordinal regression - weighted sum of probabilities
            age_probs = torch.softmax(age_logits, dim=1)
            age_indices = torch.arange(len(CLASSES[0]), device=age_probs.device).float()
            age_pred_continuous = torch.sum(age_probs * age_indices.unsqueeze(0), dim=1)
            age_pred_batch = torch.round(age_pred_continuous).long().cpu()
            # Clamp to valid age group range
            age_pred_batch = torch.clamp(age_pred_batch, 0, len(CLASSES[0]) - 1)
            
            gender_pred_batch = torch.argmax(torch.softmax(gender_logits, dim=1), dim=1).cpu()
            
            emotion_pred_batch = torch.argmax(torch.softmax(emotion_logits, dim=1), dim=1).cpu()
            
            # Process entire batch efficiently using vectorized operations
            # Filter valid (non-missing) labels and corresponding predictions
                        
            age_valid_mask = labels[0] != -1
            if age_valid_mask.any():
                age_true_labels.extend(labels[0][age_valid_mask].cpu().numpy())
                age_predictions.extend(age_pred_batch[age_valid_mask].numpy())
                    
            gender_valid_mask = labels[1] != -1
            if gender_valid_mask.any():
                gender_true_labels.extend(labels[1][gender_valid_mask].cpu().numpy())
                gender_predictions.extend(gender_pred_batch[gender_valid_mask].numpy())
                    
        
            emotion_valid_mask = labels[2] != -1
            if emotion_valid_mask.any():
                emotion_true_labels.extend(labels[2][emotion_valid_mask].cpu().numpy())
                emotion_predictions.extend(emotion_pred_batch[emotion_valid_mask].numpy())

    # Compute confusion matrices only for available predictions
    confusion_age = None
    confusion_gender = None
    confusion_emotion = None
    age_acc = 0.0
    gender_acc = 0.0
    emotion_acc = 0.0
    
    if age_true_labels:
        confusion_age = confusion_matrix(age_true_labels, age_predictions, 
                                       labels=list(range(len(CLASSES[0]))))
        age_acc = accuracy_score(age_true_labels, age_predictions)
        print(f"Age predictions: {len(age_predictions)} samples")
    
    if gender_true_labels:
        confusion_gender = confusion_matrix(gender_true_labels, gender_predictions,
                                          labels=list(range(len(CLASSES[1]))))
        gender_acc = accuracy_score(gender_true_labels, gender_predictions)
        print(f"Gender predictions: {len(gender_predictions)} samples")
    
    if emotion_true_labels:
        confusion_emotion = confusion_matrix(emotion_true_labels, emotion_predictions,
                                           labels=list(range(len(CLASSES[2]))))
        emotion_acc = accuracy_score(emotion_true_labels, emotion_predictions)
        print(f"Emotion predictions: {len(emotion_predictions)} samples")
    
    return confusion_age, confusion_emotion, confusion_gender, age_acc, emotion_acc, gender_acc

def save_confusion_matrix(confusion_matrix_data, class_names, task_name, output_path):
    """
    Save confusion matrix as an image.
    
    Args:
        confusion_matrix_data: The confusion matrix array
        class_names: List of class names for the task
        task_name: Name of the task (for title and filename)
        output_path: Directory to save the image
    """
    if confusion_matrix_data is None:
        print(f"No data available for {task_name} confusion matrix")
        return
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix_data, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {task_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save the plot
    save_path = os.path.join(output_path, f'{task_name.lower()}_confusion_matrix.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix for {task_name} saved to: {save_path}")


def save_accuracy_results(age_acc, gender_acc, emotion_acc, output_path):
    """
    Save accuracy results to a text file.
    
    Args:
        age_acc: Age task accuracy
        gender_acc: Gender task accuracy  
        emotion_acc: Emotion task accuracy
        output_path: Directory to save the results
    """
    # Calculate mean accuracy
    valid_accuracies = []
    if age_acc > 0:
        valid_accuracies.append(age_acc)
    if gender_acc > 0:
        valid_accuracies.append(gender_acc)
    if emotion_acc > 0:
        valid_accuracies.append(emotion_acc)
    
    mean_acc = sum(valid_accuracies) / len(valid_accuracies) if valid_accuracies else 0.0
    
    # Save to file
    results_path = os.path.join(output_path, 'accuracy_results.txt')
    with open(results_path, 'w') as f:
        f.write(f"age_accuracy: {age_acc:.4f}\n")
        f.write(f"gender_accuracy: {gender_acc:.4f}\n")
        f.write(f"emotion_accuracy: {emotion_acc:.4f}\n")
        f.write(f"mean_accuracy: {mean_acc:.4f}\n")
    
    print(f"Accuracy results saved to: {results_path}")
    print(f"Age accuracy: {age_acc:.4f}")
    print(f"Gender accuracy: {gender_acc:.4f}")
    print(f"Emotion accuracy: {emotion_acc:.4f}")
    print(f"Mean accuracy: {mean_acc:.4f}")

import sys
def main():
    """Main function to run the testing pipeline."""
    # Configuration - modify these paths as needed
    MODEL_CHECKPOINT_PATH = sys.argv[1] #"../TRAINING_OUTPUT/CustomModel_multitask/ckpt/best_model.pth"
    DATASET_PATH = "../processed_datasets/datasets_with_standard_labels"
    OUTPUT_PATH = "TEST"
    BATCH_SIZE = 512

    CNTX = sys.argv[2]
    
    # Create output directory
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    # Load model
    print("Loading model...")
    model = load_model(MODEL_CHECKPOINT_PATH, CNTX)
    
    # Load dataset
    print("Loading dataset...")
    dataset = MultiDataset(dataset_names=["CelebA_HQ","RAF-DB","FairFace","UTKFace"], transform=transforms.get_image_transform(model.image_size), split='test', datasets_root=DATASET_PATH, all_datasets=False)
    info = dataset.get_dataset_info(compute_stats=True)
    for name, details in info.items():
        print(f"Dataset: {name}")
        print(f"  Number of samples: {details['length']}")
        print(f"  Global index range: [{details['start_idx']} - {details['end_idx']}]")
        stats = details.get('stats', {})
        for task in ['age', 'gender', 'emotion']:
            dist = stats.get(f"{task}_distribution", {})
            missing = stats.get(f"{task}_missing", None)
            print(f"  {task.capitalize()} statistics:")
            if dist:
                print(f"    Distribution:")
                for k, v in dist.items():
                    print(f"      Class {k}: {v} samples")
            else:
                print("    Distribution: (none)")
            print(f"    Missing: {missing} samples")
        print()

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Make predictions
    print("Making predictions...")
    confusion_age, confusion_emotion, confusion_gender, age_acc, emotion_acc, gender_acc = predict(model, dataloader)
    
    # Save confusion matrices
    print("Saving confusion matrices...")
    save_confusion_matrix(confusion_age, CLASSES[0], "Age", OUTPUT_PATH)
    save_confusion_matrix(confusion_emotion, CLASSES[2], "Emotion", OUTPUT_PATH)
    save_confusion_matrix(confusion_gender, CLASSES[1], "Gender", OUTPUT_PATH)
    
    # Save accuracy results
    print("Saving accuracy results...")
    save_accuracy_results(age_acc, gender_acc, emotion_acc, OUTPUT_PATH)
    
    print("Testing completed successfully!")

if __name__ == "__main__":
    main()
