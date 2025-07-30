import os
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import torch
from torch.utils.data import DataLoader
import json

import matplotlib.pyplot as plt

def create_model():
    """
    Create and return the PEcore model for gender classification.
    This function should be implemented to return your specific model.
    """
    # TODO: Implement model creation
    pass

def create_dataset():
    """
    Create and return the BaseDataset for gender classification testing.
    This function should be implemented to return your specific dataset.
    """
    # TODO: Implement dataset creation
    pass

def create_output_directories():
    """Create output directories if they don't exist."""
    base_dir = "MODEL_TEST/GENDER"
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(os.path.join(base_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "results"), exist_ok=True)
    return base_dir

def test_gender_classification():
    """Test the gender classification model and generate comprehensive metrics."""
    
    # Create output directories
    output_dir = create_output_directories()
    
    # Initialize model and dataset
    model = create_model()
    dataset = create_dataset()
    
    # Create data loader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # Set model to evaluation mode
    model.eval()
    
    all_predictions = []
    all_true_labels = []
    
    print("Starting gender classification testing...")
    
    # Perform inference
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            outputs = model(inputs)
            predictions = torch.argmax(outputs, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())
            
            if batch_idx % 10 == 0:
                print(f"Processed batch {batch_idx}/{len(dataloader)}")
    
    # Convert to numpy arrays
    y_true = np.array(all_true_labels)
    y_pred = np.array(all_predictions)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Print metrics
    print("\n" + "="*50)
    print("GENDER CLASSIFICATION TEST RESULTS")
    print("="*50)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['Male', 'Female']))
    
    # Save metrics to file
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist(),
        'total_samples': len(y_true),
        'correct_predictions': int(np.sum(y_true == y_pred))
    }
    
    with open(os.path.join(output_dir, "results", "gender_metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Plot and save confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Male', 'Female'], 
                yticklabels=['Male', 'Female'])
    plt.title('Gender Classification Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plots", "confusion_matrix.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot metrics bar chart
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    metrics_values = [accuracy, precision, recall, f1]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics_names, metrics_values, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
    plt.title('Gender Classification Metrics')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, value in zip(bars, metrics_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plots", "metrics_bar_chart.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot prediction distribution
    unique, counts = np.unique(y_pred, return_counts=True)
    gender_labels = ['Male', 'Female']
    
    plt.figure(figsize=(8, 6))
    plt.pie(counts, labels=[gender_labels[i] for i in unique], autopct='%1.1f%%', startangle=90)
    plt.title('Distribution of Gender Predictions')
    plt.savefig(os.path.join(output_dir, "plots", "prediction_distribution.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save detailed results
    results_text = f"""
GENDER CLASSIFICATION TEST RESULTS
{'='*50}

Model Performance Metrics:
- Accuracy: {accuracy:.4f}
- Precision: {precision:.4f}
- Recall: {recall:.4f}
- F1-Score: {f1:.4f}

Dataset Information:
- Total samples: {len(y_true)}
- Correct predictions: {np.sum(y_true == y_pred)}
- Incorrect predictions: {np.sum(y_true != y_pred)}

Confusion Matrix:
{cm}

Class Distribution (True Labels):
- Male (0): {np.sum(y_true == 0)}
- Female (1): {np.sum(y_true == 1)}

Class Distribution (Predictions):
- Male (0): {np.sum(y_pred == 0)}
- Female (1): {np.sum(y_pred == 1)}

Detailed Classification Report:
{classification_report(y_true, y_pred, target_names=['Male', 'Female'])}
"""
    
    with open(os.path.join(output_dir, "results", "detailed_results.txt"), 'w') as f:
        f.write(results_text)
    
    print(f"\nResults saved to: {output_dir}")
    print("- Plots saved in: plots/")
    print("- Metrics saved in: results/")
    
    return metrics

if __name__ == "__main__":
    test_gender_classification()