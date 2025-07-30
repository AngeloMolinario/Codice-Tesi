import os
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import json
import argparse
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

from dataset.dataset import BaseDataset
from wrappers.PerceptionEncoder.pe import PECore
from core.vision_encoder.transforms import *

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Age Classification Training')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading (default: 4)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training (default: 64)')
    parser.add_argument('--dataset_name', type=str, default='FairFace',
                        help='Dataset name for training (default: FairFace)')
    parser.add_argument('--test_dataset_name', type=str, default="UTKFace",
                        help='Dataset name for testing (default: same as training dataset)')
    parser.add_argument('--num_prompt', type=int, default=0,
                        help='Number of context prompt tokens to be prepended to the image patches (default: 0)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay (default: 0.01)')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Number of warmup epochs (default: 5)')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Save model every N epochs (default: 10)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Temperature for softmax (default: 1.0)')
    parser.add_argument('--freeze_vision', action='store_true',
                        help='Freeze vision encoder weights')
    parser.add_argument('--freeze_text', action='store_true',
                        help='Freeze text encoder weights')
    parser.add_argument('--emd_p', type=int, default=2,
                        help='EMD loss norm degree (default: 2)')
    parser.add_argument('--use_scheduler', action='store_true',
                        help='Use learning rate scheduler (default: False)')
    
    return parser.parse_args()

def create_model(num_prompt=0):
    """
    Create and return the PEcore model for age classification.
    """
    model = PECore.from_config("PE-Core-B16-224", pretrained=True, num_prompt=num_prompt)
    return model

def create_datasets(dataset_name="UTKFace", test_dataset_name=None, image_transform=None):
    """
    Create and return train and test datasets for age classification.
    If test_dataset_name is provided, use it for testing, otherwise use the same dataset.
    """
    train_dataset = BaseDataset(
        root=f"./datasets_with_standard_labels/{dataset_name}",
        transform=image_transform,
        split="train"
    )
    
    # Use different dataset for testing if specified
    test_data_name = test_dataset_name if test_dataset_name else dataset_name
    test_dataset = BaseDataset(
        root=f"./datasets_with_standard_labels/{test_data_name}",
        transform=image_transform,
        split="test"
    )
    
    return train_dataset, test_dataset

def create_output_directories(base_name="MODEL_TRAINING/AGE"):
    """Create output directories with timestamp and incremental numbering."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Find the next available directory number
    counter = 0
    while True:
        if counter == 0:
            base_dir = f"{base_name}_{timestamp}"
        else:
            base_dir = f"{base_name}_{timestamp}_{counter}"
        
        if not os.path.exists(base_dir):
            break
        counter += 1
    
    # Create the directories
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(os.path.join(base_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "logs"), exist_ok=True)
    
    print(f"Created training directory: {base_dir}")
    return base_dir

def get_age_text_prompts():
    """Return text prompts for age classification."""
    return [
        "A photo of a person 0-2 years old",       # Class 0
        "A photo of a person 3-9 years old",       # Class 1
        "A photo of a person 10-19 years old",     # Class 2
        "A photo of a person 20-29 years old",     # Class 3
        "A photo of a person 30-39 years old",     # Class 4
        "A photo of a person 40-49 years old",     # Class 5
        "A photo of a person 50-59 years old",     # Class 6
        "A photo of a person 60-69 years old",     # Class 7
        "A photo of a person more than 70 years old"  # Class 8
    ]

def get_age_categories():
    """Return age category labels."""
    return [
        "0-2", "3-9", "10-19", "20-29", "30-39", 
        "40-49", "50-59", "60-69", "more than 70"
    ]

def tokenize_text_prompts(text_prompts, tokenizer):
    """Tokenize the text prompts for the model."""
    return tokenizer(text_prompts)

class EMDLoss(nn.Module):
    """
    Earth Mover's Distance (EMD) Loss for ordinal classification.
    Assumes output probabilities from softmax.
    """
    def __init__(self, reduction='mean', p=2):
        """
        Args:
            reduction (str): 'mean', 'sum' or 'none'
            p (int): the norm degree for distance (p=1: Wasserstein-1, p=2: squared EMD)
        """
        super().__init__()
        self.reduction = reduction
        self.p = p

    def forward(self, pred_probs, target):
        """
        Args:
            pred_probs: Tensor of shape (B, C) with class probabilities.
            target: Tensor of shape (B,) with class indices (int from 0 to C-1).
        Returns:
            emd_loss: scalar tensor
        """
        B, C = pred_probs.shape
        device = pred_probs.device

        # Create one-hot target distribution
        target_onehot = F.one_hot(target, num_classes=C).float()

        # CDFs
        pred_cdf = torch.cumsum(pred_probs, dim=1)
        target_cdf = torch.cumsum(target_onehot, dim=1)

        # EMD = Lp norm between the two CDFs
        cdf_diff = pred_cdf - target_cdf
        emd = torch.norm(cdf_diff, p=self.p, dim=1)

        if self.reduction == 'mean':
            return emd.mean()
        elif self.reduction == 'sum':
            return emd.sum()
        else:
            return emd  # shape (B,)

def setup_optimizer_and_scheduler(model, args, num_training_steps):
    """Setup optimizer and scheduler - only train prompt_learner parameters."""
    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False
    
    # Only enable gradient for prompt_learner parameters
    param_groups = []
    trainable_params = 0
    
    # Check for prompt_learner in vision encoder
    if hasattr(model.visual, 'prompt_learner'):
        for param in model.visual.prompt_learner.parameters():
            param.requires_grad = True
        param_groups.append({
            'params': model.visual.prompt_learner.parameters(), 
            'lr': args.lr
        })
        trainable_params += sum(p.numel() for p in model.visual.prompt_learner.parameters() if p.requires_grad)
        print(f"Enabled training for visual prompt_learner: {trainable_params} parameters")
    
    # Always keep logit_scale trainable
    if hasattr(model, 'logit_scale'):
        model.logit_scale.requires_grad = True
        param_groups.append({'params': [model.logit_scale], 'lr': args.lr})
        trainable_params += 1
        print(f"Enabled training for logit_scale: 1 parameter")
    
    if not param_groups:
        raise ValueError("No trainable parameters found! Make sure the model has prompt_learner.")
    
    print(f"Total trainable parameters: {trainable_params}")
    print(f"Total model parameters: {sum(p.numel() for p in model.parameters())}")
    
    optimizer = AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    
    if args.use_scheduler:
        scheduler = CosineAnnealingLR(optimizer, T_max=num_training_steps, eta_min=args.lr * 0.01)
        print("Learning rate scheduler enabled")
    else:
        scheduler = None
        print("Learning rate scheduler disabled")
    
    return optimizer, scheduler

def evaluate_model(model, dataloader, text_features, device, age_categories, temperature=1.0, emd_p=2):
    """Evaluate the model on a dataset with enhanced progress tracking."""
    model.eval()
    all_predictions = []
    all_true_labels = []
    all_losses = []
    
    criterion = EMDLoss(reduction='mean', p=emd_p)
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating", leave=False)
        for images, labels in pbar:
            images = images.to(device)
            age_labels = labels['age'].to(device)
            
            # Filter out missing labels (-1)
            valid_mask = age_labels != -1
            if valid_mask.sum() == 0:
                continue
                
            valid_images = images[valid_mask]
            valid_labels = age_labels[valid_mask]
            
            # Get image features
            image_features = model.encode_image(valid_images, normalize=True)
            
            # Compute similarity scores and convert to probabilities
            logits = torch.matmul(image_features, text_features.t()) / temperature
            probabilities = F.softmax(logits, dim=1)
            
            # Compute EMD loss
            loss = criterion(probabilities, valid_labels)
            all_losses.append(loss.item())
            
            # Compute predictions
            predictions = torch.argmax(probabilities, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_true_labels.extend(valid_labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({
                'samples': len(all_true_labels),
                'avg_loss': f"{np.mean(all_losses):.4f}"
            })
    
    # Convert to numpy
    y_true = np.array(all_true_labels)
    y_pred = np.array(all_predictions)
    
    # Calculate metrics
    if len(y_true) > 0:
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    else:
        accuracy = precision = recall = f1 = 0.0
    
    avg_loss = np.mean(all_losses) if all_losses else 0.0
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'valid_samples': len(y_true),
        'total_samples': len(y_true)
    }

def save_checkpoint(model, optimizer, scheduler, epoch, loss, output_dir, is_best=False):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'loss': loss,
    }
    
    # Save regular checkpoint
    checkpoint_path = os.path.join(output_dir, "checkpoints", f"checkpoint_epoch_{epoch}.pth")
    torch.save(checkpoint, checkpoint_path)
    
    # Save best model
    if is_best:
        best_path = os.path.join(output_dir, "checkpoints", "best_model.pth")
        torch.save(checkpoint, best_path)
        print(f"New best model saved at epoch {epoch}")
    
    # Save latest model (for resuming)
    latest_path = os.path.join(output_dir, "checkpoints", "latest_model.pth")
    torch.save(checkpoint, latest_path)

def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """Load model checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    best_loss = checkpoint.get('loss', float('inf'))
    
    print(f"Resumed from epoch {checkpoint['epoch']}, loss: {best_loss:.4f}")
    return start_epoch, best_loss

def plot_training_curves(train_losses, train_accs, output_dir):
    """Plot and save training curves."""
    epochs = range(1, len(train_losses) + 1)
    
    # Plot losses and accuracies
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.title('Training Loss (EMD)')
    plt.xlabel('Epoch')
    plt.ylabel('EMD Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plots", "training_curves.png"), dpi=300, bbox_inches='tight')
    plt.close()

def save_final_model_and_report(model, results, args, output_dir):
    """Save the final model and training report in run/age folder structure."""
    
    # Create run/age directory structure
    run_dir = os.path.join(output_dir, "run")
    age_dir = os.path.join(run_dir, "age")
    os.makedirs(age_dir, exist_ok=True)
    
    print(f"Saving final model and report to: {age_dir}")
    
    # Save the final model checkpoint
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'args': vars(args),
        'final_metrics': results['final_metrics'],
        'training_completed': True
    }
    
    model_path = os.path.join(age_dir, "final_model.pth")
    torch.save(final_checkpoint, model_path)
    
    # Determine test dataset name for reporting
    test_dataset_display = args.test_dataset_name if args.test_dataset_name else args.dataset_name
    
    # Create comprehensive training report
    training_report = f"""
AGE CLASSIFICATION TRAINING REPORT (EMD Loss)
{'='*60}

TRAINING CONFIGURATION:
- Training Dataset: {args.dataset_name}
- Testing Dataset: {test_dataset_display}
- Model: PE-Core-B16-224
- Number of Prompts: {args.num_prompt}
- Epochs: {args.epochs}
- Batch Size: {args.batch_size}
- Learning Rate: {args.lr}
- Weight Decay: {args.weight_decay}
- Temperature: {args.temperature}
- EMD Loss p-norm: {args.emd_p}
- Save Frequency: {args.save_freq}
- Number of Workers: {args.num_workers}
- Training Mode: Only prompt_learner parameters trained

FINAL PERFORMANCE METRICS (on {test_dataset_display}):
{'='*30}
- Final Test EMD Loss: {results['final_metrics']['loss']:.4f}
- Final Test Accuracy: {results['final_metrics']['accuracy']:.4f}
- Final Test Precision: {results['final_metrics']['precision']:.4f}
- Final Test Recall: {results['final_metrics']['recall']:.4f}
- Final Test F1-Score: {results['final_metrics']['f1']:.4f}
- Valid Samples: {results['final_metrics']['valid_samples']}
- Total Samples: {results['final_metrics']['total_samples']}

TRAINING PROGRESS:
{'='*20}
- Total Training Epochs: {len(results['train_losses'])}
- Final Training EMD Loss: {results['train_losses'][-1] if results['train_losses'] else 'N/A':.4f}
- Final Training Accuracy: {results['train_accs'][-1] if results['train_accs'] else 'N/A':.4f}

TRAINING LOSSES (EMD, by epoch):
{', '.join([f'{loss:.4f}' for loss in results['train_losses']])}

TRAINING ACCURACIES (by epoch):
{', '.join([f'{acc:.4f}' for acc in results['train_accs']])}

MODEL USAGE INSTRUCTIONS:
{'='*30}
To load and use this trained model:

1. Load the checkpoint:
   ```python
   checkpoint = torch.load('final_model.pth')
   model = PECore.from_config("PE-Core-B16-224", pretrained=True, num_prompt={args.num_prompt})
   model.load_state_dict(checkpoint['model_state_dict'])
   model.eval()
   ```

2. The model expects age groups (0-8) as defined:
   - 0: 0-2 years old
   - 1: 3-9 years old
   - 2: 10-19 years old
   - 3: 20-29 years old
   - 4: 30-39 years old
   - 5: 40-49 years old
   - 6: 50-59 years old
   - 7: 60-69 years old
   - 8: 70+ years old

3. Use the same text prompts for inference:
   {get_age_text_prompts()}

4. The model was trained with EMD Loss (p={args.emd_p}) which considers ordinal relationships between age groups.

DATASET INFORMATION:
- Training performed on: {args.dataset_name}
- Testing performed on: {test_dataset_display}
- Cross-dataset evaluation: {'Yes' if args.test_dataset_name else 'No'}

TRAINING COMPLETED: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
    
    # Save training report
    report_path = os.path.join(age_dir, "training_report.txt")
    with open(report_path, 'w') as f:
        f.write(training_report)
    
    # Save training configuration as JSON
    config_path = os.path.join(age_dir, "training_config.json")
    training_config = {
        'args': vars(args),
        'final_metrics': results['final_metrics'],
        'training_stats': {
            'total_epochs': len(results['train_losses']),
            'final_train_loss': results['train_losses'][-1] if results['train_losses'] else None,
            'final_train_acc': results['train_accs'][-1] if results['train_accs'] else None,
        },
        'datasets': {
            'training_dataset': args.dataset_name,
            'testing_dataset': test_dataset_display,
            'cross_dataset_evaluation': bool(args.test_dataset_name)
        },
        'age_categories': get_age_categories(),
        'age_text_prompts': get_age_text_prompts(),
        'training_completed_at': datetime.now().isoformat()
    }
    
    with open(config_path, 'w') as f:
        json.dump(training_config, f, indent=2)
    
    # Copy the best model checkpoint to the run/age folder if it exists
    best_model_path = os.path.join(output_dir, "checkpoints", "best_model.pth")
    if os.path.exists(best_model_path):
        import shutil
        best_model_copy_path = os.path.join(age_dir, "best_model.pth")
        shutil.copy2(best_model_path, best_model_copy_path)
        print(f"Best model checkpoint copied to: {best_model_copy_path}")
    
    print(f"Final model saved to: {model_path}")
    print(f"Training report saved to: {report_path}")
    print(f"Training config saved to: {config_path}")
    
    return age_dir

def train_age_classification(args):
    """Main training function with enhanced progress tracking."""
    
    # Create output directories
    output_dir = create_output_directories()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Using EMD Loss with p={args.emd_p}")
    print("Training mode: Only prompt_learner parameters will be trained")
    
    # Display dataset configuration
    test_dataset_name = args.test_dataset_name if args.test_dataset_name else args.dataset_name
    print(f"Training dataset: {args.dataset_name}")
    print(f"Testing dataset: {test_dataset_name}")
    if args.test_dataset_name:
        print("Cross-dataset evaluation enabled")
    
    # Create model
    print("Creating model...")
    model = create_model(num_prompt=args.num_prompt)
    model = model.to(device)
    
    # Create datasets with progress tracking
    print("Loading datasets...")
    image_transform = get_image_transform(model.image_size)
    tokenizer = get_text_tokenizer(model.text_model.context_length)
    train_dataset, test_dataset = create_datasets(args.dataset_name, args.test_dataset_name, image_transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"Train dataset ({args.dataset_name}): {len(train_dataset)} samples")
    print(f"Test dataset ({test_dataset_name}): {len(test_dataset)} samples")
    
    # Setup text prompts
    print("Encoding text prompts...")
    age_text_prompts = get_age_text_prompts()
    age_categories = get_age_categories()
    tokenized_texts = tokenize_text_prompts(age_text_prompts, tokenizer)
    
    # Encode text prompts
    model.eval()
    with torch.no_grad():
        if torch.is_tensor(tokenized_texts):
            tokenized_texts = tokenized_texts.to(device)
        text_features = model.encode_text(tokenized_texts, normalize=True)
    
    # Setup training - this will freeze all parameters except prompt_learner
    print("Setting up optimizer and scheduler...")
    num_training_steps = len(train_loader) * args.epochs
    optimizer, scheduler = setup_optimizer_and_scheduler(model, args, num_training_steps)
    criterion = EMDLoss(reduction='mean', p=args.emd_p)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        start_epoch, _ = load_checkpoint(model, optimizer, scheduler, args.resume)
    
    # Training loop
    train_losses = []
    train_accs = []
    
    print(f"Starting training for {args.epochs} epochs...")
    print("="*60)
    
    for epoch in range(start_epoch, args.epochs):
        # Training phase
        model.train()
        epoch_losses = []
        epoch_predictions = []
        epoch_labels = []
        
        # Enhanced progress bar for training
        pbar = tqdm(
            train_loader, 
            desc=f"Epoch {epoch+1:3d}/{args.epochs}",
            ncols=120,
            position=0,
            leave=True
        )
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(device)
            age_labels = labels['age'].to(device)
            
            # Filter out missing labels (-1)
            valid_mask = age_labels != -1
            if valid_mask.sum() == 0:
                continue
                
            valid_images = images[valid_mask]
            valid_labels = age_labels[valid_mask]
            
            # Forward pass
            image_features = model.encode_image(valid_images, normalize=True)
            
            # Compute similarity scores and convert to probabilities
            logits = torch.matmul(image_features, text_features.t()) / args.temperature
            probabilities = F.softmax(logits, dim=1)
            
            # Compute EMD loss
            loss = criterion(probabilities, valid_labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            
            # Track metrics
            epoch_losses.append(loss.item())
            
            # Compute predictions for accuracy
            with torch.no_grad():
                predictions = torch.argmax(probabilities, dim=1)
                epoch_predictions.extend(predictions.cpu().numpy())
                epoch_labels.extend(valid_labels.cpu().numpy())
            
            # Enhanced progress bar update
            current_lr = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]['lr']
            current_acc = accuracy_score(epoch_labels, epoch_predictions) if len(epoch_labels) > 0 else 0.0
            
            pbar.set_postfix({
                'EMD': f"{loss.item():.4f}",
                'Acc': f"{current_acc:.3f}",
                'LR': f"{current_lr:.2e}",
                'Batch': f"{batch_idx+1}/{len(train_loader)}"
            })
        
        # Calculate training metrics
        train_loss = np.mean(epoch_losses)
        train_acc = accuracy_score(epoch_labels, epoch_predictions) if epoch_labels else 0.0
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        print(f"\nEpoch {epoch+1:3d} Summary - Train EMD: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        
        # Save checkpoint periodically
        if (epoch + 1) % args.save_freq == 0:
            print(f"Saving checkpoint at epoch {epoch+1}...")
            save_checkpoint(model, optimizer, scheduler, epoch, train_loss, output_dir)
        
        print("-" * 60)
    
    # Final evaluation
    print("\n" + "="*60)
    print("Running final evaluation on test set...")
    final_metrics = evaluate_model(model, test_loader, text_features, device, age_categories, args.temperature, args.emd_p)
    
    # Save final results
    results = {
        'final_metrics': final_metrics,
        'train_losses': train_losses,
        'train_accs': train_accs,
        'args': vars(args)
    }
    
    with open(os.path.join(output_dir, "training_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot training curves
    plot_training_curves(train_losses, train_accs, output_dir)
    
    # Save final model
    save_checkpoint(model, optimizer, scheduler, args.epochs-1, final_metrics['loss'], output_dir)
    
    # Save the final model and report in run/age folder structure
    final_model_dir = save_final_model_and_report(model, results, args, output_dir)
    
    print(f"\nTraining completed!")
    print(f"Results saved to: {output_dir}")
    print(f"Final model and report saved to: {final_model_dir}")
    print(f"Final test accuracy: {final_metrics['accuracy']:.4f}")
    print(f"Final test EMD loss: {final_metrics['loss']:.4f}")
    
    return model, results

if __name__ == "__main__":
    args = parse_args()
    train_age_classification(args)