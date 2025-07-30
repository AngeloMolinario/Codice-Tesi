import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from dataset.dataset import BaseDataset
from wrappers.promptopt.prompt_learner import CustomModel
from wrappers.PerceptionEncoder.pe import PECore
import core.vision_encoder.transforms as transforms

# Constants
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Task definitions
TASK_NAMES = ["Age estimation", "Gender recognition", "Emotion Classification"]
AGE_GROUPS = ["0-9", "10-30", "31-50", "51-70", "71+"]
GENDERS = ["male", "female"]
EMOTIONS = ["Happy", "sad", "natural", "surprised"]
CLASSNAMES = [AGE_GROUPS, GENDERS, EMOTIONS]

# Model parameters
N_CTX = 16

def create_loss_functions():
    """Create CrossEntropy loss functions for each task"""
    age_criterion = nn.CrossEntropyLoss(reduction='none')
    gender_criterion = nn.CrossEntropyLoss(reduction='none')
    emotion_criterion = nn.CrossEntropyLoss(reduction='none')
    
    return age_criterion, gender_criterion, emotion_criterion

def compute_multitask_loss(logits, labels, loss_functions):
    """
    Compute multi-task loss with equal weights
    Missing labels (-1) contribute 0 to the loss
    """
    age_criterion, gender_criterion, emotion_criterion = loss_functions
    
    # Split logits for each task
    num_age_classes = len(AGE_GROUPS)
    num_gender_classes = len(GENDERS)
    num_emotion_classes = len(EMOTIONS)
    
    age_logits = logits[:, :num_age_classes]
    gender_logits = logits[:, num_age_classes:num_age_classes + num_gender_classes]
    emotion_logits = logits[:, num_age_classes + num_gender_classes:]
    
    # Extract labels
    age_labels = labels['age'].long()
    gender_labels = labels['gender'].long()
    emotion_labels = labels['emotion'].long()
    
    # Create masks for valid labels (not -1)
    age_mask = (age_labels != -1)
    gender_mask = (gender_labels != -1)
    emotion_mask = (emotion_labels != -1)
    
    # Initialize losses to 0
    age_loss = torch.tensor(0.0, device=logits.device)
    gender_loss = torch.tensor(0.0, device=logits.device)
    emotion_loss = torch.tensor(0.0, device=logits.device)
    
    # Compute losses only for valid labels
    if age_mask.any():
        valid_age_logits = age_logits[age_mask]
        valid_age_labels = age_labels[age_mask]
        age_loss = age_criterion(valid_age_logits, valid_age_labels).mean()
    
    if gender_mask.any():
        valid_gender_logits = gender_logits[gender_mask]
        valid_gender_labels = gender_labels[gender_mask]
        gender_loss = gender_criterion(valid_gender_logits, valid_gender_labels).mean()
    
    if emotion_mask.any():
        valid_emotion_logits = emotion_logits[emotion_mask]
        valid_emotion_labels = emotion_labels[emotion_mask]
        emotion_loss = emotion_criterion(valid_emotion_logits, valid_emotion_labels).mean()
    
    # Equal weights for all tasks
    total_loss = (age_loss + gender_loss + emotion_loss)
    
    return total_loss, age_loss, gender_loss, emotion_loss

def training_epoch(model, dataloader, optimizer, loss_functions, device):
    """
    Perform one training epoch
    Returns mean loss over the epoch
    """
    model.train()
    total_loss = 0.0
    total_age_loss = 0.0
    total_gender_loss = 0.0
    total_emotion_loss = 0.0
    num_batches = 0
    
    for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="Training")):
        # Move data to device
        images = images.to(device)
        for key in labels:
            labels[key] = labels[key].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(images)
        
        # Compute loss
        loss, age_loss, gender_loss, emotion_loss = compute_multitask_loss(
            logits, labels, loss_functions
        )
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Accumulate losses
        total_loss += loss.item()
        total_age_loss += age_loss.item()
        total_gender_loss += gender_loss.item()
        total_emotion_loss += emotion_loss.item()
        num_batches += 1
    
    # Return mean losses
    mean_loss = total_loss / num_batches
    mean_age_loss = total_age_loss / num_batches
    mean_gender_loss = total_gender_loss / num_batches
    mean_emotion_loss = total_emotion_loss / num_batches
    
    return mean_loss, mean_age_loss, mean_gender_loss, mean_emotion_loss

def main():
    """Main training loop"""
    print(f"Using device: {DEVICE}")
    
    # Load CLIP model
    
    model = PECore.from_config("PE-Core-B16-224", pretrained=True, num_prompt=0)
    preprocess = transforms.get_image_transform(model.image_size)
    tokenizer = transforms.get_text_tokenizer(model.text_model.context_length)
    # Initialize custom model
    model = CustomModel(
        n_ctx=N_CTX,
        tasknames=TASK_NAMES,
        classnames=CLASSNAMES,
        model=model,
        tokenizer=tokenizer
    ).to(DEVICE)
    
    # Create dataset and dataloader
    train_dataset = BaseDataset(
        root="./datasets_with_standard_labels/CelebA_HQ",  # Update this path
        transform=preprocess,
        split="train"
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )
    
    # Create loss functions
    loss_functions = create_loss_functions()
    
    # Create optimizer
    model.train()
    for param in model.parameters():
        param.requires_grad = False
    
    for param in model.get_softCPT_parameters():
        param.requires_grad = True

    optimizer = optim.AdamW(model.get_softCPT_parameters(), lr=LEARNING_RATE)
    
    # Training loop
    print("Starting training...")
    best_loss = float('inf')
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        
        # Training epoch
        mean_loss, age_loss, gender_loss, emotion_loss = training_epoch(
            model, train_dataloader, optimizer, loss_functions, DEVICE
        )
        
        # Print losses
        print(f"Mean Loss: {mean_loss:.4f}")
        print(f"Age Loss: {age_loss:.4f}")
        print(f"Gender Loss: {gender_loss:.4f}")
        print(f"Emotion Loss: {emotion_loss:.4f}")
        
        # Save best model
        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': mean_loss,
            }, 'best_model.pth')
            print(f"New best model saved with loss: {best_loss:.4f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': mean_loss,
            }, f'checkpoint_epoch_{epoch + 1}.pth')
    
    print("Training completed!")

if __name__ == "__main__":
    main()