import os
import sys
import json
import torch
import shutil
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataset.dataset import BaseDataset, MultiDataset
from wrappers.PerceptionEncoder.pe import PECore
from wrappers.promptopt.prompt_learner import CustomModel
from training import training_functions
from training.loss import *
from core.vision_encoder import transforms
from training.training_functions import *
from utils.metric import TrainingMetrics
from utils.configuration import Config



# Load configuration from JSON file
configuration_path = sys.argv[1] if len(sys.argv) > 1 else "config/PECore_VPT_age.json"
config = Config(configuration_path)
# Set a generator
generator = torch.Generator().manual_seed(config.SEED)


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")
print(f"Loaded configuration: {config}")

os.makedirs(config.OUTPUT_DIR, exist_ok=True)
shutil.copy2(configuration_path, f'{config.OUTPUT_DIR}/training_configuration.json')

model = get_model(config).to(DEVICE)

# Load model weights if checkpoint path is provided
if hasattr(config, 'CHECKPOINT_PATH') and config.CHECKPOINT_PATH:
    print(f"Loading model weights from: {config.CHECKPOINT_PATH}")
    checkpoint = torch.load(config.CHECKPOINT_PATH, map_location=DEVICE)
    
    # Handle different checkpoint formats
    state_dict = None
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print("Found model_state_dict in checkpoint")
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print("Found state_dict in checkpoint")
    else:
        # Assume the checkpoint is the state dict itself
        state_dict = checkpoint
        print("Using checkpoint as state dict")
    
    # Remove _orig_mod. prefix if present (from torch.compile)
    if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
        print("Removing _orig_mod. prefix from compiled model checkpoint...")
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('_orig_mod.'):
                new_key = key[10:]  # Remove '_orig_mod.' prefix
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        state_dict = new_state_dict
    
    # Load the cleaned state dict
    model.load_state_dict(state_dict)
    print("Model weights loaded successfully!")
else:
    print("No checkpoint path provided, using pretrained weights")


tokenizer = transforms.get_text_tokenizer(model.text_model.context_length)
img_transform = transforms.get_image_transform(model.image_size)


loss_fn = get_task_loss_fn(config)


dataset = get_datasets(config.DATASET_NAMES, transforms=img_transform, split="test", dataset_root=config.DATASET_ROOT)

validation_loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, persistent_workers=True, pin_memory=True)


text_features = None
if config.MODEL_TYPE !=  "SoftCPT":
    print(f"Encoding text features for {config.TASK} classification...")
    texts = tokenizer(config.TEXT_CLASSES_PROMPT).to(DEVICE)
    with torch.no_grad():
        text_features = model.get_text_features(texts, normalize=True)
    print(f"Encoded text features shape: {text_features.shape}")

epoch_train_fn = get_training_step_fn(config.TASK)
epoch_val_fn   = get_validation_step_fn(config.TASK)

training_losses = []
training_accuracies = []
validation_ordinal_losses = []
validation_ordinal_accuracies = []
validation_ce_losses = []
validation_ce_accuracies = []

metrics_tracker = TrainingMetrics(output_dir=f'{config.OUTPUT_DIR}/metrics', class_names=config.CLASSES)
test_name = "baseline" if hasattr(config, 'CHECKPOINT_PATH') else "NUM_VISUAL_PROMPTS_" + str(config.NUM_VISUAL_PROMPTS)
with torch.no_grad():
    epoch_loss, epoch_accuracy, all_preds, all_labels = epoch_val_fn(model, validation_loader, loss_fn, config.TASK, DEVICE, text_features, use_tqdm=config.USE_TQDM)
    print(f"Validation Loss ORDINAL BEFORE TRAINING: {epoch_loss:.4f}, Validation Accuracy: {epoch_accuracy:.4f}")
    metrics_tracker.update_predictions(torch.cat(all_preds), torch.cat(all_labels))
    metrics_tracker.plot_confusion_matrix(epoch=test_name)
    metrics_tracker.reset_predictions()
