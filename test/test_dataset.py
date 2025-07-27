import sys
import os
import random
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

# Add the parent directory to the path to import the dataset module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.dataset import BaseDataset

def test_dataset_visualization():
    """
    Test function to visualize random samples from the dataset
    """
    # Define the path to your dataset
    dataset_root = "./datasets_with_standard_labels/CelebA_HQ"  # Adjust this path as needed

    # Define basic transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    try:
        # Load the dataset
        dataset = BaseDataset(root=dataset_root, transform=transform, split="train")
        print(f"Dataset loaded successfully! Total samples: {len(dataset)}")
        
        # Choose 5 random indices
        random_indices = random.sample(range(len(dataset)), min(3, len(dataset)))
        
        # Create a figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(20, 4))
        
        for i, idx in enumerate(random_indices):
            # Get image and label
            image, label = dataset[idx]
            
            # Convert tensor to numpy for display
            if isinstance(image, torch.Tensor):
                image_np = image.permute(1, 2, 0).numpy()
            else:
                image_np = image
            
            # Display the image
            axes[i].imshow(image_np)
            axes[i].axis('off')
            
            # Create label text
            label_text = []
            label_mapping = {
                'gender': {1: 'Female', 0: 'Male', -1: 'Unknown'},
                'age': lambda x: f"Age: {int(x)}" if x != -1 else "Age: Unknown",
                'ethnicity': lambda x: f"Ethnicity: {int(x)}" if x != -1 else "Ethnicity: Unknown",
                'emotion': lambda x: f"Emotion: {int(x)}" if x != -1 else "Emotion: Unknown",
                'identity': lambda x: f"ID: {int(x)}" if x != -1 else "ID: Unknown"
            }
            
            # Format labels
            if label['gender'] in label_mapping['gender']:
                label_text.append(label_mapping['gender'][label['gender']])
            
            label_text.append(label_mapping['age'](label['age']))
            label_text.append(label_mapping['ethnicity'](label['ethnicity']))
            label_text.append(label_mapping['emotion'](label['emotion']))
            label_text.append(label_mapping['identity'](label['identity']))
            
            # Set title with labels
            axes[i].set_title('\n'.join(label_text), fontsize=8, pad=10)
        
        plt.tight_layout()
        plt.show()
        
        # Print sample information
        print("\nSample details:")
        for i, idx in enumerate(random_indices):
            _, label = dataset[idx]
            print(f"Sample {i+1} (Index {idx}): {label}")
            
    except FileNotFoundError as e:
        print(f"Error: Could not find dataset at {dataset_root}")
        print("Please make sure the dataset path is correct and the dataset structure is:")
        print("  dataset_root/")
        print("    train/")
        print("      images/")
        print("      labels.csv")
        print("    test/")
        print("      images/")
        print("      labels.csv")
    except Exception as e:
        print(f"Error loading dataset: {e}")

if __name__ == "__main__":
    test_dataset_visualization()