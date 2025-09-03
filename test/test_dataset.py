import sys
import os
import random
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

# Add the parent directory to the path to import the dataset module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.dataset import BaseDataset, MultiDataset

def test_dataset_visualization(num_samples=5, show_images=True):
    """
    Test function to visualize random samples from the dataset.
    
    Args:
        num_samples (int): The number of samples to visualize.
        show_images (bool): Whether to display the images or not.
    """
    # Define the path to your dataset
    dataset_root = "../datasets_with_standard_labels/FairFace"  # Adjust this path as needed

    # Define basic transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    try:
        # Load the dataset
        dataset = BaseDataset(root=dataset_root, transform=transform, split="train")
        print(f"Dataset loaded successfully! Total samples: {len(dataset)}")
        
        # Choose N random indices
        random_indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
        
        if not random_indices:
            print("No samples to display.")
            return

        # Calculate grid size
        num_images = len(random_indices)
        cols = min(num_images, 3)
        rows = (num_images + cols - 1) // cols

        # Create a figure with subplots
        fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
        axes = axes.flatten()  # Flatten the axes array for easy iteration
        
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
            
            age_group_mapping = {
                0: "0-2 years",
                1: "3-9 years",
                2: "10-19 years",
                3: "20-29 years",
                4: "30-39 years",
                5: "40-49 years",
                6: "50-59 years",
                7: "60-69 years",
                8: "70+ years",
            }

            label_mapping = {
                'gender': {1: 'Female', 0: 'Male', -1: 'Unknown'},
                'age': lambda x: f"Age: {age_group_mapping.get(int(x), 'Unknown')}" if x != -1 else "Age: Unknown",
                'emotion': lambda x: f"Emotion: {int(x)}" if x != -1 else "Emotion: Unknown",
            }
            
            # Format labels
            if label['gender'].item() in label_mapping['gender']:
                label_text.append(label_mapping['gender'][label['gender'].item()])
            
            label_text.append(label_mapping['age'](label['age'].item()))
            label_text.append(label_mapping['emotion'](label['emotion'].item()))
            
            # Set title with labels
            axes[i].set_title('\n'.join(label_text), fontsize=8, pad=10)
        
        # Hide unused subplots
        for j in range(num_images, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        
        # Print sample information
        print("\nSample details:")
        for i, idx in enumerate(random_indices):
            _, label = dataset[idx]
            print(f"Sample {i+1} (Index {idx}): {label}")

        if show_images:
            plt.show()
        else:
            plt.close(fig)

            
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

def test_multidataset_visualization(num_samples=5, show_images=True):
    """
    Test function to visualize random samples from the MultiDataset.
    
    Args:
        num_samples (int): The number of samples to visualize.
        show_images (bool): Whether to display the images or not.
    """
    # Define the datasets to combine
    dataset_names = ["FairFace", "CelebA_HQ", "RAF-DB"]  # Adjust these names as needed
    datasets_root = "../datasets_with_standard_labels"

    # Define basic transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    try:
        # Load the MultiDataset
        multi_dataset = MultiDataset(
            dataset_names=dataset_names, 
            transform=transform, 
            split="train",
            datasets_root=datasets_root,
            all_datasets=True  # Set to True to load all datasets in the root directory
        )
        print(f"MultiDataset loaded successfully! Total samples: {len(multi_dataset)}")
        
        # Print dataset info
        dataset_info = multi_dataset.get_dataset_info()
        print("\nDataset breakdown:")
        for name, info in dataset_info.items():
            print(f"  {name}: {info['length']} samples (indices {info['start_idx']}-{info['end_idx']})")
        
        # Choose N random indices
        random_indices = random.sample(range(len(multi_dataset)), min(num_samples, len(multi_dataset)))
        
        if not random_indices:
            print("No samples to display.")
            return

        # Calculate grid size
        num_images = len(random_indices)
        cols = min(num_images, 3)
        rows = (num_images + cols - 1) // cols

        # Create a figure with subplots
        fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
        if rows == 1 and cols == 1:
            axes = [axes]  # Make it iterable for single subplot
        else:
            axes = axes.flatten()  # Flatten the axes array for easy iteration
        
        for i, idx in enumerate(random_indices):
            # Get image and label
            image, label = multi_dataset[idx]
            
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
            
            age_group_mapping = {
                0: "0-2 years",
                1: "3-9 years",
                2: "10-19 years",
                3: "20-29 years",
                4: "30-39 years",
                5: "40-49 years",
                6: "50-59 years",
                7: "60-69 years",
                8: "70+ years",
            }

            label_mapping = {
                'gender': {1: 'Female', 0: 'Male', -1: 'Unknown'},
                'age': lambda x: f"Age: {age_group_mapping.get(int(x), 'Unknown')}" if x != -1 else "Age: Unknown",
                'emotion': lambda x: f"Emotion: {int(x)}" if x != -1 else "Emotion: Unknown",
            }
            
            # Format labels
            if label['gender'].item() in label_mapping['gender']:
                label_text.append(label_mapping['gender'][label['gender'].item()])
            
            label_text.append(label_mapping['age'](label['age'].item()))
            label_text.append(label_mapping['emotion'](label['emotion'].item()))
            
            # Determine which dataset this sample comes from
            dataset_source = "Unknown"
            for name, info in dataset_info.items():
                if info['start_idx'] <= idx <= info['end_idx']:
                    dataset_source = name
                    break
            
            label_text.append(f"Source: {dataset_source}")
            
            # Set title with labels
            axes[i].set_title('\n'.join(label_text), fontsize=8, pad=10)
        
        # Hide unused subplots
        for j in range(num_images, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        
        # Print sample information
        print("\nSample details:")
        for i, idx in enumerate(random_indices):
            _, label = multi_dataset[idx]
            # Determine source dataset
            dataset_source = "Unknown"
            for name, info in dataset_info.items():
                if info['start_idx'] <= idx <= info['end_idx']:
                    dataset_source = name
                    break
            print(f"Sample {i+1} (Global Index {idx}, Source: {dataset_source}): {label}")

        if show_images:
            plt.show()
        else:
            plt.close(fig)

            
    except FileNotFoundError as e:
        print(f"Error: Could not find datasets in {datasets_root}")
        print("Please make sure the datasets path is correct and the dataset structures are:")
        print("  datasets_root/")
        print("    dataset_name1/")
        print("      train/")
        print("        images/")
        print("        labels.csv")
        print("    dataset_name2/")
        print("      train/")
        print("        images/")
        print("        labels.csv")
    except Exception as e:
        print(f"Error loading MultiDataset: {e}")

if __name__ == "__main__":
    SHOW_IMAGES = False
    print("=" * 50)
    print("Testing BaseDataset")
    print("=" * 50)
    test_dataset_visualization(num_samples=5, show_images=SHOW_IMAGES)

    print("\n" + "=" * 50)
    print("Testing MultiDataset")
    print("=" * 50)
    test_multidataset_visualization(num_samples=8, show_images=SHOW_IMAGES)