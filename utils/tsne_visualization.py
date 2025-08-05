import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

def extract_similarity_features(model, dataloader, device, task_name, text_features=None, max_samples=None):
    """
    Estrae features di similarity usando cosine similarity e logit_scale
    """
    model.eval()
    similarity_features_list = []
    task_labels = []
    
    # Ottieni text features per il task specifico
    if text_features is None:
        try:
            text_features = model.get_text_features(normalize=True)
            if isinstance(text_features, dict) and task_name in text_features:
                task_text_features = text_features[task_name]
            else:
                task_text_features = text_features
        except:
            print(f"Error: Could not extract text features for task {task_name}")
            return None, None
    else:
        if isinstance(text_features, dict) and task_name in text_features:
            task_text_features = text_features[task_name]
        else:
            task_text_features = text_features
    
    # Ottieni logit_scale dal modello
    if hasattr(model, 'logit_scale'):
        logit_scale = model.logit_scale.exp()
    else:
        logit_scale = torch.tensor(1.0, device=device)
        print("Warning: No logit_scale found in model, using 1.0")
    
    print(f"Using logit_scale: {logit_scale.item():.4f}")
    print(f"Text features shape for {task_name}: {task_text_features.shape}")
    
    # Estrai image features e calcola similarity
    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(dataloader, desc=f"Computing similarities for {task_name}")):
            if max_samples and i * dataloader.batch_size >= max_samples:
                break
                
            images = images.to(device)
            
            # Estrai image features normalizzate
            image_features = model.get_image_features(images, normalize=True)
            
            # Calcola cosine similarity con text features
            similarities = torch.mm(image_features, task_text_features.t())
            
            # Scala con logit_scale
            scaled_similarities = logit_scale * similarities
            
            similarity_features_list.append(scaled_similarities.cpu().numpy())
            
            # Raccogli solo le labels per il task specifico
            batch_labels = labels[task_name].numpy()
            task_labels.extend(batch_labels)
    
    # Concatena tutte le features
    similarity_features = np.concatenate(similarity_features_list, axis=0)
    task_labels = np.array(task_labels)
    
    print(f"Similarity features shape: {similarity_features.shape}")
    
    return similarity_features, task_labels

def create_similarity_tsne_plot(similarity_features, labels, task_name, output_dir, 
                               class_names=None, epoch=None, perplexity=30, n_components=2, 
                               random_state=42, use_pca=True, pca_components=50):
    """
    Crea solo lo scatterplot t-SNE delle similarity features
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Creating t-SNE for {task_name} similarity features...")
    
    # Applica PCA se richiesto
    features_for_tsne = similarity_features
    if use_pca and features_for_tsne.shape[1] > pca_components:
        print(f"Applying PCA: {features_for_tsne.shape[1]} -> {pca_components} dimensions")
        pca = PCA(n_components=pca_components, random_state=random_state)
        features_for_tsne = pca.fit_transform(features_for_tsne)
        print(f"PCA explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")
    
    # Applica t-SNE
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state, 
                verbose=1, n_iter=1000)
    similarity_tsne = tsne.fit_transform(features_for_tsne)
    
    # Filtra campioni validi
    valid_mask = labels != -1
    if not valid_mask.any():
        print(f"No valid labels for task {task_name}")
        return
    
    valid_similarity_tsne = similarity_tsne[valid_mask]
    valid_labels = labels[valid_mask]
    
    # Crea il plot
    plt.figure(figsize=(12, 9))
    
    unique_labels = np.unique(valid_labels)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    
    # Plot delle similarity features per classe
    for i, label in enumerate(unique_labels):
        mask = valid_labels == label
        class_name = class_names[int(label)] if class_names and int(label) < len(class_names) else f'Class {int(label)}'
        plt.scatter(valid_similarity_tsne[mask, 0], valid_similarity_tsne[mask, 1], 
                   c=[colors[i]], label=f'{class_name}', 
                   alpha=0.7, s=30)
    
    # Aggiorna il titolo per includere l'epoca se fornita
    title = f't-SNE: {task_name.capitalize()} - Cosine Similarity Features (Scaled)'
    if epoch is not None:
        title += f' - Epoch {epoch}'
    
    plt.title(title, fontsize=16)
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Crea il nome del file con l'epoca
    if epoch is not None:
        filename = f'tsne_{task_name}_similarity_features_epoch_{epoch:03d}.png'
    else:
        filename = f'tsne_{task_name}_similarity_features.png'
    
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved similarity t-SNE plot to {output_dir}/{filename}")

def visualize_similarity_representations(model, dataloader, device, task_name, output_dir, 
                                       class_names=None, text_features=None, max_samples=5000, 
                                       perplexity=30, epoch=None):
    """
    Funzione principale per visualizzare solo le similarity features
    """
    print(f"Computing cosine similarities with logit scaling for {task_name} task...")
    
    similarity_features, labels = extract_similarity_features(
        model, dataloader, device, task_name, text_features=text_features, max_samples=max_samples
    )
    
    if similarity_features is None:
        print(f"Failed to extract similarity features for {task_name}")
        return
    
    print(f"Similarity features shape: {similarity_features.shape}")
    print(f"Valid labels: {np.sum(labels != -1)}")
    
    print(f"Creating similarity t-SNE visualization for {task_name}...")
    create_similarity_tsne_plot(
        similarity_features, labels, task_name, output_dir, 
        class_names=class_names, 
        epoch=epoch,  # Passa il numero di epoca
        perplexity=perplexity
    )
    
    print(f"Similarity visualization for {task_name} saved to {output_dir}")