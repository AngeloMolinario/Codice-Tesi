#!/usr/bin/env python3
"""
Script to regenerate plots from saved .npy analysis files.
Usage: python regenerate_plots.py <TRAIN_FOLDER>
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def regenerate_from_analysis_data(npy_path, output_dir):
    """
    Regenerate all plots from analysis_data.npy file.
    """
    print(f"Processing: {npy_path}")
    
    try:
        data = np.load(npy_path, allow_pickle=True).item()
    except Exception as e:
        print(f"ERROR: Could not load {npy_path}: {e}")
        return False
    
    # Extract data
    class_names = data.get('class_names', [])
    task_names = data.get('task_names', [])
    accuracies = data.get('accuracies', [])
    num_classes = data.get('num_classes', len(class_names))
    per_class = data.get('per_class', [])
    offset_matrix = data.get('offset_matrix', None)
    offset_ticks = data.get('offset_ticks', None)
    prob_matrix = data.get('prob_matrix', None)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Helper function for bar annotations
    def _annotate_bars_and_fix_ylim(ax, bars):
        ax.set_ylim(0, 1)
        for rect in bars:
            h = rect.get_height()
            y = min(h + 0.02, 0.98)
            ax.text(rect.get_x() + rect.get_width() / 2.0, y, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=9)
    
    # 1. Probability distribution per class
    prob_dir = os.path.join(output_dir, "Prob_distribution_per_class")
    os.makedirs(prob_dir, exist_ok=True)
    
    for class_data in per_class:
        c = class_data['class_index']
        class_name = class_data['class_name']
        norm_counts = class_data.get('norm_counts', None)
        
        if norm_counts is not None:
            plt.figure(figsize=(8, 5))
            ax = plt.gca()
            bars = ax.bar(class_names, norm_counts)
            plt.xticks(rotation=45)
            plt.xlabel("Classi")
            plt.ylabel("Frazione di campioni (normalizzata)")
            plt.title(f"Distribuzione di probabilità normalizzata - {class_name}")
            _annotate_bars_and_fix_ylim(ax, bars)
            plt.tight_layout()
            plt.savefig(os.path.join(prob_dir, f"class_{c}_{class_name}.png"))
            plt.close()
    
    # 2. Argmax distribution per class
    argmax_dir = os.path.join(output_dir, "Argmax_distribution_per_class")
    os.makedirs(argmax_dir, exist_ok=True)
    
    for class_data in per_class:
        c = class_data['class_index']
        class_name = class_data['class_name']
        argmax_norm_counts = class_data.get('argmax_norm_counts', None)
        
        if argmax_norm_counts is not None:
            plt.figure(figsize=(8, 5))
            ax = plt.gca()
            bars = ax.bar(class_names, argmax_norm_counts)
            plt.xticks(rotation=45)
            plt.xlabel("Classi Predette (Argmax)")
            plt.ylabel("Frazione di campioni")
            plt.title(f"Distribuzione argmax predetto - {class_name}")
            _annotate_bars_and_fix_ylim(ax, bars)
            plt.tight_layout()
            plt.savefig(os.path.join(argmax_dir, f"class_{c}_{class_name}.png"))
            plt.close()
    
    # 3. Accuracy per task
    if len(task_names) > 0 and len(accuracies) > 0:
        plt.figure(figsize=(8, 5))
        ax = plt.gca()
        bars = ax.bar(task_names, accuracies)
        ax.set_ylim(0, 1)
        for i, rect in enumerate(bars):
            v = rect.get_height()
            y = min(v + 0.02, 0.98)
            ax.text(rect.get_x() + rect.get_width() / 2.0, y, f"{v:.2f}", ha='center', va='bottom')
        plt.ylabel("Accuracy")
        plt.title("Accuracy per task")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "2_accuracy_per_task.png"))
        plt.close()
    
    # 4. Error distribution (normalized)
    if offset_matrix is not None and offset_ticks is not None:
        plt.figure(figsize=(12, 6))
        sns.heatmap(offset_matrix, annot=True, fmt=".2f",
                    xticklabels=offset_ticks,
                    yticklabels=class_names, cmap="Blues")
        plt.xlabel("Offset (Pred - Reale)")
        plt.ylabel("Classe Reale")
        plt.title("Distribuzione normalizzata degli errori (Task 0 - Age)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "3_error_distribution_with_prob.png"))
        plt.close()
    
    # 5. Average probability matrix
    if prob_matrix is not None:
        plt.figure(figsize=(10, 8))
        sns.heatmap(prob_matrix, annot=True, fmt=".2f",
                    xticklabels=class_names, yticklabels=class_names, cmap="YlOrRd")
        plt.xlabel("Classe Predetta")
        plt.ylabel("Classe Reale")
        plt.title("Probabilità media di scelta (Task 0 - Age)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "4_avg_prob_matrix.png"))
        plt.close()
    
    print(f"✅ Regenerated plots in: {output_dir}")
    return True


def regenerate_epoch_heatmap(npy_path, output_dir, class_names=None):
    """
    Regenerate heatmap from avg_probs_matrix.npy file.
    """
    try:
        prob_matrix = np.load(npy_path)
    except Exception as e:
        print(f"ERROR: Could not load {npy_path}: {e}")
        return False
    
    if prob_matrix.ndim != 2:
        print(f"WARNING: Expected 2D matrix, got shape {prob_matrix.shape}")
        return False
    
    # Extract epoch number from path
    epoch_num = "unknown"
    if "epoch_" in str(npy_path):
        try:
            epoch_num = str(npy_path).split("epoch_")[1].split("/")[0]
        except:
            pass
    
    # Use generic class names if not provided
    if class_names is None:
        num_classes = prob_matrix.shape[0]
        class_names = [f"Class_{i}" for i in range(num_classes)]
    
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(prob_matrix, annot=True, fmt=".2f",
                xticklabels=class_names, yticklabels=class_names, cmap="YlOrRd")
    plt.xlabel("Classe Predetta")
    plt.ylabel("Classe Reale")
    plt.title(f"Average Probability Matrix - Epoch {epoch_num}")
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, f"avg_prob_matrix_epoch_{epoch_num}.png")
    plt.savefig(output_file)
    plt.close()
    
    print(f"✅ Regenerated heatmap: {output_file}")
    return True


def generate_evolution_plots(base_dir, class_names=None):
    """
    Generate probability evolution plots from multiple epoch directories.
    """
    epoch_dirs = sorted([d for d in os.listdir(base_dir) if d.startswith("epoch_") and os.path.isdir(os.path.join(base_dir, d))])
    
    if not epoch_dirs:
        print(f"No epoch directories found in {base_dir}")
        return False
    
    print(f"Found {len(epoch_dirs)} epoch directories")
    
    history = []
    epochs = []
    
    for epoch_dir in epoch_dirs:
        npy_path = os.path.join(base_dir, epoch_dir, "avg_probs_matrix.npy")
        if os.path.exists(npy_path):
            try:
                prob_matrix = np.load(npy_path)
                history.append(prob_matrix)
                # Extract epoch number
                epoch_num = epoch_dir.replace("epoch_", "")
                epochs.append(int(epoch_num))
            except Exception as e:
                print(f"WARNING: Could not load {npy_path}: {e}")
                continue
    
    if not history:
        print("No valid probability matrices found")
        return False
    
    history = np.stack(history, axis=0)  # [epochs, num_classes, num_classes]
    num_classes = history.shape[1]
    
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(num_classes)]
    
    # Plot evolution of correct class probability
    plt.figure(figsize=(12, 8))
    for c in range(num_classes):
        correct_probs = history[:, c, c]  # Diagonal elements (correct predictions)
        plt.plot(epochs, correct_probs, marker='o', label=class_names[c])
    
    plt.xlabel("Epoch")
    plt.ylabel("Mean prob of correct class")
    plt.title("Evolution of correct class probability (Age)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    
    output_file = os.path.join(base_dir, "prob_evolution_correct.png")
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Generated evolution plot: {output_file}")
    return True


def scan_and_regenerate(train_folder):
    """
    Scan TRAIN folder for .npy files and regenerate plots.
    """
    train_path = Path(train_folder)
    if not train_path.exists():
        print(f"ERROR: Train folder does not exist: {train_folder}")
        return
    
    print(f"Scanning: {train_folder}")
    
    # Find all .npy files
    npy_files = list(train_path.rglob("*.npy"))
    
    if not npy_files:
        print("No .npy files found!")
        return
    
    print(f"Found {len(npy_files)} .npy files")
    
    # Process analysis_data.npy files
    analysis_files = [f for f in npy_files if f.name == "analysis_data.npy"]
    epoch_matrix_files = [f for f in npy_files if f.name == "avg_probs_matrix.npy"]
    
    print(f"- {len(analysis_files)} analysis_data.npy files")
    print(f"- {len(epoch_matrix_files)} avg_probs_matrix.npy files")
    
    # Process analysis_data.npy files
    for npy_file in analysis_files:
        output_dir = npy_file.parent
        regenerate_from_analysis_data(str(npy_file), str(output_dir))
    
    # Process avg_probs_matrix.npy files
    for npy_file in epoch_matrix_files:
        output_dir = npy_file.parent
        regenerate_epoch_heatmap(str(npy_file), str(output_dir))
    
    # Generate evolution plots for directories containing multiple epochs
    age_analysis_dirs = set()
    for npy_file in epoch_matrix_files:
        # Check if this is in an epoch_X subdirectory
        if "epoch_" in str(npy_file):
            age_analysis_dir = npy_file.parent.parent
            age_analysis_dirs.add(age_analysis_dir)
    
    for age_dir in age_analysis_dirs:
        generate_evolution_plots(str(age_dir))
    
    print(f"✅ Completed processing {train_folder}")


def main():
    parser = argparse.ArgumentParser(description="Regenerate plots from .npy analysis files")
    parser.add_argument("train_folder", help="Path to the TRAIN folder to scan")
    parser.add_argument("--recursive", "-r", action="store_true", 
                        help="Scan recursively for all .npy files")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.train_folder):
        print(f"ERROR: Path does not exist: {args.train_folder}")
        sys.exit(1)
    
    scan_and_regenerate(args.train_folder)


if __name__ == "__main__":
    main()