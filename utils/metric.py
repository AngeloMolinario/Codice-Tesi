import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import os
import torch

class TrainingMetrics:
    """
    A class to track and plot training and validation metrics.
    """
    def __init__(self, class_names, output_dir='results'):
        """
        Initializes the TrainingMetrics tracker.

        Args:
            class_names (list): A list of strings representing the class names for the confusion matrix.
            output_dir (str): The directory where plots will be saved.
        """
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.class_names = class_names
        self.output_dir = output_dir
        self.all_preds = []
        self.all_labels = []

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def update_train_metrics(self, loss, accuracy):
        """Adds new training loss and accuracy to the tracker."""
        self.train_losses.append(loss)
        self.train_accuracies.append(accuracy)

    def update_val_metrics(self, loss, accuracy):
        """Adds new validation loss and accuracy to the tracker."""
        self.val_losses.append(loss)
        self.val_accuracies.append(accuracy)

    def update_predictions(self, preds, labels):
        """Collects predictions and labels for the confusion matrix."""
        self.all_preds.extend(preds.cpu().numpy())
        self.all_labels.extend(labels.cpu().numpy())

    def reset_predictions(self):
        """Resets the collected predictions and labels."""
        self.all_preds = []
        self.all_labels = []

    def plot_metrics(self):
        """Plots and saves the training/validation loss and accuracy curves."""
        epochs = range(1, len(self.train_losses) + 1)
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.train_losses, 'b-o', label='Training Loss')
        plt.plot(epochs, self.val_losses, 'r-o', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.train_accuracies, 'b-o', label='Training Accuracy')
        plt.plot(epochs, self.val_accuracies, 'r-o', label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_plots.png'))
        plt.close()

    def plot_confusion_matrix(self, epoch='final'):
        """Plots and saves the confusion matrix."""
        if not self.all_labels or not self.all_preds:
            print("No predictions to plot confusion matrix.")
            return
            
        cm = confusion_matrix(self.all_labels, self.all_preds)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title(f'Confusion Matrix - Epoch {epoch}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig(os.path.join(self.output_dir, f'confusion_matrix_epoch_{epoch}.png'))
        plt.close()