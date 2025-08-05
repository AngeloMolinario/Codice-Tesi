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
    def __init__(self, class_names, output_dir='results', is_multitask=False, task_names=None):
        """
        Initializes the TrainingMetrics tracker.

        Args:
            class_names (list): A list of strings representing the class names for the confusion matrix.
                              For multitask: list of lists, where each sublist contains class names for each task.
                              For single task: simple list of strings.
            output_dir (str): The directory where plots will be saved.
            is_multitask (bool): Whether this is for multitask learning.
            task_names (list): Names of the tasks (e.g., ['age', 'gender', 'emotion']) for multitask.
        """
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.class_names = class_names
        self.output_dir = output_dir
        self.is_multitask = is_multitask
        self.task_names = task_names if task_names else []
        
        if is_multitask:
            # For multitask: list of lists for each task
            self.all_preds = [[] for _ in range(len(task_names))]
            self.all_labels = [[] for _ in range(len(task_names))]
        else:
            # For single task: simple lists
            self.all_preds = []
            self.all_labels = []

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def update_train_metrics(self, loss, accuracy):
        """Adds new training loss and accuracy to the tracker."""
        # For both single and multitask, we typically track the overall/main metric
        if isinstance(loss, (list, np.ndarray)):
            self.train_losses.append(loss[0])  # Take the first (overall) loss
        else:
            self.train_losses.append(loss)
            
        if isinstance(accuracy, (list, np.ndarray)):
            self.train_accuracies.append(accuracy[0])  # Take the first (overall) accuracy
        else:
            self.train_accuracies.append(accuracy)

    def update_val_metrics(self, loss, accuracy):
        """Adds new validation loss and accuracy to the tracker."""
        # For both single and multitask, we typically track the overall/main metric
        if isinstance(loss, (list, np.ndarray)):
            self.val_losses.append(loss[0])  # Take the first (overall) loss
        else:
            self.val_losses.append(loss)
            
        if isinstance(accuracy, (list, np.ndarray)):
            self.val_accuracies.append(accuracy[0])  # Take the first (overall) accuracy
        else:
            self.val_accuracies.append(accuracy)

    def update_predictions(self, preds, labels, task_idx=None):
        """
        Collects predictions and labels for the confusion matrix.
        
        Args:
            preds: Predictions tensor or list of tensors (for multitask).
            labels: Labels tensor or list of tensors (for multitask).
            task_idx: Index of the task (for multitask). If None, assumes single task.
        """
        if self.is_multitask:
            if task_idx is None:
                raise ValueError("task_idx must be provided for multitask learning")
            
            # Filter out invalid labels (-1) for multitask
            preds_np = preds.cpu().numpy()
            labels_np = labels.cpu().numpy()
            valid_mask = labels_np != -1
            
            if valid_mask.sum() > 0:
                self.all_preds[task_idx].extend(preds_np[valid_mask])
                self.all_labels[task_idx].extend(labels_np[valid_mask])
        else:
            # Single task behavior
            self.all_preds.extend(preds.cpu().numpy())
            self.all_labels.extend(labels.cpu().numpy())

    def reset_predictions(self, task_idx=None):
        """
        Resets the collected predictions and labels.
        
        Args:
            task_idx: Index of the task to reset (for multitask). If None, resets all.
        """
        if self.is_multitask:
            if task_idx is None:
                # Reset all tasks
                for i in range(len(self.task_names)):
                    self.all_preds[i] = []
                    self.all_labels[i] = []
            else:
                self.all_preds[task_idx] = []
                self.all_labels[task_idx] = []
        else:
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

    def plot_confusion_matrix(self, epoch='final', task_idx=None):
        """
        Plots and saves the confusion matrix.
        
        Args:
            epoch: Epoch identifier for the filename.
            task_idx: Index of the task to plot (for multitask). If None, plots all tasks.
        """
        if self.is_multitask:
            if task_idx is None:
                # Plot confusion matrix for all tasks
                for i, task_name in enumerate(self.task_names):
                    self._plot_single_confusion_matrix(i, task_name, epoch)
            else:
                # Plot confusion matrix for specific task
                if task_idx < len(self.task_names):
                    self._plot_single_confusion_matrix(task_idx, self.task_names[task_idx], epoch)
        else:
            # Single task behavior
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
    
    def _plot_single_confusion_matrix(self, task_idx, task_name, epoch):
        """Helper method to plot confusion matrix for a single task."""
        if not self.all_labels[task_idx] or not self.all_preds[task_idx]:
            print(f"No predictions to plot confusion matrix for task {task_name}.")
            return
            
        cm = confusion_matrix(self.all_labels[task_idx], self.all_preds[task_idx])
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.class_names[task_idx], 
                    yticklabels=self.class_names[task_idx])
        plt.title(f'Confusion Matrix - {task_name.capitalize()} - Epoch {epoch}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig(os.path.join(self.output_dir, f'{task_name}_confusion_matrix_epoch_{epoch}.png'))
        plt.close()