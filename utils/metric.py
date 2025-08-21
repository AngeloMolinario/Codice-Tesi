import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class MultitaskTracker:
    def __init__(self, num_tasks, output_dir, task_names, class_names):
        self.num_tasks = num_tasks
        self.output_dir = output_dir
        self.task_names = task_names
        self.class_names = class_names

        self.loss = {'train': [[] for _ in range(num_tasks)], 'val': [[] for _ in range(num_tasks)], 'multitask': {'train': [], 'val': []}}
        self.accuracy = {'train': [[] for _ in range(num_tasks)], 'val': [[] for _ in range(num_tasks)], 'mean': {'train': [], 'val': []}}
        self.confusion_data = {'val': [[] for _ in range(num_tasks)]}  # Per ogni task, lista di tuple (pred, label) per ogni epoca

        os.makedirs(os.path.join(output_dir, "cm"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "plot"), exist_ok=True)

    def update_loss(self, task_idx, value, train=True, multitask=False):
        key = 'train' if train else 'val'
        if multitask:
            self.loss['multitask'][key].append(value)
        else:
            self.loss[key][task_idx].append(value)

    def update_accuracy(self, task_idx, value, train=True, mean=False):
        key = 'train' if train else 'val'
        if mean:
            self.accuracy['mean'][key].append(value)
        else:
            self.accuracy[key][task_idx].append(value)

    def update_confusion(self, task_idx, preds, labels, epoch):
        # preds, labels: torch.Tensor or np.ndarray
        preds = preds.cpu().numpy() if torch.is_tensor(preds) else np.array(preds)
        labels = labels.cpu().numpy() if torch.is_tensor(labels) else np.array(labels)
        while len(self.confusion_data['val'][task_idx]) <= epoch:
            self.confusion_data['val'][task_idx].append(([], []))
        self.confusion_data['val'][task_idx][epoch] = (preds, labels)

    def save(self, path=None):
        # Save all metrics and confusion data
        save_path = path or os.path.join(self.output_dir, "multitask_tracker_history.json")
        data = {
            'loss': self.loss,
            'accuracy': self.accuracy,
            'confusion_data': {
                k: [
                    [
                        (p.tolist() if hasattr(p, 'tolist') else p, l.tolist() if hasattr(l, 'tolist') else l)
                        for (p, l) in (v if isinstance(v, list) else [v])
                    ]
                    for v in self.confusion_data[k]
                ]
                for k in self.confusion_data
            },
            'task_names': self.task_names,
            'class_names': self.class_names
        }
        with open(save_path, 'w') as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            data = json.load(f)
        tracker = cls(
            num_tasks=len(data['task_names']),
            output_dir=os.path.dirname(path),
            task_names=data['task_names'],
            class_names=data['class_names']
        )
        tracker.loss = data['loss']
        tracker.accuracy = data['accuracy']
        # Convert confusion_data back to numpy arrays
        for k in data['confusion_data']:
            tracker.confusion_data[k] = []
            for task_list in data['confusion_data'][k]:
                tracker.confusion_data[k].append([(np.array(p), np.array(l)) for (p, l) in task_list])
        return tracker

    def save_confusion_matrices(self, epoch):
        cm_dir = os.path.join(self.output_dir, "cm")
        for task_idx, task_name in enumerate(self.task_names):
            if len(self.confusion_data['val'][task_idx]) > epoch:
                preds, labels = self.confusion_data['val'][task_idx][epoch]
                if len(preds) == 0 or len(labels) == 0:
                    continue
                cm = confusion_matrix(labels, preds, labels=range(len(self.class_names[task_idx])))
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.class_names[task_idx])
                fig, ax = plt.subplots(figsize=(6, 6))
                disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
                plt.title(f"Confusion Matrix - {task_name} - Epoch {epoch+1}")
                plt.tight_layout()
                plt.savefig(os.path.join(cm_dir, f"{task_name}_epoch{epoch+1}.png"))
                plt.close(fig)

    def plot_losses(self):
        plot_dir = os.path.join(self.output_dir, "plot")
        epochs = range(1, len(self.loss['train'][0]) + 1)
        # All tasks in one plot
        plt.figure(figsize=(10, 6))
        for i, task_name in enumerate(self.task_names):
            plt.plot(epochs, self.loss['train'][i], label=f"{task_name} Train")
            plt.plot(epochs, self.loss['val'][i], '--', label=f"{task_name} Val")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss per Task")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "all_tasks_loss.png"))
        plt.close()

        # Separate plots
        for i, task_name in enumerate(self.task_names):
            plt.figure()
            plt.plot(epochs, self.loss['train'][i], label="Train")
            plt.plot(epochs, self.loss['val'][i], label="Val")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"Loss - {task_name}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f"{task_name}_loss.png"))
            plt.close()