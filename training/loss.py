import torch
from torch.nn import functional as F
import math
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence, Optional


import torch
import torch.nn as nn
import torch.nn.functional as F

class OrdinalAgeLoss(nn.Module):
    def __init__(self, num_classes=9, class_frequencies=None, lambda_ordinal=0.3):
        super(OrdinalAgeLoss, self).__init__()
        self.num_classes = num_classes
        self.lambda_ordinal = lambda_ordinal
        
        # Calcolo dei pesi per gestire lo sbilanciamento
        if class_frequencies is not None:
            total_samples = sum(class_frequencies)
            self.class_weights = torch.tensor([
                total_samples / (num_classes * max(freq, 1)) 
                for freq in class_frequencies
            ]).to('cuda')
        else:
            self.class_weights = torch.ones(num_classes).to('cuda')

    def forward(self, predictions, targets, return_predicted_label=False):
        # 1. Weighted Cross-Entropy
        ce_loss = F.cross_entropy(
            predictions, 
            targets, 
            weight=self.class_weights.to(predictions.device)
        )
        
        # 2. Weighted Ordinal Loss (CORRETTO)
        batch_size = predictions.size(0)
        indices = torch.arange(self.num_classes, device=predictions.device, dtype=predictions.dtype)
        
        # Probabilità normalizzate
        probs = F.softmax(predictions, dim=1)
        expected_index = torch.sum(probs * indices, dim=1)
        
        # Calcolo gli errori ordinali PESATI
        target_indices = targets.float()
        ordinal_errors = (expected_index - target_indices) ** 2  # Errore quadratico per esempio
        
        # Applico i pesi specifici per ogni esempio in base alla sua classe
        sample_weights = self.class_weights[targets].to(predictions.device)
        weighted_ordinal_loss = torch.mean(sample_weights * ordinal_errors)
        normalized_ordinal_error = weighted_ordinal_loss / ((self.num_classes - 1) ** 2)
        # 3. Combinazione finale
        total_loss = ce_loss + self.lambda_ordinal * normalized_ordinal_error

        return total_loss, predictions.argmax(dim=1)


class CrossEntropyLoss():
    def __init__(self, weights=None):
        self.ce = nn.CrossEntropyLoss(weight=weights)        
        self.softmax = nn.Softmax(dim=1)
        self.class_weights = weights
        self.factor = math.log(len(weights)) if weights is not None else 1.0

    def __call__(self, logit, true_labels, return_predicted_label=False):
        loss = self.ce(logit, true_labels)  / self.factor

    
        if return_predicted_label:            
            return loss, logit.argmax(dim=1)
        return loss


class MaskedLoss():    
    def __init__(self, base_loss, ignore_index=-1):        
        self.base_loss = base_loss
        self.ignore_index = ignore_index

    def __call__(self, logit, true_labels, return_predicted_label=False):                
        valid_mask = true_labels != self.ignore_index
        
        if not valid_mask.any():
            loss = torch.tensor(0.0, device=logit.device, requires_grad=True)
            if return_predicted_label:                
                predicted_labels = torch.full_like(true_labels, self.ignore_index)
                return loss, predicted_labels
            return loss
        
        valid_logits = logit[valid_mask]
        valid_labels = true_labels[valid_mask]

        if return_predicted_label:
            loss, valid_predicted_labels = self.base_loss(valid_logits, valid_labels, return_predicted_label=True)
            
            full_predicted_labels = torch.full_like(true_labels, self.ignore_index)
            full_predicted_labels[valid_mask] = valid_predicted_labels
            
            return loss, full_predicted_labels
        else:
            loss = self.base_loss(valid_logits, valid_labels, return_predicted_label=False)
            return loss