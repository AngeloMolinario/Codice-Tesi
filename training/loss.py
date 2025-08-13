import torch
from torch import nn

import math
import torch
import torch.nn as nn

class OrdinalLoss(nn.Module):
    def __init__(self, num_classes, ordinal_loss='mae', weights=None, alpha=0.05):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.classes_range = torch.arange(num_classes).unsqueeze(0).float().to(device)  # [K]

        if weights is not None:
            self.class_weights = weights.float().to(device)
        else:
            self.class_weights = torch.ones(num_classes).to(device)

        self.ce = nn.CrossEntropyLoss(weight=self.class_weights)

    def forward(self, logit, true_labels, return_predicted_label=False):
        K = self.num_classes
        probs = torch.softmax(logit, dim=1)                  # [B,K]

        # CE normalizzata
        ce_loss = self.ce(logit, true_labels) / math.log(K)

        # Expected Absolute Error (EAE) normalizzata
        # dist[k] = |k - y|  -> shape [B,K]
        dist = (self.classes_range - true_labels.unsqueeze(1).float()).abs()
        eae = (probs * dist).sum(dim=1) / (K - 1)            # [B] in [0,1]

        # Pesi per classe anche sulla parte ordinale (opzionale)
        eae = eae * self.class_weights[true_labels]
        ord_loss = eae.mean()

        loss = ce_loss + self.alpha * ord_loss
        #print(f"Ordinal loss: {ord_loss.item():.4f} - CE_loss : {ce_loss.item():.4f} - Total loss: {loss.item():.4f}")        
        if return_predicted_label:
            ev = (probs * self.classes_range).sum(dim=1)
            pred = torch.clamp(torch.round(ev), 0, K-1).long()
            return loss, torch.argmax(probs, dim=1)
        return loss


class CrossEntropyLoss():
    def __init__(self, weights=None):
        self.ce = nn.CrossEntropyLoss(weight=weights)        
        self.softmax = nn.Softmax(dim=1)
        self.class_weights = weights
        self.factor = math.log(len(weights)) if weights is not None else 1.0

    def __call__(self, logit, true_labels, return_predicted_label=False):
        loss = self.ce(logit, true_labels) / self.factor

    
        if return_predicted_label:
            probabilities = self.softmax(logit)
            predicted_label = torch.argmax(probabilities, dim=1)
            return loss, predicted_label
        return loss

class BCELoss():
    def __init__(self, weights=None):
        self.bce = nn.BCEWithLogitsLoss(weight=weights)

    def __call__(self, logit, true_labels, return_predicted_label=False):
        loss = self.bce(logit, true_labels.float())

        if return_predicted_label:
            predicted_label = 0 if torch.sigmoid(logit) < 0.5 else 1
            return loss, predicted_label
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