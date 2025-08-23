import torch
from torch.nn import functional as F
import math
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence, Optional

class CrossEntropyOrdinalLoss_(nn.Module):
    def __init__(self, bin_centers, scale=1.0, beta=0.5, class_weights=None, reduction="mean", p=1):
        super().__init__()
        self.register_buffer("bin_centers", torch.as_tensor(bin_centers, dtype=torch.float32))
        self.scale = float(scale)
        self.beta  = float(beta)
        self.p     = int(p)  # 1 = L1 (Wasserstein-1), 2 = L2
        self.reduction = reduction
        self.class_weights = None if class_weights is None else class_weights.to(torch.float32)
        self.ce = nn.CrossEntropyLoss(weight=self.class_weights, reduction="none")

    def forward(self, logits, target, return_predicted_label=False):
        N, C = logits.shape
        z = logits * self.scale
        ce_loss = self.ce(z, target)                      # [N]
        probs   = F.softmax(z, dim=-1)                    # [N, C]

        # distanza ordinale tra ciascun bin e il target (in spazio dei centri)
        c_y   = self.bin_centers[target].view(-1, 1)      # [N, 1]
        dist  = (torch.abs(self.bin_centers.view(1, -1) - c_y))  # [N, C]
        if self.p == 2:
            dist = dist ** 2

        ord_pen = (probs * dist).sum(dim=-1)              # E[ |c_k - c_y|^p ]   [N]

        if self.class_weights is not None:
            ord_pen = ord_pen * self.class_weights[target]

        loss = ce_loss + self.beta * ord_pen              # [N]
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return (loss, logits.argmax(dim=1)) if return_predicted_label else loss


import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyOrdinalLoss(nn.Module):
    def __init__(self, bin_centers, scale=1.0, beta=0.1, class_weights=None,
                 reduction="mean", p=1, normalize_dist=True):
        super().__init__()
        self.register_buffer("bin_centers", torch.as_tensor(bin_centers, dtype=torch.float32))
        self.scale = float(scale)
        self.beta = float(beta)
        self.p = int(p)                 # 1 = L1 (consigliato), 2 = L2
        self.normalize_dist = bool(normalize_dist)
        self.reduction = reduction

        self.class_weights = None if class_weights is None else class_weights.to(torch.float32)
        self.ce = nn.CrossEntropyLoss(weight=self.class_weights, reduction="none")

    def forward(self, logits, target, return_predicted_label: bool = False):
        # CE (massimizza accuracy)
        z = logits * self.scale
        ce_loss = self.ce(z, target)            # [N]

        # Penalità ordinale (distanza dal centro del bin target)
        probs = F.softmax(z, dim=-1)            # [N, C]
        c = self.bin_centers                    # [C]
        c_y = c[target].unsqueeze(1)            # [N, 1]
        dist = (c.view(1, -1) - c_y).abs()      # [N, C]

        if self.normalize_dist:
            span = (c.max() - c.min()).clamp_min(1e-6)
            dist = dist / span                  # ora in ~[0, 1]

        if self.p == 2:
            dist = dist ** 2

        ord_pen = (probs * dist).sum(dim=1)     # E[|c_k - c_y|^p]  [N]

        if self.class_weights is not None:
            ord_pen = ord_pen * self.class_weights[target]

        loss = ce_loss + self.beta * ord_pen

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return (loss, logits.argmax(dim=1)) if return_predicted_label else loss


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