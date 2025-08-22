import torch
from torch.nn import functional as F
import math
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence, Optional

class CrossEntropyMAELoss(nn.Module):
    """
    Combina CrossEntropy e MAE tra:
      E[p] = sum_i softmax(scale * logits)_i * bin_centers[i]
    e il centro del bin della classe target.

    Loss = (1 - alpha) * CE(scale*logits, target) + alpha * w_y * |E[p] - c_y|

    Args:
        bin_centers: lista/seq dei centri di bin (lunghezza = num classi)
        scale: fattore di scala (temperature inversa) applicato ai logits
        alpha: peso tra CE e MAE (0=solo CE, 1=solo MAE)
        class_weights: pesi per le classi (tensor [C], opzionale)
        reduction: 'mean' | 'sum' | 'none'
    """
    def __init__(
        self,
        bin_centers: Sequence[float],
        scale: float = 1.0,
        alpha: float = 0.5,
        class_weights: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("`alpha` deve essere in [0, 1].")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError("`reduction` deve essere 'mean', 'sum' o 'none'.")

        self.register_buffer("bin_centers", torch.as_tensor(bin_centers, dtype=torch.float32))
        self.scale = float(scale)
        self.alpha = float(alpha)
        self.reduction = reduction

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.to(torch.float32))
        else:
            self.class_weights = None

        # CE senza riduzione (gestiamo noi la combinazione finale)
        self.ce = nn.CrossEntropyLoss(weight=class_weights, reduction="none")

    def forward(self, logits: torch.Tensor, target: torch.Tensor, return_predicted_label: bool = False):
        """
        logits: [N, C]
        target: [N] (long)
        """
        if logits.dim() != 2:
            raise ValueError("`logits` deve avere shape [N, C].")
        N, C = logits.shape
        if C != self.bin_centers.numel():
            raise ValueError(f"num classi ({C}) != num bin_centers ({self.bin_centers.numel()}).")

        # 1) CrossEntropy (con scale applicato ai logits)
        z = logits * self.scale
        ce_loss = self.ce(z, target)  # [N]

        # 2) Valore atteso con softmax(scale * logits)
        probs = F.softmax(z, dim=-1)                         # [N, C]
        expected_value = torch.sum(probs * self.bin_centers, dim=-1)  # [N]

        # 3) Centro del bin target
        target_centers = self.bin_centers[target]            # [N]

        # 4) MAE
        mae_loss = torch.abs(expected_value - target_centers)  # [N]

        # 5) Se ci sono pesi di classe, applicali anche al MAE
        if self.class_weights is not None:
            mae_loss = mae_loss * self.class_weights[target]

        # 6) Combinazione
        loss = (1.0 - self.alpha) * ce_loss + self.alpha * mae_loss  # [N]

        return loss, logits.argmax(dim=1)


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