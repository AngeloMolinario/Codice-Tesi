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
        
        # 3. Combinazione finale
        total_loss = ce_loss + self.lambda_ordinal * weighted_ordinal_loss

        return total_loss, predictions.argmax(dim=1)

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


class CrossEntropyOrdinalLoss2(nn.Module):
    """
    Loss =  CE_pesata_focale
            + beta_ord * E[|c_k - c_y|^p]   (penalità ordinale "soft")
            + lambda_rank * L_rank          (hinge ranking ordinale)

    Dove:
      - CE_pesata_focale = CE pesata per classe * (1 - p_t)^gamma  (gamma>=0)
      - L_rank = sum_k ReLU( m_k - (z_y - z_k) ), con m_k proporzionale alla distanza tra bin
    """
    def __init__(
        self,
        bin_centers,
        scale: float = 1.0,
        beta_ord: float = 0.1,        # peso penalità ordinale "soft"
        p: int = 1,                   # 1 = L1 (consigliato), 2 = L2
        normalize_dist: bool = True,
        class_weights: torch.Tensor | None = None,   # pesi per classe
        gamma: float = 0.0,           # focal focusing (0 = disabilitato)
        lambda_rank: float = 0.0,     # peso ranking ordinale (0 = disabilitato)
        margin: float = 0.5,          # margine base per il ranking
        rank_power: int = 1,          # esponente sulla distanza nel margine
        reduction: str = "mean",
        only_adjacent: bool = False,  # True = ranking solo con classi adiacenti
    ):
        super().__init__()
        self.register_buffer("bin_centers", torch.as_tensor(bin_centers, dtype=torch.float32))
        self.scale = float(scale)

        self.beta_ord = float(beta_ord)
        self.p = int(p)
        self.normalize_dist = bool(normalize_dist)

        # Pesi di classe come buffer (così vanno sul device giusto)
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.to(torch.float32))
        else:
            self.class_weights = None

        # Parametri focal e ranking
        self.gamma = float(gamma)
        self.lambda_rank = float(lambda_rank)
        self.margin = float(margin)
        self.rank_power = int(rank_power)
        self.only_adjacent = bool(only_adjacent)

        assert reduction in ("none", "mean", "sum")
        self.reduction = reduction

    def _reduce(self, x):
        if self.reduction == "mean":
            return x.mean()
        elif self.reduction == "sum":
            return x.sum()
        return x

    def forward(self, logits, target, return_predicted_label: bool = False):
        """
        logits: [N, C]
        target: [N] (long)
        """
        # 1) CE pesata (senza riduzione, calcolata a mano per poter applicare il focal term)
        z = logits * self.scale                         # [N, C]
        log_probs = F.log_softmax(z, dim=-1)            # [N, C]
        n, c = z.shape
        idx = torch.arange(n, device=z.device)

        ce_per_sample = -log_probs[idx, target]         # [N] CE "base"
        if self.class_weights is not None:
            ce_per_sample = ce_per_sample * self.class_weights[target]

        # 1b) Focal term opzionale per accentuare il picco della classe vera
        #     CE_focale = CE * (1 - p_t)^gamma
        if self.gamma > 0.0:
            probs = log_probs.exp()                     # [N, C]
            pt = probs[idx, target].clamp_min(1e-12)    # [N]
            focal_factor = (1.0 - pt) ** self.gamma
            ce_per_sample = ce_per_sample * focal_factor

        ce_loss = ce_per_sample                         # [N]

        # 2) Penalità ordinale "soft" (aspettativa della distanza)
        probs = F.softmax(z, dim=-1)                    # [N, C]
        c_vec = self.bin_centers                        # [C]
        c_y = c_vec[target].unsqueeze(1)                # [N, 1]
        dist = (c_vec.view(1, -1) - c_y).abs()          # [N, C]

        if self.normalize_dist:
            span = (c_vec.max() - c_vec.min()).clamp_min(1e-6)
            dist = dist / span                          # ~[0,1]

        if self.p == 2:
            dist = dist ** 2

        ord_pen = (probs * dist).sum(dim=1)             # [N]

        if self.class_weights is not None:
            ord_pen = ord_pen * self.class_weights[target]

        ord_loss = self.beta_ord * ord_pen              # [N]

        # 3) Ranking ordinale (hinge) per aumentare il gap della classe vera
        #    L_rank_i = sum_k ReLU( m_k - (z_y - z_k) )
        #    con m_k = margin * (|c_k - c_y|/span)^rank_power  (o non normalizzato)
        rank_loss = torch.zeros_like(ce_loss)
        if self.lambda_rank > 0.0:
            z_y = z[idx, target].unsqueeze(1)           # [N, 1]
            # distanza per il margine
            d = (c_vec.view(1, -1) - c_y).abs()         # [N, C]
            if self.normalize_dist:
                span = (c_vec.max() - c_vec.min()).clamp_min(1e-6)
                d = d / span

            if self.rank_power == 2:
                d = d ** 2

            margin_mat = self.margin * d                # [N, C]
            # maschera per escludere la classe vera
            mask_not_y = torch.ones((n, c), dtype=torch.bool, device=z.device)
            mask_not_y[idx, target] = False

            # opzionale: confronta solo con le classi adiacenti (y-1, y+1)
            if self.only_adjacent:
                mask_adj = torch.zeros_like(mask_not_y)
                y = target
                if c > 1:
                    # precedente
                    prev_idx = (y - 1).clamp_min(0)
                    # successiva
                    next_idx = (y + 1).clamp_max(c - 1)
                    mask_adj[idx, prev_idx] = True
                    mask_adj[idx, next_idx] = True
                    # evita di includere y se y-1==y o y+1==y ai bordi
                    mask_adj[idx, y] = False
                mask_not_y = mask_adj

            # hinge: ReLU( m_k - (z_y - z_k) ) = ReLU( m_k + z_k - z_y )
            # calcolo per tutte le classi e poi maschero
            hinge_all = (margin_mat + z - z_y).clamp_min(0.0)   # [N, C]
            hinge_all[~mask_not_y] = 0.0
            rank_per_sample = hinge_all.sum(dim=1)              # [N]

            if self.class_weights is not None:
                rank_per_sample = rank_per_sample * self.class_weights[target]

            rank_loss = self.lambda_rank * rank_per_sample      # [N]

        # 4) Loss totale
        loss_per_sample = ce_loss + ord_loss + rank_loss
        loss = self._reduce(loss_per_sample)

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
        loss = self.ce(logit, true_labels) # / self.factor

    
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