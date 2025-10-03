import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class OrdinalAgeLossEMD(nn.Module):
    """
    Loss per classificazione di gruppi d'età con regolarizzazione EMD^2 (Eq. 16).
    - Cross-entropy (principale)
    - Termine di regolarizzazione: sum_i p_i^2 * ( D_{i,k}^omega + mu )
    Riferimento: Hou et al., "Squared Earth Mover's Distance-based Loss..." (Eq.16).
    """

    def __init__(self,
                 num_classes=9,
                 class_frequencies=None,
                 lambda_ordinal=0.5,
                 age_bins=("0-2","3-9","10-19","20-29","30-39","40-49","50-59","60-69","70+"),
                 plus_cap=90,          # limite superiore per "70+"
                 normalize_D=True,     # normalizza D su [0,1] dividendo per D.max()
                 omega=3.0,              # ω nell'Eq.16 (XEMD2 usa ω=2)
                 mu=-0.0             # μ nell'Eq.16 (paper: μ negativo per "premiare" vicini)
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_ordinal = float(lambda_ordinal)
        self.omega = float(omega)
        self.mu = float(mu)

        if class_frequencies is not None:
            self.class_weights = torch.as_tensor(class_frequencies, dtype=torch.float32)
        else:
            self.class_weights = torch.ones(num_classes, dtype=torch.float32)

        self.softmax = nn.Softmax(dim=1)
        self.factor = math.log(num_classes)

        self.register_buffer("D", self._build_D_from_age_bins(
            age_bins=age_bins, plus_cap=plus_cap, normalize=normalize_D
        )) 

    # --------- costruzione D ----------
    @staticmethod
    def _parse_age_bin(bin_str, plus_cap):
        if "+" in bin_str:
            lo = int(bin_str.replace("+", ""))
            hi = plus_cap
        else:
            a, b = bin_str.split("-")
            lo, hi = int(a), int(b)
        return lo, hi

    def _build_D_from_age_bins(self, age_bins, plus_cap=90, normalize=True):
        mids = []
        for s in age_bins:
            lo, hi = self._parse_age_bin(s, plus_cap)
            mids.append((lo + hi) / 2.0)
        mids = torch.tensor(mids, dtype=torch.float32)  # [C]
        D = torch.abs(mids[:, None] - mids[None, :])    # [C, C]

        if normalize and D.max() > 0:
            D = D / D.max()

        return D 

    # --------- regolarizzazione Eq.16 ----------
    def _emd_regularizer_eq16(self, probs, targets):
        """
        Implementa esattamente il termine: sum_i p_i^2 * ( D_{i,k}^omega + mu )
        dove k è la classe vera per ciascun campione del batch.
        probs: (B, C) softmax
        targets: (B,)
        return: scalare medio sul batch
        """
        
        Ddevice = self.D.to(probs.device)
        D_cols = Ddevice.index_select(dim=1, index=targets)          # [C, B]
        D_cols = D_cols.transpose(0, 1)                               # [B, C]
        term = torch.pow(D_cols.clamp(min=0), self.omega) + self.mu   # [B, C]
        reg_per_sample = torch.sum((probs ** 2) * term, dim=1)        # [B]
        return torch.mean(reg_per_sample)                             # scalare

    def forward(self, predictions, targets):
        """
        predictions: logits (B, C)
        targets: indices (B,)
        """
        device = predictions.device

        # CE con pesi di classe sul device
        ce = nn.functional.cross_entropy(
            predictions, targets,
            weight=self.class_weights.to(device),
            reduction='mean'
        )

        probs = self.softmax(predictions)

        # Termine di regolarizzazione Eq.16
        reg = self._emd_regularizer_eq16(probs, targets)

        #print(f"CE Loss: {ce.item():.4f}, EMD Reg: {reg.item():.4f}, total_loss_weighted: {self.lambda_ordinal*reg:.4f} - Total {ce.item() + self.lambda_ordinal*reg:.4f}")

        total = ce + self.lambda_ordinal * reg
        return total



class CrossEntropyLoss():
    def __init__(self, num_classes, weights=None):
        self.ce = nn.CrossEntropyLoss(weight=weights)        
        self.factor = math.log(num_classes)

    def __call__(self, logit, true_labels):
        loss = self.ce(logit, true_labels) 

        return loss


class MaskedLoss():    
    def __init__(self, base_loss, ignore_index=-1):        
        self.base_loss = base_loss
        self.ignore_index = ignore_index

    def __call__(self, logit, true_labels):                
        valid_mask = true_labels != self.ignore_index
        if not valid_mask.any():
            loss = torch.tensor(0.0, device=logit.device, requires_grad=True)

            return loss
        
        valid_logits = logit[valid_mask]
        valid_labels = true_labels[valid_mask]

        loss = self.base_loss(valid_logits, valid_labels)
        return loss