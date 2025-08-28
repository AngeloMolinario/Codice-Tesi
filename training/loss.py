import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class OrdinalAgeLossEMD_(nn.Module):
    '''
        This class compute the loss for the Age group classification task.
        The loss is made of 2 parts, a weighted cross-entropy as the main loss and a weighted Earth Mover's Distance (EMD) loss as the auxiliary loss.
        The CE is used because the problem is a multiclass classification problem where the main performance evaluator is the accuracy. Given the nature
        of the CE wrongly classify a sample of a 10-19 with 70+ hold the same weight of wrongly classy it as 20-29 but in age classification problem
        this is not true as the cost of misclassifying an age group can vary significantly. For example, misclassifying a 10-19 year old as 70+ is likely
        to be more detrimental than misclassifying them as 20-29. The EMD loss helps to address this issue by considering the ordinal nature of age groups.
    '''
    def __init__(self, num_classes=9, class_frequencies=None, lambda_ordinal=0.3, use_squared_emd=True):
        super(OrdinalAgeLossEMD, self).__init__()
        self.num_classes = num_classes
        self.lambda_ordinal = lambda_ordinal
        self.use_squared_emd = use_squared_emd
        self.inverse_factor = 1/math.log(num_classes)
        if class_frequencies is not None:
            self.class_weights = class_frequencies
        else:
            self.class_weights = torch.ones(num_classes).to('cuda')

        self.ce = nn.CrossEntropyLoss(weight=self.class_weights)
        self.softmax = nn.Softmax(dim=1)

    def compute_emd_loss(self, probs, targets, squared=False):
        """
        Compute Earth Mover's Distance between target and predicted distribution.
        
        Args:
            probs: Predicted probabilities (batch_size, num_classes)
            targets: Target class indices (batch_size,)
            squared: If True, use squared EMD (EMD²)
        
        Returns:
            EMD loss
        """
        
        target_dist = torch.zeros_like(probs)
        target_dist.scatter_(1, targets.unsqueeze(1), 1.0)
        
        pred_cdf = torch.cumsum(probs, dim=1)
        target_cdf = torch.cumsum(target_dist, dim=1)
        
        if squared:
            emd_loss = torch.sum((pred_cdf - target_cdf) ** 2, dim=1)
        else:
            emd_loss = torch.sum(torch.abs(pred_cdf - target_cdf), dim=1)
        
        return emd_loss
    
    def compute_weighted_emd_loss(self, probs, targets, squared=False):
        """
        Wighted version of EMD loss with class weights
        Args:
            squared: If True use squared EMD to penalize larger errors more heavily
        """
        device = probs.device
                
        target_dist = torch.zeros_like(probs)
        target_dist.scatter_(1, targets.unsqueeze(1), 1.0)
                
        pred_cdf = torch.cumsum(probs, dim=1)
        target_cdf = torch.cumsum(target_dist, dim=1)
        
    
        if squared:
            emd_losses = torch.sum((pred_cdf - target_cdf) ** 2, dim=1)
        else:
            emd_losses = torch.sum(torch.abs(pred_cdf - target_cdf), dim=1)
                
        sample_weights = self.class_weights[targets].to(device)
        weighted_emd_loss = torch.mean(sample_weights * emd_losses)
        
        return weighted_emd_loss
    
    def forward(self, predictions, targets):        
        ce_loss = self.ce(
            predictions, 
            targets            
        ) #* self.inverse_factor

        probs = self.softmax(predictions)

        weighted_emd_loss = self.compute_weighted_emd_loss(probs, targets, squared=self.use_squared_emd)
                
        if self.use_squared_emd:            
            normalized_emd_loss = weighted_emd_loss / ((self.num_classes - 1) ** 2)
        else:        
            normalized_emd_loss = weighted_emd_loss / (self.num_classes - 1)
        
        total_loss = ce_loss + self.lambda_ordinal * normalized_emd_loss

        return total_loss
    
import math
import torch
import torch.nn as nn

class OrdinalAgeLossEMD(nn.Module):
    """
    Loss per classificazione di gruppi d'età con regolarizzazione EMD^2 (Eq. 16).
    - Cross-entropy (principale)
    - Termine di regolarizzazione: sum_i p_i^2 * ( D_{i,k}^omega + mu )
    Riferimento: Hou et al., "Squared Earth Mover’s Distance-based Loss..." (Eq.16).
    """

    def __init__(self,
                 num_classes=9,
                 class_frequencies=None,
                 lambda_ordinal=0.5,
                 age_bins=("0-2","3-9","10-19","20-29","30-39","40-49","50-59","60-69","70+"),
                 plus_cap=90,          # limite superiore per "70+"
                 normalize_D=True,     # normalizza D su [0,1] dividendo per D.max()
                 omega=2.0,              # ω nell'Eq.16 (XEMD2 usa ω=2)
                 mu=-0.25             # μ nell'Eq.16 (paper: μ negativo per "premiare" vicini)
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_ordinal = float(lambda_ordinal)
        self.omega = float(omega)
        self.mu = float(mu)

        # pesi di classe per la CE
        if class_frequencies is not None:
            # Assumi già su device esterno; sposteremo in forward
            self.class_weights = torch.as_tensor(class_frequencies, dtype=torch.float32)
        else:
            self.class_weights = torch.ones(num_classes, dtype=torch.float32)

        self.softmax = nn.Softmax(dim=1)

        # ---- COSTRUISCI D (statica) ----
        self.register_buffer("D", self._build_D_from_age_bins(
            age_bins=age_bins, plus_cap=plus_cap, normalize=normalize_D
        ))  # shape [C, C]

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
        """
        Costruisce D come distanza tra i mid-point degli intervalli (1D embedding ordinato).
        D_{i,j} = | m_i - m_j |, con m_i midpoint dell'intervallo i.
        Opzionalmente normalizza dividendo per max(D) per avere scala in [0,1].
        """
        assert len(age_bins) == self.num_classes, "num_classes e numero di bin devono coincidere"
        mids = []
        for s in age_bins:
            lo, hi = self._parse_age_bin(s, plus_cap)
            mids.append((lo + hi) / 2.0)
        mids = torch.tensor(mids, dtype=torch.float32)  # [C]
        D = torch.abs(mids[:, None] - mids[None, :])    # [C, C]

        if normalize and D.max() > 0:
            D = D / D.max()

        return D  # buffer su CPU; verrà spostata con .to(device) in forward

    # --------- regolarizzazione Eq.16 ----------
    def _emd_regularizer_eq16(self, probs, targets):
        """
        Implementa esattamente il termine: sum_i p_i^2 * ( D_{i,k}^omega + mu )
        dove k è la classe vera per ciascun campione del batch.
        probs: (B, C) softmax
        targets: (B,)
        return: scalare medio sul batch
        """
        B, C = probs.shape
        # D_col[b, i] = D_{i, k_b}
        # Prendiamo per ogni elemento del batch la colonna della D corrispondente al target
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

        # NOTA: nel paper non normalizzano per (C-1) o (C-1)^2 nell'Eq.16.
        # Se vuoi mantenere la scala simile alla tua loss precedente, puoi agire su lambda_ordinal.
        total = ce + self.lambda_ordinal * reg
        return total



class CrossEntropyLoss():
    def __init__(self, num_classes, weights=None):
        self.ce = nn.CrossEntropyLoss(weight=weights)        
        self.softmax = nn.Softmax(dim=1)
        if weights is not None:
            self.class_weights = weights
        else:
            self.class_weights = torch.ones(num_classes).to('cuda')
        self.factor = math.log(num_classes)

    def __call__(self, logit, true_labels):
        loss = self.ce(logit, true_labels) # / self.factor

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