import torch
from torch import nn
from torch.nn import functional as F
import math
import torch
import torch.nn as nn


class OrdinalPeakedCELoss(nn.Module):
    """
    Loss = CE_norm
         + alpha * [ w_far   * FarProbMargin
                     w_tail  * TailWeighted
                     w_lpeak * LocalPeak
                     w_emd   * EMD^2 ]

    - CE_norm           : massimizza l'accuracy
    - FarProbMargin     : vieta errori lontani (|k-y|>1) in probabilità
    - TailWeighted      : localizza la massa fuori ±1 con peso crescente
    - LocalPeak         : gap p[y] ≥ max(p[y±1]) + margin
    - EMD^2 (CDF)       : ordinalità globale
    - widths (opz.)     : pesi per EMD^2 quando i bin hanno ampiezze diverse
    """
    def __init__(self, num_classes, weights=None, widths=[3, 7, 10, 10, 10, 10, 10, 10, 25],
                 alpha=0.5,
                 w_far=1.0,      delta_far=0.15,
                 w_lpeak=1.0,    prob_margin=0.2,
                 w_emd=1.0,
                 ce_weight=1.0,
                 eps=1e-8):
        super().__init__()
        self.num_classes = int(num_classes)
        self.alpha = float(alpha)
        self.w_far = float(w_far)
        self.delta_far = float(delta_far)
        self.w_lpeak = float(w_lpeak)
        self.prob_margin = float(prob_margin)
        self.w_emd = float(w_emd)
        self.ce_weight = float(ce_weight)
        self.eps = float(eps)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        K = self.num_classes
        assert K >= 3, "Servono almeno 3 classi."

        # costanti
        self.register_buffer("inv_logK", torch.tensor(1.0 / math.log(K), device=device))

        # distanze intere |j-i|
        classes = torch.arange(K, dtype=torch.float32, device=device)
        i = classes.view(K, 1); j = classes.view(1, K)
        d_idx = (j - i).abs()                        # [K,K] ∈ {0,1,2,...}
        self.register_buffer("dist_idx", d_idx)

        # pesi tail iniziali: (|d|-1)^p per |d|>1
        tail_w = torch.zeros_like(d_idx)
        tail_w[d_idx > 1] = (d_idx[d_idx > 1] - 1.0) ** self.tail_power
        self.register_buffer("tail_weights", tail_w)  # [K,K]

        # CDF target per EMD^2
        self.register_buffer("cdf_true_table", torch.tril(torch.ones(K, K, device=device)))

        # pesi per EMD^2 (bin non uniformi)
        if widths is not None:
            widths = torch.tensor(widths, device=device, dtype=torch.float32)
            widths = widths.to(device=device, dtype=torch.float32)
            w = widths.to(device=device, dtype=torch.float32)
            if w.numel() != K:
                raise ValueError(f"'widths' deve avere lunghezza {K}, got {w.numel()}")
            cdf_w = w / w.sum()
        else:
            cdf_w = torch.full((K,), 1.0 / K, device=device)
        self.register_buffer("cdf_weights", cdf_w)    # [K]

        # pesi di classe per CE
        if weights is not None:
            cw = weights.to(device=device, dtype=torch.float32)
            if cw.numel() != K:
                raise ValueError(f"'weights' deve avere lunghezza {K}, got {cw.numel()}")
            self.register_buffer("class_weights", cw)
        else:
            self.register_buffer("class_weights", torch.ones(K, device=device))

        self.ce = nn.CrossEntropyLoss(weight=self.class_weights, reduction='mean')


    def forward(self, logits, y, return_predicted_label=False):
        """
        logits: [B,K], y: [B] int64
        """
        B, K = logits.shape
        device = logits.device
        idx = torch.arange(B, device=device)

        # 1) CE sempre presente (nessun controllo)
        ce_loss = self.ce(logits, y)  * self.inv_logK               # [B]
        ce_loss = ce_loss * self.ce_weight

        # Probabilità normalizzate
        probs = F.softmax(logits, dim=1)

        # 2) FarProbMargin: vieta errori lontani (|k-y|>1)
        far_mask = self.dist_idx.index_select(0, y) > 1          # [B,K] bool
        far_vals = probs.masked_fill(~far_mask, -float('inf'))
        far_max = far_vals.max(dim=1).values
        far_max = torch.where(torch.isfinite(far_max), far_max, torch.zeros_like(far_max))
        py = probs[idx, y]
        far_margin = F.relu(far_max - (py - self.delta_far)) * self.class_weights[y]  # [B]
        far_margin = far_margin.mean()

        # 4) LocalPeak: gap con i vicini
        left_ok  = (y > 0)
        right_ok = (y + 1 < K)
        py_left  = torch.zeros(B, device=device, dtype=probs.dtype)
        py_right = torch.zeros(B, device=device, dtype=probs.dtype)
        if left_ok.any():
            py_left[left_ok] = probs[idx[left_ok], y[left_ok] - 1]
        if right_ok.any():
            py_right[right_ok] = probs[idx[right_ok], y[right_ok] + 1]
        neighbor_max = torch.maximum(py_left, py_right)
        local_peak = F.relu(neighbor_max - (py - self.prob_margin)) * self.class_weights[y]

        local_peak = local_peak.mean()

        # EMD^2 (CDF) per ordinalità globale
        emd2 = 0.0
        if self.w_emd != 0.0:
            cdf_pred = probs.cumsum(dim=-1)                         # [B,K]
            cdf_true = self.cdf_true_table.index_select(0, y)       # [B,K]
            diff2 = (cdf_pred - cdf_true).pow(2) * self.cdf_weights # [B,K] broadcast
            emd2 = diff2.sum(dim=-1) * self.class_weights[y]  # [B]
            emd2 = emd2.mean()

        # combinazione
        ord_block = (
            self.w_far   * far_margin +
            self.w_lpeak * local_peak +
            self.w_emd   * emd2
        )
        loss = ce_loss + self.alpha * ord_block

        if return_predicted_label:
            pred = probs.argmax(dim=1)
            return loss, pred
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