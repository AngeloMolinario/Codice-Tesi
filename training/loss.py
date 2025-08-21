import torch
from torch import nn
from torch.nn import functional as F
import math
import torch
import torch.nn as nn


import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridOrdinalLossV2(nn.Module):
    def __init__(self,
                 num_classes=9,
                 alpha=0.3,
                 beta=0.7,
                 gamma=0.0,
                 eta=0.30,
                 lambda_dist=0.9,
                 support_radius=2,
                 temperature=None,
                 class_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = float(alpha)
        self.beta  = float(beta)
        self.gamma = float(gamma)
        self.eta   = float(eta)
        self.temperature = temperature
        self.lambda_dist = float(lambda_dist)
        self.support_radius = None if support_radius is None else int(support_radius)

        # pesi di classe
        if class_weights is None:
            cw = torch.ones(num_classes, dtype=torch.float32)
        elif isinstance(class_weights, (list, tuple)):
            cw = torch.tensor(class_weights, dtype=torch.float32)
        else:
            cw = class_weights.float()
        cw = cw / cw.mean().clamp_min(1e-12)
        self.register_buffer('class_weights', cw)

        self.register_buffer('distance_weights',
                             self._build_distance_weights(num_classes,
                                                          self.lambda_dist,
                                                          self.support_radius))

    @staticmethod
    def _build_distance_weights(C, lam, R):
        device = "cpu"  # inizialmente, poi spostato da register_buffer
        i = torch.arange(C, device=device).unsqueeze(1)
        j = torch.arange(C, device=device).unsqueeze(0)
        dist = (i - j).abs().float()
        W = torch.exp(-lam * dist)
        if R is not None:
            W = torch.where(dist <= R, W, torch.zeros_like(W))
        W = W / (W.sum(dim=1, keepdim=True) + 1e-12)
        return W

    @torch.no_grad()
    def predict(self, logits):
        z = logits / self.temperature if (self.temperature is not None) else logits
        return z.argmax(dim=1)

    def forward(self, logits, targets, return_predicted_label=False):
        device = logits.device
        z = logits / self.temperature if (self.temperature is not None) else logits
        loss_total = 0.0

        if self.alpha > 0:
            loss_total = loss_total + self.alpha * self._coral_adapted_loss(z, targets, device)

        if self.beta > 0:
            loss_total = loss_total + self.beta * self._soft_ce_peaked(z, targets, eta=self.eta, device=device)

        if self.gamma > 0:
            loss_total = loss_total + self.gamma * self._emd_loss(z, targets)

        return loss_total, logits.argmax(dim=1)

    def _soft_ce_peaked(self, logits, targets, eta=0.3, device="cpu"):
        B, C = logits.shape
        log_p = F.log_softmax(logits, dim=1)
        q_soft = self.distance_weights[targets]              # [B,C], già buffer -> segue device del modulo
        one_hot = F.one_hot(targets, num_classes=C).float().to(device)
        q = (1.0 - eta) * one_hot + eta * q_soft

        ce = -(q * log_p).sum(dim=1)
        ce = ce * self.class_weights[targets]
        return ce.mean()

    def _coral_adapted_loss(self, logits, targets, device="cpu"):
        B, C = logits.shape
        T = C - 1
        p = F.softmax(logits, dim=1)
        cdf_left = p.cumsum(dim=1)
        q_right = 1.0 - cdf_left[:, :-1]
        q_right = q_right.clamp(1e-8, 1 - 1e-8)
        ordinal_logits = torch.log(q_right) - torch.log1p(-q_right)

        i = torch.arange(T, device=device).view(1, -1)
        bin_targets = (targets.unsqueeze(1) > i).float()
        w = self.class_weights[targets].unsqueeze(1)
        loss = F.binary_cross_entropy_with_logits(ordinal_logits, bin_targets, weight=w, reduction='none')
        return loss.mean()

    def _emd_loss(self, logits, targets):
        p = F.softmax(logits, dim=1)
        cdf_p = p.cumsum(dim=1)
        t = F.one_hot(targets, num_classes=self.num_classes).float().to(logits.device)
        cdf_t = t.cumsum(dim=1)
        diff2 = (cdf_p - cdf_t).pow(2).sum(dim=1) * self.class_weights[targets]
        return diff2.mean()



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
                 alpha=1.0,
                 w_far=1.5,      delta_far=0.15,
                 w_lpeak=1.0,    prob_margin=0.25,
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
        ce_loss = self.ce(logits, y)             # [B]
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

        loss = loss * self.inv_logK  # normalizzazione per K

        if return_predicted_label:
            pred = probs.argmax(dim=1)
            return loss, pred
        return loss

    def set_alpha(self, new_alpha):
        self.alpha = float(new_alpha)
        

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HybridOrdinalLoss(nn.Module):
    """
    Loss ibrida adattata per logit di similarità coseno da CLIP con supporto per classi sbilanciate
    """
    def __init__(self, num_classes=9, alpha=0.7, beta=0.3, temperature=None, 
                 class_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha  # peso per CORAL loss
        self.beta = beta   # peso per distance-weighted CE
        self.temperature = temperature  # None = usa i logit così come sono dal modello
        
        # Gestione pesi delle classi
        if class_weights is not None:
            if isinstance(class_weights, (list, tuple)):
                class_weights = torch.tensor(class_weights, dtype=torch.float32)
            self.register_buffer('class_weights', class_weights)
        else:
            self.register_buffer('class_weights', torch.ones(num_classes))
        
        # Pre-calcola matrice dei pesi per distance-weighted CE
        self.register_buffer('distance_weights', self._create_distance_weights(num_classes))
    
    def _create_distance_weights(self, num_classes):
        """Crea matrice di pesi basata su distanza ordinale"""
        weights = torch.zeros(num_classes, num_classes, device='cuda' if torch.cuda.is_available() else 'cpu')
        for i in range(num_classes):
            for j in range(num_classes):
                # Penalità esponenziale per distanza ordinale
                weights[i, j] = torch.exp(torch.tensor(abs(i - j) * 1.5, device=weights.device))
        return weights

    def forward(self, cosine_logits, targets, return_predicted_label=False):
        """
        Forward pass della loss ibrida
        
        Args:
            cosine_logits: tensor [batch_size, num_classes] - similarità coseno * logit_scale
            targets: tensor [batch_size] - indici delle classi target (0 to num_classes-1)
        """
        # Ensure tensors are on the same device
        device = cosine_logits.device
        self.class_weights = self.class_weights.to(device)
        self.distance_weights = self.distance_weights.to(device)
        targets = targets.to(device)

        # Applica temperature scaling SOLO se specificato
        if self.temperature is not None:
            scaled_logits = cosine_logits / self.temperature
        else:
            # Usa i logit così come vengono dal modello (raccomandato)
            scaled_logits = cosine_logits
        
        # Calcola CORAL loss adattata
        coral_loss = self._adapted_coral_loss(scaled_logits, targets)
        
        # Calcola distance-weighted cross entropy
        dwce_loss = self._distance_weighted_ce(scaled_logits, targets)
        
        # Combina le loss
        total_loss = self.alpha * coral_loss + self.beta * dwce_loss

        return total_loss, torch.argmax(scaled_logits, dim=1)

    def _adapted_coral_loss(self, logits, targets):
        """
        CORAL loss adattata per logit di classificazione standard con pesi di classe
        """
        device = logits.device
        batch_size = logits.size(0)
        num_thresholds = self.num_classes - 1  # 8 soglie per 9 classi
        
        # Converte target in binary targets ordinali
        binary_targets = torch.zeros(batch_size, num_thresholds, device=device)
        for i in range(num_thresholds):
            binary_targets[:, i] = (targets > i).float()
        
        # Converte logit di classe in logit ordinali
        probs = F.softmax(logits, dim=1)  # [batch_size, num_classes]
        ordinal_logits = torch.zeros(batch_size, num_thresholds, device=device)
        
        for i in range(num_thresholds):
            prob_greater = torch.sum(probs[:, i+1:], dim=1)
            prob_greater = torch.clamp(prob_greater, min=1e-8, max=1-1e-8)
            prob_leq = 1 - prob_greater
            prob_leq = torch.clamp(prob_leq, min=1e-8, max=1-1e-8)
            ordinal_logits[:, i] = torch.log(prob_greater / prob_leq)
        
        # Calcola binary cross entropy per ogni soglia CON PESI DI CLASSE
        losses = []
        for i in range(num_thresholds):
            # Calcola pesi per questa soglia basandosi sui target
            # Per soglia i: peso medio tra classi <= i e classi > i
            weights_left = self.class_weights[:i+1].mean() if i >= 0 else self.class_weights[0]
            weights_right = self.class_weights[i+1:].mean() if i+1 < self.num_classes else self.class_weights[-1]
            
            # Peso per ogni campione basato sul suo target binario
            sample_weights = torch.where(
                binary_targets[:, i] == 1.0,
                weights_right,  # Peso per classi > soglia
                weights_left    # Peso per classi <= soglia
            )
            
            loss = F.binary_cross_entropy_with_logits(
                ordinal_logits[:, i], 
                binary_targets[:, i],
                weight=sample_weights,
                reduction='none'
            )
            losses.append(loss)
        
        coral_loss = torch.stack(losses, dim=1).mean()
        return coral_loss
    
    def _distance_weighted_ce(self, logits, targets):
        """
        Cross entropy pesata per distanza ordinale E pesi di classe
        """
        device = logits.device
        self.class_weights = self.class_weights.to(device)
        self.distance_weights = self.distance_weights.to(device)
        targets = targets.to(device)

        # Calcola probabilità predette
        log_probs = F.log_softmax(logits, dim=1)  # [batch_size, num_classes]
        
        # Combina distance weights e class weights
        # distance_weights: [num_classes, num_classes] - penalizza errori lontani
        # class_weights: [num_classes] - bilancia classi sbilanciate
        batch_distance_weights = self.distance_weights[targets]  # [batch_size, num_classes]
        batch_class_weights = self.class_weights.unsqueeze(0).expand(logits.size(0), -1)  # [batch_size, num_classes]
        
        # Combina i due tipi di pesi
        combined_weights = batch_distance_weights * batch_class_weights
        
        # Applica i pesi alle log-probabilità
        weighted_log_probs = log_probs * combined_weights
        
        # Calcola loss per la classe corretta, pesata per class weight
        target_log_probs = weighted_log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        target_class_weights = self.class_weights[targets]
        
        dwce_loss = -(target_log_probs * target_class_weights).mean()
        return dwce_loss


class OrdinalConcentratedLoss(nn.Module):
    """
    Loss = CE_norm_weighted
         + alpha * [ w_far   * FarLoss_weighted
                     w_conc  * ConcentrationLoss_weighted
                     w_emd   * EMD^2_weighted
                     w_peak  * LocalPeak_weighted ]   <-- NEW
    """
    def __init__(self, num_classes, weights=None, widths=[3, 7, 10, 10, 10, 10, 10, 10, 25],
                 alpha=1.0,
                 ce_weight=1.0,
                 w_far=2.0,
                 w_conc=1.0,
                 w_emd=1.0,
                 # --- NEW ---
                 w_peak=3.0,         # peso del vincolo di picco locale
                 delta_peak=0.12,    # margine richiesto tra p[y] e i vicini
                 eps=1e-8):
        super().__init__()
        self.num_classes = int(num_classes)
        self.alpha = float(alpha)
        self.ce_weight = float(ce_weight)
        self.w_far = float(w_far)
        self.w_conc = float(w_conc)
        self.w_emd = float(w_emd)
        # --- NEW ---
        self.w_peak = float(w_peak)
        self.delta_peak = float(delta_peak)

        self.eps = float(eps)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        K = self.num_classes
        assert K >= 3, "Servono almeno 3 classi."

        # Buffer e costanti
        self.register_buffer("inv_logK", torch.tensor(1.0 / math.log(K), device=device))
        classes = torch.arange(K, dtype=torch.float32, device=device)
        i = classes.view(K, 1); j = classes.view(1, K)
        d_idx = (j - i).abs()
        self.register_buffer("dist_idx", d_idx)
        self.register_buffer("cdf_true_table", torch.tril(torch.ones(K, K, device=device)))
        
        # Pesi di classe
        if weights is not None:
            cw = weights.to(device=device, dtype=torch.float32)
            if cw.numel() != K:
                raise ValueError(f"'weights' deve avere lunghezza {K}, got {cw.numel()}")
            self.register_buffer("class_weights", cw)
        else:
            self.register_buffer("class_weights", torch.ones(K, device=device))

        # Pesi EMD (widths)
        if widths is not None:
            widths_tensor = torch.tensor(widths, device=device, dtype=torch.float32)
            w_emd_weights = widths_tensor / widths_tensor.sum()
            self.register_buffer("cdf_weights", w_emd_weights)
        else:
            self.register_buffer("cdf_weights", torch.full((K,), 1.0 / K, device=device))

        # CE pesata
        self.ce = nn.CrossEntropyLoss(weight=self.class_weights, reduction='mean')

    def forward(self, logits, y, return_predicted_label=False):
        B, K = logits.shape
        device = logits.device
        idx = torch.arange(B, device=device)

        # Probabilità
        probs = F.softmax(logits, dim=1)

        # --- 1) CE pesata ---
        ce_loss = self.ce(logits, y) * self.ce_weight
        
        # Pesi per campione
        sample_weights = self.class_weights.index_select(0, y)

        # --- 2) FarLoss ---
        far_mask = self.dist_idx.index_select(0, y) > 1          # [B,K] bool
        prob_far_per_sample = (probs * far_mask.float()).sum(dim=1)
        far_loss = (prob_far_per_sample * sample_weights).mean()

        # --- 3) Concentration (tri-locale) ---
        py = probs[idx, y]
        py_left = torch.zeros_like(py)
        py_right = torch.zeros_like(py)
        left_ok = y > 0
        right_ok = y < K - 1
        py_left[left_ok] = probs[idx[left_ok], y[left_ok] - 1]
        py_right[right_ok] = probs[idx[right_ok], y[right_ok] + 1]
        
        local_mass_sum = py + py_left + py_right
        concentration_loss_per_sample = 1.0 - local_mass_sum
        concentration_loss = (concentration_loss_per_sample * sample_weights).mean()

        # --- 4) EMD^2 (CDF) ---
        cdf_pred = probs.cumsum(dim=-1)
        cdf_true = self.cdf_true_table.index_select(0, y)
        diff2 = (cdf_pred - cdf_true).pow(2) * self.cdf_weights
        emd2_per_sample = diff2.sum(dim=-1)
        emd2 = (emd2_per_sample * sample_weights).mean()

        # --- 5) Local Peak (NEW) ---
        # Enforce: p[y] >= p[y±1] + delta_peak (ai bordi solo il vicino esistente)
        viol_left  = torch.zeros_like(py)
        viol_right = torch.zeros_like(py)
        viol_left[left_ok]   = F.relu(py_left[left_ok]  - py[left_ok]  + self.delta_peak)
        viol_right[right_ok] = F.relu(py_right[right_ok] - py[right_ok] + self.delta_peak)

        peak_violation = torch.where(~left_ok,  viol_right,
                              torch.where(~right_ok, viol_left,
                                          torch.maximum(viol_left, viol_right)))
        peak_loss = (peak_violation * sample_weights).mean()

        # --- Combinazione finale ---
        ord_block = (
            self.w_far   * far_loss +
            self.w_conc  * concentration_loss +
            self.w_emd   * emd2 +
            self.w_peak  * peak_loss            # <-- NEW
        )
        loss = ce_loss + self.alpha * ord_block
        loss = loss * self.inv_logK

        if return_predicted_label:
            predicted_labels = torch.argmax(probs, dim=1)
            return loss, predicted_labels
        return loss

    def set_alpha(self, new_alpha: float):
        self.alpha = float(new_alpha)


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