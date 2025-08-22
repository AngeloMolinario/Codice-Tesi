import torch
from torch.nn import functional as F
import math
import torch.nn as nn

class HybridOrdinalLossV2_(nn.Module):
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

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridOrdinalLossV3EMD(nn.Module):
    """
    Logits CLIP-like con:
      - CE con soft-target 'peaked'              [peso beta]  <-- principale
      - EMD^2 tra CDF (ordinal ranking)          [peso gamma]
      - Hinge locale vs classi adiacenti         [peso neighbor_margin_weight]
        con margine per-classe m_y (da etichetta y)

    Dinamica margini (opzionale):
      - Durante i primi 'warmup_steps' (solo in training) accumula i gap verso i vicini
        e ricalibra m_y = quantile(target_quantile) per classe.
      - In validazione/Eval non accumula né aggiorna nulla.
    """

    def __init__(self,
                 num_classes=9,
                 # CE soft-peaked
                 beta=0.7,
                 eta=0.3,
                 lambda_dist=1.8,
                 support_radius=1,
                 # EMD^2
                 gamma=0.5,
                 # pesi di classe
                 class_weights=None,
                 # hinge locale vs vicini
                 neighbor_margin=0.3,
                 neighbor_margin_weight=0.5,
                 margin_by_class=None,
                 # ==== dinamica margini ====
                 dynamic_margins=True,
                 warmup_steps=1500,
                 update_interval=100,
                 target_quantile=0.40,
                 per_class_cap=6000,
                 min_margin=0.02,
                 max_margin=0.80):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = int(num_classes)

        # Pesi dei termini
        self.beta  = float(beta)
        self.gamma = float(gamma)

        # Parametri CE peaked
        self.eta   = float(eta)
        self.lambda_dist = float(lambda_dist)
        self.support_radius = None if support_radius is None else int(support_radius)

        # Pesi di classe
        if class_weights is None:
            cw = torch.ones(self.num_classes, dtype=torch.float32, device=self.device)
        elif isinstance(class_weights, (list, tuple)):
            cw = torch.tensor(class_weights, dtype=torch.float32, device=self.device)
        else:
            cw = class_weights.float().to(self.device)
        cw = cw / cw.mean().clamp_min(1e-12)
        self.register_buffer('class_weights', cw)

        # Kernel di distanza per soft-target
        dw = self._build_distance_weights(self.num_classes,
                                          self.lambda_dist,
                                          self.support_radius,
                                          device=self.device)
        self.register_buffer('distance_weights', dw)

        # Margini per-classe (inizializzazione seed)
        if margin_by_class is None:
            mbc = torch.full((self.num_classes,),
                             float(neighbor_margin),
                             dtype=torch.float32,
                             device=self.device)
        else:
            if isinstance(margin_by_class, (list, tuple)):
                mbc = torch.tensor(margin_by_class,
                                   dtype=torch.float32,
                                   device=self.device)
            else:
                mbc = margin_by_class.float().to(self.device)
            assert mbc.numel() == self.num_classes, \
                "margin_by_class deve avere lunghezza = num_classes"
        self.register_buffer('margin_by_class', mbc)
        self.neighbor_margin_weight = float(neighbor_margin_weight)

        # Stato dinamico
        self.dynamic_margins   = bool(dynamic_margins)
        self.warmup_steps      = int(warmup_steps)
        self.update_interval   = int(update_interval)
        self.target_quantile   = float(target_quantile)
        self.per_class_cap     = int(per_class_cap)
        self.min_margin        = float(min_margin)
        self.max_margin        = float(max_margin)

        self._global_step = 0
        self._margins_frozen = not self.dynamic_margins
        # pool CPU per statistiche (liste di tensori fp32)
        self._gap_pool = [[] for _ in range(self.num_classes)]
        self._pool_sizes = [0 for _ in range(self.num_classes)]

    # -------- utils --------
    @staticmethod
    def _build_distance_weights(C, lam, R, device):
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
        return logits.argmax(dim=1)

    # -------- forward --------
    def forward(self, logits, targets, return_predicted_label=False):
        if logits.device.type != self.device.type:
            logits = logits.to(self.device)
        if targets.device.type != self.device.type:
            targets = targets.to(self.device)

        # Warm-up dinamico: SOLO in training
        if self.training and (not self._margins_frozen) and self.neighbor_margin_weight > 0:
            self._accumulate_gap_stats(logits, targets)
            if (self._global_step % self.update_interval == 0) or (self._global_step == self.warmup_steps):
                self._recompute_margins()
            self._global_step += 1
            if self._global_step >= self.warmup_steps:
                self._recompute_margins()
                self._freeze_margins()

        loss_total = 0.0

        if self.beta > 0:
            loss_total = loss_total + self.beta * self._soft_ce_peaked(logits, targets, eta=self.eta)

        if self.gamma > 0:
            loss_total = loss_total + self.gamma * self._emd_loss(logits, targets)

        if self.neighbor_margin_weight > 0:
            loss_total = loss_total + self.neighbor_margin_weight * self._neighbor_margin(logits, targets)

        preds = logits.argmax(dim=1)
        return (loss_total, preds) if return_predicted_label else loss_total

    # -------- componenti di perdita --------
    def _soft_ce_peaked(self, logits, targets, eta=0.3):
        B, C = logits.shape
        log_p = F.log_softmax(logits, dim=1)
        q_soft = self.distance_weights[targets]
        one_hot = F.one_hot(targets, num_classes=C).float()
        q = (1.0 - eta) * one_hot + eta * q_soft
        ce = -(q * log_p).sum(dim=1)
        ce = ce * self.class_weights[targets]
        return ce.mean()

    def _emd_loss(self, logits, targets):
        p = F.softmax(logits, dim=1)
        cdf_p = p.cumsum(dim=1)
        t = F.one_hot(targets, num_classes=self.num_classes).float()
        cdf_t = t.cumsum(dim=1)
        diff2 = (cdf_p - cdf_t).pow(2).sum(dim=1) * self.class_weights[targets]
        return diff2.mean()

    def _neighbor_margin(self, logits, targets):
        B, C = logits.shape
        y = targets

        zy = logits.gather(1, y.view(-1, 1)).squeeze(1)
        left  = (y - 1).clamp(min=0)
        right = (y + 1).clamp(max=C - 1)
        zl = logits.gather(1, left.view(-1, 1)).squeeze(1)
        zr = logits.gather(1, right.view(-1, 1)).squeeze(1)

        has_left  = (y > 0).float()
        has_right = (y < C - 1).float()

        m = self.margin_by_class[y]
        loss_l = F.relu(m - (zy - zl)) * has_left
        loss_r = F.relu(m - (zy - zr)) * has_right
        loss = (loss_l + loss_r) * self.class_weights[y]
        return loss.mean()

    # -------- dinamica margini --------
    @torch.no_grad()
    def _accumulate_gap_stats(self, logits, targets):
        """
        Accumula su CPU il gap peggiore verso i vicini:
        g = min(z_y - z_{y-1}, z_y - z_{y+1}) (ignora gli estremi non esistenti).
        """
        B, C = logits.shape
        y = targets
        left  = (y - 1).clamp(min=0)
        right = (y + 1).clamp(max=C - 1)

        zy = logits.gather(1, y.view(-1,1)).squeeze(1)
        zl = logits.gather(1, left.view(-1,1)).squeeze(1)
        zr = logits.gather(1, right.view(-1,1)).squeeze(1)

        g_l = zy - zl
        g_r = zy - zr

        g_l = torch.where(y > 0,   g_l, torch.full_like(g_l, float('inf')))
        g_r = torch.where(y < C-1, g_r, torch.full_like(g_r, float('inf')))
        g = torch.minimum(g_l, g_r)  # [B]

        # Porta su CPU in fp32 (compatibile AMP)
        y_cpu = y.detach().cpu()
        g_cpu = g.detach().float().cpu()

        for cls in y_cpu.unique(sorted=True):
            cls = int(cls.item())
            vals = g_cpu[y_cpu == cls]
            if vals.numel() == 0:
                continue
            self._gap_pool[cls].append(vals)  # fp32
            self._pool_sizes[cls] += int(vals.numel())
            if self._pool_sizes[cls] > self.per_class_cap:
                cat = torch.cat(self._gap_pool[cls], dim=0).float()
                self._gap_pool[cls] = [cat[-self.per_class_cap:]]
                self._pool_sizes[cls] = int(self._gap_pool[cls][0].numel())

    @torch.no_grad()
    def _recompute_margins(self):
        """
        m_y = quantile(target_quantile) dei gap per classe (fp32),
        clamp su [min_margin, max_margin].
        """
        new_m = torch.clone(self.margin_by_class).detach().cpu().float()
        any_update = False
        for cls in range(self.num_classes):
            if self._pool_sizes[cls] == 0:
                continue
            cat = torch.cat(self._gap_pool[cls], dim=0).float()
            q = torch.quantile(cat, q=self.target_quantile).item()
            new_m[cls] = q
            any_update = True

        # fallback: se nessuna classe ha stats, mantieni i margini correnti
        if not any_update:
            new_m = new_m.clamp(min=self.min_margin, max=self.max_margin).to(self.device)
            self.margin_by_class.data.copy_(new_m)
            return

        # se qualche classe è senza stats, usa la media delle altre
        mask = new_m != 0
        if mask.any():
            mean_val = new_m[mask].mean().item()
            new_m[~mask] = mean_val

        new_m = new_m.clamp(min=self.min_margin, max=self.max_margin).to(self.device)
        self.margin_by_class.data.copy_(new_m)

    @torch.no_grad()
    def _freeze_margins(self):
        """Congela la calibrazione (svuota pool per liberare memoria)."""
        self._margins_frozen = True
        self._gap_pool = [[] for _ in range(self.num_classes)]
        self._pool_sizes = [0 for _ in range(self.num_classes)]

import torch
import torch.nn.functional as F
import torch
from torch import nn
import torch.nn.functional as F

class OrdinalPeakLossImbalance(nn.Module):
    """
    Loss per classificazione d'età multiclasse *ordinale* con:
      - Cross-Entropy/Focal per massimizzare l'accuracy e il picco su p_y
      - Earth Mover's Distance 1D (EMD^2) per rispettare l'ordine delle classi
      - Hinge di margine contro le classi adiacenti (riduce errori ai bordi)
      - (opzionale) Entropia per rendere più appuntite le distribuzioni

    Parametri:
      num_classes: K
      class_weights: Tensor (K,) con pesi inverse-frequency (verranno registrati come buffer)
      ce_coef, emd_coef, margin_coef, entropy_coef: coefficienti dei termini
      margin: margine desiderato per p_y - max(p_{y-1}, p_{y+1})
      focal_gamma: 0 -> CE standard, >0 -> focal
    """
    def __init__(
        self,
        num_classes: int,
        class_weights: torch.Tensor,
        ce_coef: float = 1.0,
        emd_coef: float = 0.6,
        margin_coef: float = 0.4,
        entropy_coef: float = 0.02,
        margin: float = 0.05,
        focal_gamma: float = 1.0,
        normalize_weights: bool = True,
    ):
        super().__init__()
        assert class_weights.shape[0] == num_classes, "class_weights deve avere shape (K,)."
        self.K = int(num_classes)

        # Registriamo i pesi come buffer, così seguono .to(device)/.cuda() e finiscono nel state_dict
        cw = class_weights.clone().detach().float()

        if normalize_weights:
            # esempio: normalizza a media = 1
            mean_val = cw.mean()
            if not torch.isclose(mean_val, torch.tensor(1.0), rtol=1e-2, atol=1e-2):
                cw = cw / mean_val

        self.register_buffer("class_weights", cw)

        # Iperparametri
        self.ce_coef = ce_coef
        self.emd_coef = emd_coef
        self.margin_coef = margin_coef
        self.entropy_coef = entropy_coef
        self.margin = margin
        self.focal_gamma = focal_gamma

    @staticmethod
    def build_class_weights_from_counts(counts: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        Utility: genera pesi inverse-frequency da conteggi per classe.
        counts: Tensor (K,) con numero di esempi per classe nel TRAIN.
        """
        counts = counts.float().clamp_min(1.0)
        inv = 1.0 / counts
        if normalize:
            inv = inv / inv.mean()
        return inv

    @torch.no_grad()
    def set_class_weights(self, new_weights: torch.Tensor):
        assert new_weights.shape[0] == self.K
        self.class_weights.copy_(new_weights.float())

    def forward(self, logits: torch.Tensor, target: torch.Tensor, return_predicted_label: bool = False):
        """
        logits: (B, K) — output non normalizzato del modello
        target: (B,)  — etichette int64 in [0..K-1]
        Ritorna: loss (scalar) e un dict con le componenti.
        """
        B, K = logits.shape
        assert K == self.K, "Dimensione K dei logits non coincide con num_classes."

        # Softmax su logits (niente temperatura qui)
        p = F.softmax(logits, dim=1)              # (B, K)
        logp = F.log_softmax(logits, dim=1)       # (B, K)

        # Pesi per-sample in base alla classe vera (inverse-frequency)
        w = self.class_weights.gather(0, target)  # (B,)

        # ---- 1) (Focal) Cross-Entropy con pesi inverse-frequency ----
        # NOTA: NON usiamo 'weight=' dentro nll_loss per evitare doppio pesaggio.
        ce_per = F.nll_loss(logp, target, reduction='none')  # (B,)
        if self.focal_gamma > 0.0:
            pt = torch.gather(p, 1, target.view(-1,1)).squeeze(1).clamp_min(1e-8)
            ce_per = ((1 - pt) ** self.focal_gamma) * ce_per
        ce = (ce_per * w).sum() / w.sum()

        # ---- 2) EMD^2 1D via CDF (rispetta ordine) ----
        cdf_p = torch.cumsum(p, dim=1)                       # (B, K)
        idx = torch.arange(K, device=logits.device).view(1, -1)
        cdf_y = (idx >= target.view(-1,1)).float()           # (B, K)
        emd_per = ((cdf_p - cdf_y) ** 2).mean(dim=1)         # (B,)
        emd2 = (emd_per * w).sum() / w.sum()

        # ---- 3) Hinge di margine sui vicini (riduce confusione ai bordi) ----
        py = torch.gather(p, 1, target.view(-1,1)).squeeze(1)    # (B,)
        left_idx  = (target - 1).clamp_min(0)
        right_idx = (target + 1).clamp_max(K-1)
        p_left  = torch.gather(p, 1, left_idx.view(-1,1)).squeeze(1)
        p_right = torch.gather(p, 1, right_idx.view(-1,1)).squeeze(1)
        pneigh = torch.maximum(p_left, p_right)
        margin_per = F.relu(self.margin - (py - pneigh))         # (B,)
        margin_pen = (margin_per * w).sum() / w.sum()

        # ---- 4) Entropia (opzionale) per rendere p più “appuntite” ----
        if self.entropy_coef > 0.0:
            entropy_per = -(p * logp).sum(dim=1)                 # (B,)
            entropy = (entropy_per * w).sum() / w.sum()
        else:
            entropy = torch.tensor(0.0, device=logits.device)

        loss = (
            self.ce_coef * ce
            + self.emd_coef * emd2
            + self.margin_coef * margin_pen
            + self.entropy_coef * entropy
        )

        logs = {
            "loss": float(loss),
            "ce": float(ce),
            "emd2": float(emd2),
            "margin": float(margin_pen),
            "entropy": float(entropy),
        }
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