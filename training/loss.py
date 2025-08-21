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

import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridOrdinalLossV2(nn.Module):
    def __init__(self,
                 num_classes=9,
                 alpha=0.3,
                 beta=0.7,              # peso della HINGE "peaked"
                 gamma=0.0,
                 eta=0.30,
                 lambda_dist=0.9,
                 support_radius=2,
                 temperature=None,
                 class_weights=None,
                 hinge_margin=1.0,
                 hinge_squared=False):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = float(alpha)
        self.beta  = float(beta)
        self.gamma = float(gamma)
        self.eta   = float(eta)
        self.temperature = temperature
        self.lambda_dist = float(lambda_dist)
        self.support_radius = None if support_radius is None else int(support_radius)
        self.hinge_margin = float(hinge_margin)
        self.hinge_squared = bool(hinge_squared)

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
        device = "cpu"
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
            # HINGE "peaked" al posto della CE "peaked"
            loss_total = loss_total + self.beta * self._hinge_peaked(
                z, targets, eta=self.eta, margin=self.hinge_margin,
                squared=self.hinge_squared, device=device
            )

        if self.gamma > 0:
            loss_total = loss_total + self.gamma * self._emd_loss(z, targets)

        return loss_total, logits.argmax(dim=1)

    # --------- NUOVO: Hinge con soft-label peaked ----------
    def _hinge_peaked(self, logits, targets, eta=0.3, margin=1.0, squared=False, device="cpu"):
        """
        Multiclasse 'one-vs-all' con etichette morbide:
          - Costruisco q (peaked): q = (1-eta)*one_hot + eta*kernel_ordinale
          - Mappo in y in [-1,+1]: y = 2*q - 1
          - Applico hinge per-classe: max(0, m - y*z)
            (per y≈+1 spinge z in alto; per y≈-1 spinge z in basso)
        """
        B, C = logits.shape

        # q 'peaked' (come nella tua CE)
        q_soft = self.distance_weights[targets]                      # [B,C] buffer → segue device del modulo
        one_hot = F.one_hot(targets, num_classes=C).float().to(device)
        q = (1.0 - eta) * one_hot + eta * q_soft

        # y in [-1,+1]
        y = 2.0 * q - 1.0                                            # [B,C]

        # hinge per-classe
        # m_k = margin - y_k * z_k
        m = margin - y * logits
        h = torch.clamp(m, min=0.0)
        if squared:
            h = h * h

        # media per campione sulle classi, poi pesi di classe sul target "hard"
        per_sample = h.mean(dim=1) * self.class_weights[targets]
        return per_sample.mean()

    # --------- come prima ----------
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