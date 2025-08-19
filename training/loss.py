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
                 alpha=0.4,
                 w_far=7.0,      delta_far=0.15,
                 w_tail=9.0,     tail_power=3,
                 w_lpeak=12.0,    prob_margin=0.35,
                 w_emd=1.2,
                 ce_weight=1.0,
                 eps=1e-8):
        super().__init__()
        self.num_classes = int(num_classes)
        self.alpha = float(alpha)
        self.w_far = float(w_far)
        self.delta_far = float(delta_far)
        self.w_tail = float(w_tail)
        self.tail_power = int(tail_power)
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

        self.ce = nn.CrossEntropyLoss(weight=self.class_weights, reduction='none')

    def set_weights(self, **kwargs):
        """
        Aggiorna pesi/margini a runtime (es: prob_margin=0.3, w_tail=8, ...).
        Se cambia 'tail_power', ricostruisce i pesi della tail.
        """
        device = self.cdf_weights.device
        K = self.num_classes
        tail_power_changed = False
        for k, v in kwargs.items():
            if hasattr(self, k):
                if k == "tail_power":
                    tail_power_changed = True
                    v = int(v)
                setattr(self, k, float(v) if isinstance(v, (int, float)) and k != "tail_power" else v)
        if tail_power_changed:
            # ricostruisci tail_weights con il nuovo tail_power
            classes = torch.arange(K, dtype=torch.float32, device=device)
            i = classes.view(K, 1); j = classes.view(1, K)
            d_idx = (j - i).abs()
            tail_w = torch.zeros_like(d_idx)
            mask = d_idx > 1
            tail_w[mask] = (d_idx[mask] - 1.0) ** int(self.tail_power)
            self.tail_weights = tail_w

    def forward(self, logits, y, return_predicted_label=False):
        """
        logits: [B,K], y: [B] int64
        """
        B, K = logits.shape
        device = logits.device
        idx = torch.arange(B, device=device)

        # 1) CE sempre presente (nessun controllo)
        ce_vec = self.ce(logits, y)                 # [B]
        ce_loss = ce_vec.mean() * self.inv_logK     # scalar
        ce_loss = ce_loss * self.ce_weight

        # Probabilità normalizzate
        probs = F.softmax(logits, dim=1).clamp_min(self.eps)
        probs = probs / probs.sum(dim=1, keepdim=True)

        # 2) FarProbMargin: vieta errori lontani (|k-y|>1)
        far_mask = self.dist_idx.index_select(0, y) > 1          # [B,K] bool
        far_vals = probs.masked_fill(~far_mask, -float('inf'))
        far_max = far_vals.max(dim=1).values
        far_max = torch.where(torch.isfinite(far_max), far_max, torch.zeros_like(far_max))
        py = probs[idx, y]
        far_margin = F.relu(far_max - (py - self.delta_far)).mean()

        # 3) TailWeighted: penalità crescente con la distanza fuori ±1
        tail_w = self.tail_weights.index_select(0, y)            # [B,K]
        tail = (probs * tail_w).sum(dim=1).mean()

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
        local_peak = F.relu(neighbor_max - (py - self.prob_margin)).mean()

        # EMD^2 (CDF) per ordinalità globale
        emd2 = 0.0
        if self.w_emd != 0.0:
            cdf_pred = probs.cumsum(dim=-1)                         # [B,K]
            cdf_true = self.cdf_true_table.index_select(0, y)       # [B,K]
            diff2 = (cdf_pred - cdf_true).pow(2) * self.cdf_weights # [B,K] broadcast
            emd2 = diff2.sum(dim=-1).mean()

        # combinazione
        ord_block = (
            self.w_far   * far_margin +
            self.w_tail  * tail +
            self.w_lpeak * local_peak +
            self.w_emd   * emd2
        )
        loss = ce_loss + self.alpha * ord_block

        if return_predicted_label:
            pred = probs.argmax(dim=1)
            return loss, pred
        return loss



class OrdinalLoss(nn.Module):
    def set_weights(self, alpha=None, w_ese=None, w_emd=None, w_tail=None, w_lpeak=None, w_uni=None):
        """
        Permette di modificare i pesi dei contributi della loss e alpha runtime.
        """
        if alpha is not None:
            self.alpha = float(alpha)
        if w_ese is not None:
            self.w_ese = float(w_ese)
        if w_emd is not None:
            self.w_emd = float(w_emd)
        if w_tail is not None:
            self.w_tail = float(w_tail)
        if w_lpeak is not None:
            self.w_lpeak = float(w_lpeak)
        if w_uni is not None:
            self.w_uni = float(w_uni)
    """
    Loss 'peaked & local' per età a bin ordinati:
      Loss = CE_norm
           + alpha * [ 1.0*ESE(dist^2) + 0.5*EMD + 3.0*Tail(|k-y|>1)
                      + 2.0*LocalPeak + 0.5*Unimodal ]

    - CE_norm: garantisce il picco sulla classe vera
    - ESE (quadratica): punisce fortemente errori lontani
    - EMD: allinea la CDF (ordine)
    - Tail: spinge quasi a zero la massa oltre i vicini (|k-y|>1)
    - LocalPeak: forza p[y] >= max(p[y-1], p[y+1]) (+margine)
    - Unimodal: decrescita regolare allontanandosi dal picco

    Firma invariata:
        __init__(num_classes, ordinal_loss='mae', weights=None, alpha=0.05)
    """
    def __init__(self, num_classes, ordinal_loss='mae', weights=None, alpha=0.5,
                 w_ese=2.0, w_emd=2.0, w_tail=5.0, w_lpeak=5.0, w_uni=1.5):
        super().__init__()
        self.num_classes = int(num_classes)
        self.alpha = float(alpha)
        self.w_ese = w_ese
        self.w_emd = w_emd
        self.w_tail = w_tail
        self.w_lpeak = w_lpeak
        self.w_uni = w_uni

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        K = self.num_classes

        # --- costanti ---
        self.register_buffer("inv_logK", torch.tensor(1.0 / math.log(K), device=device))
        self.register_buffer("inv_Km1", torch.tensor(1.0 / (K - 1), device=device))

        # --- indici e matrici precompute ---
        classes = torch.arange(K, dtype=torch.float32, device=device)  # [K]
        self.register_buffer("classes_range", classes.unsqueeze(0))     # [1,K]

        # Distanze normalizzate |j-i|/(K-1) e quadratiche
        i = classes.view(K, 1)
        j = classes.view(1, K)
        D = (j - i).abs() * self.inv_Km1          # [K,K] in [0,1]
        self.register_buffer("dist_matrix", D)
        self.register_buffer("dist2_matrix", D ** 2)

        # CDF target (triangolare inferiore)
        self.register_buffer("cdf_true_table", torch.tril(torch.ones(K, K, device=device)))

        # Maschera per massa fuori raggio 1: |j-i| > 1
        self.register_buffer("mask_outside1", (D > (1.0 * self.inv_Km1)).to(torch.float32))  # [K,K]

        # d = 1..K-1 per unimodalità
        self.register_buffer("d_vec", torch.arange(1, K, device=device))

        # Pesi per sbilanciamento
        if weights is not None:
            w = weights.to(device, dtype=torch.float32)
            if w.numel() != K:
                raise ValueError(f"'weights' must have length {K}, got {w.numel()}")
            self.register_buffer("class_weights", w)
        else:
            self.register_buffer("class_weights", torch.ones(K, device=device))

        # CE pesata (sbilanciamento)
        self.ce = nn.CrossEntropyLoss(weight=self.class_weights, reduction='none')

        # Margine per il vincolo locale p[y] >= max(p[y±1]) - margin
        self.local_margin = 0.25

    def forward(self, logit, true_labels, return_predicted_label=False):
        K = self.num_classes
        B = logit.size(0)
        device = logit.device

        # --- CE (picco sulla classe vera), normalizzata ---
        ce_vec = self.ce(logit, true_labels)          # [B]
        ce_loss = ce_vec.mean() * self.inv_logK

        # --- Probabilità ---
        probs = F.softmax(logit, dim=1)               # [B,K]

        # --- ESE: distanza quadratica attesa (forte sui lontani) ---
        D2_y = self.dist2_matrix.index_select(0, true_labels)      # [B,K]
        ese = (probs * D2_y).sum(dim=1)                            # [B]
        ese = ese * self.class_weights.index_select(0, true_labels)
        ese = ese.mean()

        # --- EMD/CDF (ordine globale) ---
        cdf_pred = probs.cumsum(dim=-1)                             # [B,K]
        cdf_true = self.cdf_true_table.index_select(0, true_labels) # [B,K]
        emd = ( (cdf_pred - cdf_true).pow(2).sum(dim=-1) / K ).mean()

        # --- Tail: massa fuori raggio 1 (|k-y|>1) ---
        tail_mask = self.mask_outside1.index_select(0, true_labels) # [B,K]
        tail = (probs * tail_mask).sum(dim=1).mean()

        # --- Local peak: p[y] >= max(p[y-1], p[y+1]) - margin ---
        idx = torch.arange(B, device=device)
        py = probs[idx, true_labels]
        left_ok  = (true_labels > 0)
        right_ok = (true_labels + 1 < K)
        py_left  = torch.zeros(B, device=device, dtype=probs.dtype)
        py_right = torch.zeros(B, device=device, dtype=probs.dtype)
        if left_ok.any():
            py_left[left_ok] = probs[idx[left_ok], true_labels[left_ok] - 1]
        if right_ok.any():
            py_right[right_ok] = probs[idx[right_ok], true_labels[right_ok] + 1]
        p_neighbor_max = torch.maximum(py_left, py_right)
        # penalizza se max(vicini) > py - margin  -> ReLU(max - (py - m))
        local_peak = F.relu(p_neighbor_max - (py - self.local_margin)).mean()

        # --- Unimodalità: decrescita regolare dai vicini verso l'esterno ---
        total_viols = logit.new_tensor(0.0)
        count_viols = 0
        for d in self.d_vec.tolist():  # 1..K-1
            # sinistra
            lpos, lprev = true_labels - d, true_labels - (d - 1)
            maskL = (lpos >= 0)
            if maskL.any():
                v = F.relu(probs[idx[maskL], lpos[maskL]] - probs[idx[maskL], lprev[maskL]])
                total_viols = total_viols + v.sum()
                count_viols += v.numel()
            # destra
            rpos, rprev = true_labels + d, true_labels + (d - 1)
            maskR = (rpos < K)
            if maskR.any():
                v = F.relu(probs[idx[maskR], rpos[maskR]] - probs[idx[maskR], rprev[maskR]])
                total_viols = total_viols + v.sum()
                count_viols += v.numel()
        unimodal = total_viols / count_viols if count_viols > 0 else logit.sum() * 0.0

        # --- combinazione ---
        ord_block = (
            self.w_ese   * ese   +
            self.w_emd   * emd   +
            self.w_tail  * tail  +
            self.w_lpeak * local_peak +
            self.w_uni   * unimodal
        )

        loss = ce_loss + self.alpha * ord_block

        if return_predicted_label:
            pred = torch.argmax(probs, dim=1)
            return loss, pred
        return loss

    def update_alpha(self, alpha):
        self.alpha = float(alpha)

class OrdinalLoss_(nn.Module):
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