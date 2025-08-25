import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class OrdinalAgeLossFocus(nn.Module):
    """
    Una loss function che combina la Focal Loss per la classificazione
    e una loss di ranking ordinale (Earth Mover's Distance) per tenere conto
    della natura ordinata delle classi (es. età).

    Gestisce lo sbilanciamento delle classi con pesi separati per 
    la componente di classificazione e quella ordinale.
    """
    def __init__(self, num_classes=9, class_frequencies=None, lambda_ordinal=0.3, 
                 use_squared_emd=True, focal_gamma=2.0):
        super(OrdinalAgeLossFocus, self).__init__()
        self.num_classes = num_classes
        self.lambda_ordinal = lambda_ordinal
        self.use_squared_emd = use_squared_emd
        self.focal_gamma = focal_gamma
        
        if class_frequencies is not None:
            # Pesi per la loss ordinale (EMD), normalizzati a media 1
            emd_weights = class_frequencies
            # Pesi (alpha) per la Focal Loss, normalizzati con massimo 1
            focal_alpha = class_frequencies
        else:
            emd_weights = torch.ones(num_classes)
            focal_alpha = torch.ones(num_classes)

        # <--- MIGLIORAMENTO: Registra i pesi come buffer
        # In questo modo, vengono spostati automaticamente sul dispositivo corretto (CPU/GPU)
        # quando si chiama model.to(device) e vengono salvati con lo state_dict del modello.
        self.register_buffer('emd_weights', emd_weights)
        self.register_buffer('focal_alpha', focal_alpha)

    def focal_loss_alternative(self, predictions, targets):
        """
        Implementazione della Focal Loss.
        """
        # Cross-entropy base senza riduzione, i pesi sono applicati dopo
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        
        # Calcola p_t, la probabilità della classe corretta
        probs = F.softmax(predictions, dim=1)
        pt = probs[torch.arange(len(targets), device=predictions.device), targets]
        
        # Calcola il fattore di modulazione della Focal Loss
        focal_weight = (1 - pt) ** self.focal_gamma
        
        # <--- CORREZIONE: Seleziona il peso alpha corretto per ogni campione
        # self.focal_alpha ha shape [num_classes]. targets ha shape [batch_size].
        # self.focal_alpha[targets] crea un tensore di shape [batch_size] con il peso giusto per ogni elemento.
        alpha_t = self.focal_alpha[targets]
        
        # Applica sia il fattore di modulazione che il peso alpha
        focal_loss = alpha_t * focal_weight * ce_loss
        
        return torch.mean(focal_loss)
    
    def compute_weighted_emd_loss(self, probs, targets):
        """
        Versione pesata della EMD loss.
        """
        target_dist = F.one_hot(targets, num_classes=self.num_classes).float()
        
        # Calcola CDF (Cumulative Distribution Function)
        pred_cdf = torch.cumsum(probs, dim=1)
        target_cdf = torch.cumsum(target_dist, dim=1)
        
        # Calcola la EMD (o EMD^2) per ogni campione
        if self.use_squared_emd:
            emd_losses = torch.sum((pred_cdf - target_cdf) ** 2, dim=1)
        else:
            emd_losses = torch.sum(torch.abs(pred_cdf - target_cdf), dim=1)
        
        # <--- CORREZIONE: Seleziona il peso EMD corretto per ogni campione
        sample_weights = self.emd_weights[targets]
        
        # <--- CORREZIONE: Moltiplicazione elemento per elemento e media
        # Non è necessario un unsqueeze, le dimensioni ora sono corrette: [batch_size] * [batch_size]
        weighted_emd_loss = torch.mean(sample_weights * emd_losses)
        
        return weighted_emd_loss

    def forward(self, predictions, targets, return_predicted_label=False):
        # 1. Weighted Focal Loss per la classificazione
        focal_loss = self.focal_loss_alternative(predictions, targets)
        
        # 2. Weighted Ordinal EMD Loss per il ranking
        probs = F.softmax(predictions, dim=1)
        weighted_emd_loss = self.compute_weighted_emd_loss(probs, targets)
        
        # 3. Normalizzazione della EMD loss per renderla più stabile
        # La divisione per (num_classes - 1) scala la loss in un range più prevedibile
        if self.use_squared_emd:
            # Per squared EMD, il normalizzatore è al quadrato
            normalizer = (self.num_classes - 1) ** 2
        else:
            normalizer = (self.num_classes - 1)
        
        normalized_emd_loss = weighted_emd_loss / normalizer
        
        # 4. Combinazione finale delle due componenti di loss
        total_loss = focal_loss + self.lambda_ordinal * normalized_emd_loss
        
        # <--- MIGLIORAMENTO: Usa il flag per restituire opzionalmente le predizioni
        if return_predicted_label:
            predicted_labels = predictions.argmax(dim=1)
            return total_loss, predicted_labels
        
        return total_loss

class OrdinalAgeLossEMD(nn.Module):
    def __init__(self, num_classes=9, class_frequencies=None, lambda_ordinal=0.3, use_squared_emd=True):
        super(OrdinalAgeLossEMD, self).__init__()
        self.num_classes = num_classes
        self.lambda_ordinal = lambda_ordinal
        self.use_squared_emd = use_squared_emd
        
        # Calcolo dei pesi per gestire lo sbilanciamento
        if class_frequencies is not None:
            self.class_weights = class_frequencies
        else:
            self.class_weights = torch.ones(num_classes).to('cuda')

    def compute_emd_loss(self, probs, targets, squared=False):
        """
        Computa Earth Mover's Distance tra distribuzioni predette e target.
        
        Args:
            probs: Probabilità predette (batch_size, num_classes)
            targets: Indici delle classi target (batch_size,)
            squared: Se True, usa squared EMD (EMD²)
        
        Returns:
            EMD loss per ogni esempio nel batch
        """
        batch_size = probs.size(0)
        device = probs.device
        
        # Crea distribuzione target one-hot
        target_dist = torch.zeros_like(probs)
        target_dist.scatter_(1, targets.unsqueeze(1), 1.0)
        
        # Calcola CDF (Cumulative Distribution Function)
        pred_cdf = torch.cumsum(probs, dim=1)
        target_cdf = torch.cumsum(target_dist, dim=1)
        
        if squared:
            # Squared EMD: usa differenze al quadrato
            emd_loss = torch.sum((pred_cdf - target_cdf) ** 2, dim=1)
        else:
            # EMD classica: usa differenze assolute
            emd_loss = torch.sum(torch.abs(pred_cdf - target_cdf), dim=1)
        
        return emd_loss

    def compute_weighted_emd_loss(self, probs, targets, squared=False):
        """
        Versione pesata della EMD loss che considera i pesi delle classi.
        
        Args:
            squared: Se True, usa squared EMD per penalizzazione più forte
        """
        batch_size = probs.size(0)
        device = probs.device
        
        # Crea distribuzione target one-hot
        target_dist = torch.zeros_like(probs)
        target_dist.scatter_(1, targets.unsqueeze(1), 1.0)
        
        # Calcola CDF
        pred_cdf = torch.cumsum(probs, dim=1)
        target_cdf = torch.cumsum(target_dist, dim=1)
        
        # Calcola EMD per ogni esempio
        if squared:
            emd_losses = torch.sum((pred_cdf - target_cdf) ** 2, dim=1)
        else:
            emd_losses = torch.sum(torch.abs(pred_cdf - target_cdf), dim=1)
        
        # Applica i pesi specifici per ogni esempio in base alla sua classe
        sample_weights = self.class_weights[targets].to(device)
        weighted_emd_loss = torch.mean(sample_weights * emd_losses)
        
        return weighted_emd_loss

    def forward(self, predictions, targets, return_predicted_label=False):
        # 1. Weighted Cross-Entropy
        ce_loss = F.cross_entropy(
            predictions, 
            targets, 
            weight=self.class_weights.to(predictions.device)
        )
        
        # 2. Weighted EMD Ordinal Loss (normale o squared)
        probs = F.softmax(predictions, dim=1)
        
        # Usa EMD o squared EMD in base al parametro
        weighted_emd_loss = self.compute_weighted_emd_loss(probs, targets, squared=self.use_squared_emd)
        
        # Normalizzazione adattiva in base al tipo di EMD
        if self.use_squared_emd:
            # Per squared EMD, normalizzazione diversa perché i valori sono al quadrato
            normalized_emd_loss = weighted_emd_loss / ((self.num_classes - 1) ** 2)
        else:
            # EMD classica
            normalized_emd_loss = weighted_emd_loss / (self.num_classes - 1)
        
        # 3. Combinazione finale
        total_loss = ce_loss + self.lambda_ordinal * normalized_emd_loss

        return total_loss, predictions.argmax(dim=1)
    

class OrdinalAgeLoss(nn.Module):
    def __init__(self, num_classes=9, class_frequencies=None, lambda_ordinal=0.3):
        super(OrdinalAgeLoss, self).__init__()
        self.num_classes = num_classes
        self.lambda_ordinal = lambda_ordinal
        
        # Calcolo dei pesi per gestire lo sbilanciamento
        if class_frequencies is not None:
            self.class_weights = class_frequencies
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
        normalized_ordinal_error = weighted_ordinal_loss / ((self.num_classes - 1) ** 2)
        # 3. Combinazione finale
        total_loss = ce_loss + self.lambda_ordinal * normalized_ordinal_error

        return total_loss, predictions.argmax(dim=1)


class FocalLoss():
    def __init__(self,num_classes, weights=None, gamma=2.0):
        self.gamma = gamma
        self.alpha = None
        
        # Normalizzazione dei pesi per mantenere la scala della loss ragionevole
        if weights is not None:
            self.class_weights = weights / weights.max().clamp_min(1e-8)
        else:
            self.class_weights = torch.ones(num_classes).to('cuda')
        
        self.alpha = self.class_weights

    def __call__(self, logit, true_labels, return_predicted_label=False):
        # Calcolo delle probabilità softmax
        probs = F.softmax(logit, dim=1)
        # Probabilità per la classe vera
        p_t = probs[torch.arange(true_labels.size(0)), true_labels]

        # Gestione dei pesi per classe
        if self.alpha is not None:
            alpha_t = self.alpha[true_labels].to(logit.device)
        else:
            alpha_t = 1.0

        # Calcolo della Focal Loss: -alpha * (1 - p_t)^gamma * log(p_t)
        focal_loss = -alpha_t * (1 - p_t) ** self.gamma * torch.log(p_t + 1e-8)  # Aggiunto log(p_t) con stabilità numerica

        loss = focal_loss.mean()

        if return_predicted_label:
            return loss, logit.argmax(dim=1)
        return loss

class CrossEntropyLoss():
    def __init__(self, num_classes, weights=None):
        self.ce = nn.CrossEntropyLoss(weight=weights)        
        self.softmax = nn.Softmax(dim=1)
        if weights is not None:
            self.class_weights = weights
        else:
            self.class_weights = torch.ones(num_classes).to('cuda')
        self.factor = math.log(len(weights)) if weights is not None else 1.0

    def __call__(self, logit, true_labels, return_predicted_label=False):
        loss = self.ce(logit, true_labels)  / self.factor

    
        if return_predicted_label:            
            return loss, logit.argmax(dim=1)
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