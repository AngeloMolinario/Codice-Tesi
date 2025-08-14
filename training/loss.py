import torch
from torch import nn

import math
import torch
import torch.nn as nn

class OrdinalLoss(nn.Module):
    def __init__(self, num_classes=9, alpha=0.1, max_dataset_age=108.0, weights=None, 
                 rare_classes=None, rare_class_min_weight=8.0,
                 central_classes=None, central_weight_multiplier=1.3):
        """
        num_classes: Numero di gruppi d'età (default=9)
        alpha: Peso per componente ordinale (0.1 ora accettabile grazie al bilanciamento)
        max_dataset_age: Età massima presente nel dataset (108.0)
        weights: Pesi per gestire sbilanciamento dataset (opzionale)
        rare_classes: Indici delle classi rare (es. [0, 8] per "0-2" e "70+")
        rare_class_min_weight: Peso minimo garantito per classi rare
        central_classes: Indici delle classi centrali da pesare di più (es. [3,4,5,6])
        central_weight_multiplier: Moltiplicatore per aumentare il peso delle classi centrali
        """
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.max_dataset_age = max_dataset_age
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 1. CENTRI AGGIORNATI con età massima 108 anni
        self.age_centers = torch.tensor([
            1.0,    # 0-2
            6.0,    # 3-9
            14.5,   # 10-19
            24.5,   # 20-29
            34.5,   # 30-39
            44.5,   # 40-49
            54.5,   # 50-59
            64.5,   # 60-69
            89.0    # 70-108
        ]).float().to(device)
        
        # 2. Massima differenza in anni
        self.max_age_diff = self.age_centers[-1] - self.age_centers[0]  # 88.0 anni
        
        # 3. Dimensioni reali dei bin
        bin_sizes = torch.tensor([
            3.0, 7.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 38.0
        ]).float().to(device)
        
        # 4. Pesi per lo sbilanciamento dei BIN
        bin_weights = 1.0 / bin_sizes
        bin_weights = bin_weights / bin_weights.sum() * num_classes
        
        # 5. GESTIONE CLASSI RARE
        self.rare_classes = rare_classes if rare_classes is not None else [0, 8]
        
        # 6. COMBINA con pesi per lo sbilanciamento del DATASET
        if weights is not None:
            assert len(weights) == num_classes, f"weights deve avere lunghezza {num_classes}"
            dataset_weights = weights.float().to(device)
            combined_weights = bin_weights * dataset_weights
        else:
            combined_weights = bin_weights.clone()

        # 7. APPLICA MIN_WEIGHT per classi rare
        for cls_idx in self.rare_classes:
            combined_weights[cls_idx] = max(combined_weights[cls_idx], rare_class_min_weight)

        # 8. APPLICA PESO AGGIUNTIVO PER CLASSI CENTRALI
        self.central_classes = central_classes if central_classes is not None else [3, 4, 5, 6]  # 20-29 a 50-59
        self.central_weight_multiplier = central_weight_multiplier
        
        for cls_idx in self.central_classes:
            if cls_idx not in self.rare_classes:  # Evita di sovrappesare classi già pesanti
                combined_weights[cls_idx] = combined_weights[cls_idx] * central_weight_multiplier

        # 9. Normalizza (media=1 come in CrossEntropyLoss)
        self.class_weights = combined_weights / combined_weights.mean()
        
        # 10. Inizializza CE con pesi combinati
        self.ce = nn.CrossEntropyLoss(weight=self.class_weights)

    def forward(self, logit, true_labels, return_predicted_label=False):
        # 1. Cross-Entropy NORMALIZZATA
        ce_loss = self.ce(logit, true_labels) / math.log(self.num_classes)
        
        # 2. Calcola età predetta
        probs = torch.softmax(logit, dim=1)
        predicted_ages = probs @ self.age_centers  # [B]
        
        # 3. Età vera
        true_ages = self.age_centers[true_labels]  # [B]
        
        # 4. Calcola EAE normalizzato
        age_diff = (predicted_ages - true_ages).abs()
        eae_loss = age_diff.mean() / self.max_age_diff  # [0,1]
        
        # 5. Combina con α
        loss = ce_loss + self.alpha * eae_loss
        
        # Restituisce ESATTAMENTE come nella tua implementazione
        if return_predicted_label:
            pred = torch.argmax(logit, dim=1)
            return loss, pred
        
        return loss

class OrdinalLoss_l2(nn.Module):
    def __init__(self, num_classes=9, alpha=0.1, max_dataset_age=108.0, weights=None, 
                 rare_classes=None, rare_class_min_weight=8.0):
        """
        num_classes: Numero di gruppi d'età (default=9)
        alpha: Peso per componente ordinale (0.05 consigliato)
        max_dataset_age: Età massima presente nel dataset (108.0)
        weights: Pesi per gestire sbilanciamento dataset (opzionale)
        rare_classes: Indici delle classi rare (es. [0, 8] per "0-2" e "70+")
        rare_class_min_weight: Peso minimo garantito per classi rare
        """
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.max_dataset_age = max_dataset_age
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 1. CENTRI AGGIORNATI con età massima 108 anni
        # Nota: "70+" ora va da 70 a 108 → centro = (70+108)/2 = 89.0
        self.age_centers = torch.tensor([
            1.0,    # 0-2
            6.0,    # 3-9
            14.5,   # 10-19
            24.5,   # 20-29
            34.5,   # 30-39
            44.5,   # 40-49
            54.5,   # 50-59
            64.5,   # 60-69
            89.0    # 70-108 (non più 85.0!)
        ]).float().to(device)
        
        # 2. Massima differenza in anni (aggiornata con età massima 108)
        self.max_age_diff = self.age_centers[-1] - self.age_centers[0]  # 88.0 anni
        
        # 3. Dimensioni reali dei bin (aggiornate)
        bin_sizes = torch.tensor([
            3.0,   # 0-2
            7.0,   # 3-9
            10.0,  # 10-19
            10.0,  # 20-29
            10.0,  # 30-39
            10.0,  # 40-49
            10.0,  # 50-59
            10.0,  # 60-69
            38.0   # 70-108 (non più 30.0!)
        ]).float().to(device)
        
        # 4. Pesi per lo sbilanciamento dei BIN
        bin_weights = 1.0 / bin_sizes
        bin_weights = bin_weights / bin_weights.sum() * num_classes
        
        # 5. GESTIONE CLASSI RARE (bambini e anziani)
        self.rare_classes = rare_classes if rare_classes is not None else [0, 8]  # Default: 0-2 e 70+
        
        # 6. COMBINA con pesi per lo sbilanciamento del DATASET
        if weights is not None:
            # Verifica lunghezza pesi
            assert len(weights) == num_classes, f"weights deve avere lunghezza {num_classes}"
            dataset_weights = weights.float().to(device)
            
            # Combina pesi: bin_weights * dataset_weights
            combined_weights = bin_weights * dataset_weights
            
            # 7. APPLICA MIN_WEIGHT per classi rare
            for cls_idx in self.rare_classes:
                combined_weights[cls_idx] = max(combined_weights[cls_idx], rare_class_min_weight)
            
            # Normalizza (media=1 come in CrossEntropyLoss)
            self.class_weights = combined_weights / combined_weights.mean()
        else:
            # Se nessun peso fornito, usa solo bin_weights ma con min_weight per classi rare
            self.class_weights = bin_weights.clone()
            for cls_idx in self.rare_classes:
                self.class_weights[cls_idx] = max(self.class_weights[cls_idx], rare_class_min_weight)
            self.class_weights = self.class_weights / self.class_weights.mean()
        
        # 8. Inizializza CE con pesi combinati
        self.ce = nn.CrossEntropyLoss(weight=self.class_weights)

    def forward(self, logit, true_labels, return_predicted_label=False):
        # 1. Cross-Entropy NORMALIZZATA
        ce_loss = self.ce(logit, true_labels) / math.log(self.num_classes)
        
        # 2. Calcola età predetta con CENTRI AGGIORNATI
        probs = torch.softmax(logit, dim=1)
        predicted_ages = probs @ self.age_centers  # [B]
        
        # 3. Età vera con CENTRI AGGIORNATI
        true_ages = self.age_centers[true_labels]  # [B]
        
        # 4. Calcola EAE con max_age_diff AGGIORNATO (88.0 anni)
        age_diff = (predicted_ages - true_ages).abs()
        eae_loss = age_diff.mean() / self.max_age_diff  # Normalizzato in [0,1]
        
        # 5. Combina con α
        loss = ce_loss + self.alpha * eae_loss
        
        # Restituisce ESATTAMENTE come nella tua implementazione
        if return_predicted_label:
            pred = torch.argmax(logit, dim=1)
            return loss, pred
        
        return loss

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