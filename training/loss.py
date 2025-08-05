import torch
from torch import nn


class AgeOrdinalLoss():
    def __init__(self, num_classes, ordinal_loss='mae'):
        self.num_classes = num_classes
        self.ordinal_loss = nn.L1Loss() if ordinal_loss == 'mae' else nn.MSELoss()

        self.classes_range = torch.arange(num_classes, device='cuda' if torch.cuda.is_available() else 'cpu').float().unsqueeze(0)

    def __call__(self, logit, true_labels, return_predicted_label=False):
        
        probabilities = torch.softmax(logit, dim=1)
        expected_value = torch.sum(probabilities * self.classes_range, dim=1).float()
        predicted_label = torch.round(expected_value).long()

        loss = self.ordinal_loss(expected_value, true_labels.float())

        if return_predicted_label:
            return loss, predicted_label
        return loss
    
import torch
import torch.nn as nn

class WeightedAgeOrdinalLoss():
    # TODO: to test
    def __init__(self, num_classes, ordinal_loss='mse', weights=None):
        self.num_classes = num_classes
        self.ordinal_loss = nn.L1Loss(reduction='none') if ordinal_loss == 'mae' else nn.MSELoss(reduction='none')
        self.classes_range = torch.arange(num_classes).unsqueeze(0).float()
        device='cuda' if torch.cuda.is_available() else 'cpu'
        if weights is not None:
            self.class_weights = weights
        else:
            self.class_weights = torch.ones(num_classes)

        self.class_weights = self.class_weights.to(device)
        self.classes_range = self.classes_range.to(device)

    def __call__(self, logit, true_labels, return_predicted_label=False):
        probabilities = torch.softmax(logit, dim=1)
        expected_value = torch.sum(probabilities * self.classes_range, dim=1).float()
        predicted_label = torch.round(expected_value).long()

        # Calcola la perdita per esempio
        losses = self.ordinal_loss(expected_value, true_labels.float())

        # Applica il bilanciamento ponderando la perdita
        weighted_losses = losses * self.class_weights[true_labels.long()]

        # Calcola la media ponderata
        loss = weighted_losses.mean()

        if return_predicted_label:
            return loss, predicted_label
        return loss

class CrossEntropyLoss():
    def __init__(self, weights=None):
        self.ce = nn.CrossEntropyLoss(weight=weights)
        self.softmax = nn.Softmax(dim=1)

    def __call__(self, logit, true_labels, return_predicted_label=False):
        loss = self.ce(logit, true_labels)


        if return_predicted_label:
            probabilities = self.softmax(logit)
            predicted_label = torch.argmax(probabilities, dim=1)
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