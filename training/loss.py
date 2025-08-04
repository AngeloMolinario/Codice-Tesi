import torch
from torch import nn


class AgeOrdinalLoss():
    def __init__(self, num_classes, ordinal_loss='mae'):
        self.num_classes = num_classes
        self.ordinal_loss = nn.L1Loss() if ordinal_loss == 'mae' else nn.MSELoss()

        self.classes_range = torch.arange(num_classes).float()

    def __call__(self, logit, true_labels, return_predicted_label=False):
        self.classes_range = self.classes_range.to(logit.device)
        
        probabilities = torch.softmax(logit, dim=1)
        expected_value = torch.sum(probabilities * self.classes_range.unsqueeze(0), dim=1).float()
        predicted_label = torch.round(expected_value).long()

        loss = self.ordinal_loss(expected_value, true_labels.float())

        if return_predicted_label:
            return loss, predicted_label
        return loss
    
import torch
import torch.nn as nn

class _WeightedAgeOrdinalLoss():
    # TODO: to test
    def __init__(self, num_classes, ordinal_loss='mae', weights=None):
        self.num_classes = num_classes
        self.ordinal_loss = nn.L1Loss(reduction='none') if ordinal_loss == 'mae' else nn.MSELoss(reduction='none')
        self.classes_range = torch.arange(num_classes).unsqueeze(0).float()

        if weights is not None:
            self.class_weights = weights
        else:
            self.class_weights = torch.ones(num_classes)


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
    def __init__(self):
        self.ce = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)
    def __call__(self, logit, true_labels, return_predicted_label=False):
        loss = self.ce(logit, true_labels)


        if return_predicted_label:
            probabilities = self.softmax(logit)
            predicted_label = torch.argmax(probabilities, dim=1)
            return loss, predicted_label
        return loss

