import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class OrdinalAgeLossEMD(nn.Module):
    '''
        This class compute the loss for the Age group classification task.
        The loss is made of 2 parts, a weighted cross-entropy as the main loss and a weighted Earth Mover's Distance (EMD) loss as the auxiliary loss.
        The CE is used because the problem is a multiclass classification problem where the main performance evaluator is the accuracy. Given the nature
        of the CE wrongly classify a sample of a 10-19 with 70+ hold the same weight of wrongly classy it as 20-29 but in age classification problem
        this is not true as the cost of misclassifying an age group can vary significantly. For example, misclassifying a 10-19 year old as 70+ is likely
        to be more detrimental than misclassifying them as 20-29. The EMD loss helps to address this issue by considering the ordinal nature of age groups.
    '''
    def __init__(self, num_classes=9, class_frequencies=None, lambda_ordinal=0.3, use_squared_emd=False):
        super(OrdinalAgeLossEMD, self).__init__()
        self.num_classes = num_classes
        self.lambda_ordinal = lambda_ordinal
        self.use_squared_emd = use_squared_emd
        
        if class_frequencies is not None:
            self.class_weights = class_frequencies
        else:
            self.class_weights = torch.ones(num_classes).to('cuda')

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
        ce_loss = F.cross_entropy(
            predictions, 
            targets, 
            weight=self.class_weights.to(predictions.device)
        )
                
        probs = F.softmax(predictions, dim=1)
                
        weighted_emd_loss = self.compute_weighted_emd_loss(probs, targets, squared=self.use_squared_emd)
                
        if self.use_squared_emd:            
            normalized_emd_loss = weighted_emd_loss / ((self.num_classes - 1) ** 2)
        else:        
            normalized_emd_loss = weighted_emd_loss / (self.num_classes - 1)
        
        total_loss = ce_loss + self.lambda_ordinal * normalized_emd_loss

        return total_loss
    


class CrossEntropyLoss():
    def __init__(self, num_classes, weights=None):
        self.ce = nn.CrossEntropyLoss(weight=weights)        
        self.softmax = nn.Softmax(dim=1)
        if weights is not None:
            self.class_weights = weights
        else:
            self.class_weights = torch.ones(num_classes).to('cuda')
        self.factor = math.log(len(weights)) if weights is not None else 1.0

    def __call__(self, logit, true_labels):
        loss = self.ce(logit, true_labels)  / self.factor

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