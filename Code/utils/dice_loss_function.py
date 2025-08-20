
# DICE LOSS AND GENERALIZAED LOSS FUNCTION

import torch
import torch.nn as nn
import torch.nn.functional as F

class dice_loss(nn.Module):
    def __init__(self, class_weights=None, smooth=1e-6):
        super(dice_loss, self).__init__()
        self.smooth = smooth
        if class_weights is None:
            self.class_weights = class_weights
        else:
            self.class_weights = class_weights.clone().detach()

    def forward(self, inputs, targets):
        inputs = F.softmax(inputs, dim=1)
        
        inputs_flat = inputs.view(inputs.size(0), inputs.size(1), -1)
        targets_flat = targets.view(targets.size(0), targets.size(1), -1)

        intersection = torch.sum(inputs_flat * targets_flat, dim=-1)
        cardinality = torch.sum(inputs_flat + targets_flat, dim=-1)
        
        if self.class_weights is not None:
            dice_loss = 2.0 * torch.sum(self.class_weights * intersection, dim=-1) / torch.sum(self.class_weights * cardinality, dim=-1)
        else:
            dice_loss = 2.0 * torch.sum(intersection, dim=-1) / torch.sum(cardinality, dim=-1)
        dice_loss = torch.sum(1 - dice_loss)
        
        return torch.sum(dice_loss)/inputs_flat.shape[0]