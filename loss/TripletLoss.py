import torch
import torch.nn as nn

# Triplet Loss
class Loss(nn.Module):
    def __init__(self, margin=torch.tensor(1.0)):
        super(Loss, self).__init__()
        self.margin = margin.cuda()

    def forward(self, anchor, positive, negative):
        
        distance_positive = torch.sqrt(torch.sum(torch.pow(anchor - positive, 2), dim=1))
        distance_negative = torch.sqrt(torch.sum(torch.pow(anchor - negative, 2), dim=1))
        loss = torch.mean(torch.clamp(distance_positive - distance_negative + self.margin, min=0))
        return loss