import torch
import torch.nn as nn
import numpy as np

# seperate Triplet Loss
class Loss(nn.Module):
    def __init__(self, pos_margin=torch.tensor(0.1), neg_margin=torch.tensor(5.0)):
        super(Loss, self).__init__()
        self.pos_margin = pos_margin.cuda()
        self.neg_margin = neg_margin.cuda()
        
    def forward(self, anchor, positive, negative):
        distance_positive = torch.sqrt(torch.sum(torch.pow(anchor - positive, 2), dim=1))
        distance_negative = torch.sqrt(torch.sum(torch.pow(anchor - negative, 2), dim=1))
        
        pos_loss = torch.mean(torch.clamp(distance_positive - self.pos_margin, min = 0))
        neg_loss = torch.mean(torch.clamp(self.neg_margin - distance_negative, min = 0))
        
        loss = pos_loss + neg_loss
        return loss, pos_loss, neg_loss