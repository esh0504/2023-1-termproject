import torch
import torch.nn as nn
import torchvision.models as models

class SiameseResNet(nn.Module):
    def __init__(self):
        super(SiameseResNet, self).__init__()
        self.resnet = models.resnet18(weights=True)
        self.fc = nn.Linear(1000, 1024)

    def forward_once(self, x):
        x = self.resnet(x)
        x = self.fc(x)
        return x

    def forward(self, anchor, neg, pos):
        anchor_output = self.forward_once(anchor)
        neg_output = self.forward_once(neg)
        pos_output = self.forward_once(pos)
        
        return anchor_output, neg_output, pos_output
