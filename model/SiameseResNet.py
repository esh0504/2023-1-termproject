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

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

