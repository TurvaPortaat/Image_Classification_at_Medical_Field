import torch.nn as nn
from torchvision.models import resnet18

class KneeResNet(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        m = resnet18(weights=None)
        m.conv1.in_channels = 1  # 1-kanavainen
        self.backbone = m
        in_f = m.fc.in_features
        self.backbone.fc = nn.Linear(in_f, num_classes)

    def forward(self, x): return self.backbone(x)
