import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class MarineResNet18(nn.Module):
    def __init__(self, num_classes=32):
        super().__init__()
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.model(x)