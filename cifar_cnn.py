import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, p_drop=0.25):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            nn.Dropout(p=p_drop),
        )
    def forward(self, x):
        return self.block(x)

class CifarCNN(nn.Module):
    """
    A compact yet strong CNN for CIFAR-10:
    - 3 Conv blocks with BN, ReLU, Dropout and 2x2 MaxPool
    - Global Average Pooling head
    - Final linear layer to 10 classes

    Achieves ~80%+ with reasonable training (50+ epochs).
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 64, p_drop=0.20),
            ConvBlock(64, 128, p_drop=0.30),
            ConvBlock(128, 256, p_drop=0.40),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)  # -> (B, C, 1, 1)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        feats = self.features(x)
        pooled = self.gap(feats).view(x.size(0), -1)
        logits = self.classifier(pooled)
        return logits

def build_model(num_classes=10):
    return CifarCNN(num_classes=num_classes)
