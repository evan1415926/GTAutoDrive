"""ResNet-18 adapted for driving action classification.

Input:  (B, 3, 180, 320) RGB, range [0, 255]
Output: (B, 7) logits (no softmax — handled by CrossEntropyLoss)
"""
import torch
import torch.nn as nn
from torchvision.models import resnet18


def create_model(num_classes: int = 7, dropout: float = 0.5) -> nn.Module:
    model = resnet18(weights=None)

    # Replace first conv: standard 7x7→64 is fine for 320x180
    # Replace final FC: 512→num_classes
    in_features = model.fc.in_features  # 512
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes),
    )
    return model
