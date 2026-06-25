"""ResNet-18 adapted for driving action classification.

Input:  (B, 3, 180, 320) RGB, normalized with ImageNet mean/std
Output: (B, 7) logits (no softmax — handled by CrossEntropyLoss)
"""
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def create_model(num_classes: int = 7, dropout: float = 0.5,
                 freeze_backbone: bool = True) -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features  # 512
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes),
    )
    return model
