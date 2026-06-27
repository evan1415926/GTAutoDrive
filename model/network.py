"""ResNet-18 adapted for driving action classification.

Input:  (B, C, 180, 320) — C=3 single-frame, C=12 4-frame stacked
Output: (B, 7) logits (no softmax — handled by CrossEntropyLoss)
"""
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def create_model(num_classes: int = 7, dropout: float = 0.5,
                 freeze_backbone: bool = True,
                 input_channels: int = 3) -> nn.Module:
    """Build ResNet-18 with optional multi-channel input for frame stacking.

    Args:
        num_classes: output classes (default 7)
        dropout: dropout rate on FC head
        freeze_backbone: if True, freeze layer1+layer2, train layer3+layer4
        input_channels: 3 for single-frame, 12 for 4-frame stacked
    """
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    # Adapt first conv for multi-channel input (frame stacking)
    if input_channels != 3:
        old_conv = model.conv1
        old_weight = old_conv.weight.data       # (64, 3, 7, 7)
        repeat = input_channels // 3
        new_weight = old_weight.repeat(1, repeat, 1, 1) / float(repeat)
        new_conv = nn.Conv2d(
            input_channels, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        new_conv.weight.data = new_weight
        if old_conv.bias is not None:
            new_conv.bias.data = old_conv.bias.data
        model.conv1 = new_conv
        print(f"[MODEL] Adapted conv1: 3 -> {input_channels} channels "
              f"(weights repeated {repeat}x, divided by {repeat})")

    if freeze_backbone:
        # Freeze only early layers (ImageNet low-level features transfer well).
        # Unfreeze layer3 + layer4 so the model can learn driving-specific
        # high-level features (road curvature, lane position, obstacles).
        for name, param in model.named_parameters():
            if 'layer3' in name or 'layer4' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"[MODEL] Partial unfreeze: {trainable:,}/{total:,} params "
              f"trainable ({100*trainable/total:.1f}%)")

    in_features = model.fc.in_features  # 512
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes),
    )
    return model
