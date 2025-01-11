import torch
import torch.nn as nn
import torchvision.models as models

def build_backbone(backbone_name="resnet18", pretrained=True):
    """
    Creates a backbone (e.g., ResNet) and returns the feature-extracting layers.
    """
    if backbone_name == "resnet18":
        net = models.resnet18(pretrained=pretrained)
        # Remove the classification head (fc layer)
        layers = list(net.children())[:-2]  # up to the last conv block
        backbone = nn.Sequential(*layers)
        out_channels = 512  # resnet18 final layer channels
    else:
        raise ValueError(f"Backbone {backbone_name} not implemented.")

    return backbone, out_channels