import torch
import torch.nn as nn
from .backbone import build_backbone
from .neck import SimpleFPN
from .head import DetectionHead

class PulseNet(nn.Module):
    def __init__(self, backbone_name="resnet18", pretrained=True, num_classes=20):
        super().__init__()
        self.backbone, out_channels = build_backbone(backbone_name, pretrained)
        self.neck = SimpleFPN(out_channels)
        self.head = DetectionHead(in_channels=out_channels, num_classes=num_classes)

    def forward(self, images, targets=None):
        """
        images: [B, 3, H, W]
        targets: ground-truth boxes, labels (for training)
        returns:
            In training mode: loss_dict
            In inference mode: predictions (class_scores, boxes)
        """
        features = self.backbone(images)
        fpn_features = self.neck(features)
        cls_outputs, reg_outputs = self.head(fpn_features)

        if self.training:
            # Compute losses
            loss_dict = {
                "loss_cls": torch.tensor(0.0, device=images.device),
                "loss_reg": torch.tensor(0.0, device=images.device),
            }
            # TODO: Implement classification & regression loss computations
            return loss_dict
        else:
            
            # For now, return raw outputs
            return cls_outputs, reg_outputs