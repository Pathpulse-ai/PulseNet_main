import torch
import torch.nn as nn

class DetectionHead(nn.Module):

    def __init__(self, in_channels=512, num_classes=20):
        super().__init__()
        self.cls_conv = nn.Conv2d(in_channels, num_classes, kernel_size=3, padding=1)
        self.reg_conv = nn.Conv2d(in_channels, 4, kernel_size=3, padding=1)
        

    def forward(self, features):
        
        cls_outputs = []
        reg_outputs = []
        for feat in features:
            cls_out = self.cls_conv(feat)  # [B, num_classes, H, W]
            reg_out = self.reg_conv(feat)  # [B, 4, H, W]
            cls_outputs.append(cls_out)
            reg_outputs.append(reg_out)
        return cls_outputs, reg_outputs