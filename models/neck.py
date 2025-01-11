import torch.nn as nn

class SimpleFPN(nn.Module):
    """
    A very simple Feature Pyramid Neck (optional).
    For demonstration, we just return the original features.
    """
    def __init__(self, in_channels=512):
        super().__init__()
        # You can implement a more advanced feature pyramid or aggregator here.
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        # x: [batch_size, in_channels, H, W]
        return [self.conv(x)]