import torch
import unittest
from models.pulsenet import PulseNet

class TestPulseNet(unittest.TestCase):
    def test_forward_pass(self):
        model = PulseNet(num_classes=3)
        model.eval()
        x = torch.randn(2, 3, 640, 640)  # batch of 2
        with torch.no_grad():
            cls_outputs, reg_outputs = model(x)
        self.assertEqual(len(cls_outputs), 1)  # FPN returns a single feature map in this stub
        self.assertEqual(len(reg_outputs), 1)
        print("Test forward pass passed.")

if __name__ == "__main__":
    unittest.main()