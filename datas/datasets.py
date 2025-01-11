import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class CustomDetectionDataset(Dataset):
    """
    A minimal custom dataset for object detection.
    You will need to implement __getitem__ to return:
    (image, target), where:
      - image is a Tensor [C, H, W]
      - target is a dict containing 'boxes' and 'labels'
    """
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # Suppose we have an annotation list or structure
        self.imgs = sorted(os.listdir(os.path.join(root, "images")))
        self.labels = sorted(os.listdir(os.path.join(root, "labels")))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        label_path = os.path.join(self.root, "labels", self.labels[idx])

        image = Image.open(img_path).convert("RGB")
        # Parse your label file (for bounding boxes, class IDs, etc.)
        # For example:
        boxes = torch.tensor([[10, 20, 100, 150]], dtype=torch.float32)
        labels = torch.tensor([1], dtype=torch.int64)

        target = {
            "boxes": boxes,    # [N, 4]
            "labels": labels   # [N]
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target
