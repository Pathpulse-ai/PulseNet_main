import yaml
import argparse
import torch

from data.datasets import CustomDetectionDataset
from data.transforms import DetectionTransforms
from models.pulsenet import PulseNet
from engine.evaluator import Evaluator

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PulseNet")
    parser.add_argument("--config", default="configs/default_config.yaml", help="Path to config file")
    parser.add_argument("--weights", default="pulsenet.pth", help="Path to trained weights")
    return parser.parse_args()

def main():
    args = parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Dataset
    val_dataset = CustomDetectionDataset(
        root=cfg["data"]["val_dataset_path"],
        transforms=DetectionTransforms(resize=cfg["data"]["input_size"])
    )

    # Model
    model = PulseNet(
        backbone_name=cfg["model"]["backbone"],
        pretrained=False,
        num_classes=cfg["model"]["num_classes"]
    )

    # Load weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device)

    evaluator = Evaluator(model, val_dataset, cfg)
    evaluator.evaluate()

if __name__ == "__main__":
    main()