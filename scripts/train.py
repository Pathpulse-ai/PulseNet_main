import yaml
import torch
import argparse
import os

from data.datasets import CustomDetectionDataset
from data.transforms import DetectionTransforms
from models.pulsenet import PulseNet
from engine.trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser(description="Train PulseNet")
    parser.add_argument("--config", default="configs/default_config.yaml", help="Path to config file")
    return parser.parse_args()

def main():
    args = parse_args()
    
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Datasets
    train_dataset = CustomDetectionDataset(
        root=cfg["data"]["train_dataset_path"],
        transforms=DetectionTransforms(resize=cfg["data"]["input_size"])
    )

    # Model
    model = PulseNet(
        backbone_name=cfg["model"]["backbone"],
        pretrained=cfg["model"]["pretrained"],
        num_classes=cfg["model"]["num_classes"]
    )

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Trainer
    trainer = Trainer(model, train_dataset, cfg)
    trainer.fit(cfg["train"]["epochs"])

if __name__ == "__main__":
    main()