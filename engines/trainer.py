import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import time

class Trainer:
    def __init__(self, model, train_dataset, cfg):
        self.model = model
        self.cfg = cfg
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=cfg["train"]["batch_size"],
            shuffle=True,
            num_workers=cfg["train"]["num_workers"],
            collate_fn=self.collate_fn
        )
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=cfg["train"]["lr"],
            momentum=cfg["train"]["momentum"],
            weight_decay=cfg["train"]["weight_decay"]
        )
        self.model.train()

    def collate_fn(self, batch):
        images, targets = list(zip(*batch))
        images = torch.stack(images, dim=0)
        # targets is a tuple of dicts; keep them as is
        return images, targets

    def train_one_epoch(self, epoch):
        epoch_loss = 0.0
        start = time.time()
        for i, (images, targets) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            loss_dict = self.model(images, targets)
            loss = sum(loss_dict.values())
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch [{epoch}] - Loss: {epoch_loss/len(self.train_loader):.4f}, "
              f"Time: {time.time()-start:.2f}s")

    def fit(self, epochs):
        for epoch in range(1, epochs+1):
            self.train_one_epoch(epoch)