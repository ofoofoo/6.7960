import os
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.model import *
from src.dataloader import *

import torch
import torch.nn.functional as F

def train_epoch(model, train_loader, optimizer, device):
    model.train()
    loss = 0
    total_batches = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = F.cross_entropy(outputs.logits, labels) # classificaiton loss?
        loss.backward()
        optimizer.step()
        loss += loss.item()
        total_batches += 1

    avg_loss = loss / total_batches if total_batches > 0 else 0.0
    return avg_loss

def validate(model, val_loader, device):
    with torch.no_grad():
        model.eval()
        correct = 0
        total = 0
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        accuracy = (correct / total) if total > 0 else 0.0
        return accuracy

@hydra.main(version_base=None, config_path="src", config_name="config")
def train(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    torch.manual_seed(cfg.experiment.seed)

    # WANDB stuff
    if cfg.logging.wandb.enabled:
        wandb.init(
            project=cfg.logging.wandb.project,
            entity=cfg.logging.wandb.entity,
            config=OmegaConf.to_container(cfg, resolve=True)
        )

    device = torch.device(cfg.experiment.device if torch.cuda.is_available() else "cpu")

    model = create_model().to(device)

    train_loader, val_loader = create_dataloader(
        data_dir=cfg.dataset.path, 
        batch_size=cfg.dataset.batch_size,
        num_workers=cfg.dataset.num_workers,
        pin_memory=True
    )

    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=cfg.training.learning_rate,
                                 weight_decay=cfg.optimizer.weight_decay)
    

    scheduler = scheduler = CosineAnnealingLR(optimizer, 
                                              T_max=cfg.training.epochs, 
                                              eta_min=cfg.scheduler.min_lr, 
                                              last_epoch=-1)

    for epoch in range(cfg.training.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_accuracy = validate(model, val_loader, device)

        if cfg.logging.wandb.enabled:
            wandb.log({
                "train_loss": train_loss,
                "val_accuracy": val_accuracy,
                "learning_rate": optimizer.param_groups[0]['lr']
            })

        scheduler.step(val_accuracy)

        # do we want to implement early stopping?

    if cfg.logging.wandb.enabled:
        wandb.finish()


if __name__ == "__main__":
    train()