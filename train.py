import os
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForImageClassification, ViTImageProcessor, ViTForImageClassification

from src.model import *
from src.dataloader import *

import torch
import torch.nn.functional as F

np.random.seed(42)
torch.manual_seed(42)

def train_epoch(model, train_loader, optimizer, device, processor):
    model.train()
    train_loss = 0
    total_batches = 0
    for images, labels in tqdm(train_loader):
        images = processor(images=images, return_tensors="pt").to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(**images)
        print(outputs.logits.shape)
        loss = F.cross_entropy(outputs.logits, labels) # classificaiton loss?
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        total_batches += 1

    avg_loss = train_loss / total_batches if total_batches > 0 else 0
    return avg_loss, total_batches

def validate(model, val_loader, device, processor):
    with torch.no_grad():
        model.eval()
        correct = 0
        total = 0
        val_loss = 0
        for images, labels in tqdm(val_loader):
            images = processor(images=images, return_tensors="pt").to(device)
            labels = labels.to(device)
            outputs = model(**images)
            val_loss += F.cross_entropy(outputs.logits, labels).item()
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        accuracy = correct / total if total > 0 else 0
        avg_loss = val_loss / total if total > 0 else 0
        return accuracy, avg_loss

# implement early stopping class
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_value = None
        self.counter = 0
        self.stop = False

    def __call__(self, current_value):
        if self.best_value is None: # initialize to current value if first epoch
            self.best_value = current_value
            return
        if current_value < (self.best_value - self.min_delta):
            self.best_value = current_value
            self.counter = 0
        else: 
            self.counter += 1
            print(f"No improvement for {self.counter}/{self.patience} epochs")
            if self.counter >= self.patience:
                print("Stopping training")
                self.stop = True

@hydra.main(version_base=None, config_path="src", config_name="config")
def train(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    torch.manual_seed(cfg.experiment.seed)
    # WANDB stuff
    if cfg.logging.wandb.enabled:
        wandb.init(
            project=cfg.logging.wandb.project,
            config=OmegaConf.to_container(cfg, resolve=True)
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_model(num_encoder_layers_frozen=12).to(device)
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224",)
    print(model)

    #train_loader, val_loader = create_dummy_dataloader()
    # train_loader, val_loader, test_loader = create_dataloader(
    #     data_dir=cfg.dataset.path, 
    #     batch_size=cfg.dataset.batch_size,
    #     pin_memory=True
    # )
    train_loader, val_loader = create_dataloader(
        dataset=cfg.dataset.name,
        data_dir=cfg.dataset.path, 
        batch_size=cfg.dataset.batch_size,
        val_batch_size=cfg.dataset.val_batch_size,
        pin_memory=True
    )
    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=cfg.training.learning_rate,
                                 weight_decay=cfg.optimizer.weight_decay)
    

    scheduler = CosineAnnealingLR(optimizer, 
                                T_max=cfg.training.epochs)

    total_data = 0
    early_stopper = EarlyStopping(cfg.training.patience)

    val_accuracy, val_loss = validate(model, val_loader, device, processor) # track metrics before first train epoch
    if cfg.logging.wandb.enabled:
        wandb.log({
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        })

    for epoch in range(cfg.training.epochs):
        train_loss, train_total = train_epoch(model, train_loader, optimizer, device, processor)
        val_accuracy, val_loss = validate(model, val_loader, device, processor)
        total_data += train_total
        if cfg.logging.wandb.enabled:
            wandb.log({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "total_datapoints": total_data
            })

        scheduler.step()
        
        early_stopper(val_loss)
        if early_stopper.stop:
            print("=============Early stopping============")
            break

    if cfg.logging.wandb.enabled:
        wandb.finish()

if __name__ == "__main__":
    train()