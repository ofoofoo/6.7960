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
        # print(outputs.logits.shape)
        loss = F.cross_entropy(outputs.logits, labels) # classificaiton loss?
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        total_batches += 1

    avg_loss = train_loss / total_batches if total_batches > 0 else 0
    return avg_loss, total_batches

def validate(model, val_loader, device, processor, dataset_name):
    with torch.no_grad():
        model.eval()
        correct = 0
        total = 0
        val_loss = 0
        num_classes = int(dataset_name[-3:])
        class_correct = torch.tensor([0] * num_classes)
        class_total = torch.tensor([0] * num_classes)
        for images, labels in tqdm(val_loader):
            images = processor(images=images, return_tensors="pt").to(device)
            labels = labels.to(device)
            outputs = model(**images)
            val_loss += F.cross_entropy(outputs.logits, labels, reduction='sum').item()
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # tracking per class metrics:
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_total[label] += 1
                if preds[i].item() == label:
                    class_correct[label] += 1

        accuracy = correct / total if total > 0 else 0
        avg_loss = val_loss / total if total > 0 else 0

        # per class accuracies:
        class_accuracies = torch.tensor([])
        for classes in range(num_classes):
            if class_total[classes] > 0:
                class_accuracies = torch.cat((class_accuracies, torch.tensor([class_correct[classes] / class_total[classes]])), dim=0)
            else:
                class_accuracies = torch.cat((class_accuracies, torch.tensor([0])), dim=0)  # no samples of this class in val set

        return accuracy, avg_loss, class_accuracies


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
    num_classes = int(cfg.dataset.name[-3:])
    model = create_model(num_encoder_layers_frozen=cfg.training.num_layers_frozen, 
                         num_classes=num_classes,
                         pretrained=cfg.training.from_pretrained).to(device)
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
        pin_memory=True,
        subset_weight=cfg.dataset.subset_weight
    )
    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=cfg.training.learning_rate,
                                 weight_decay=cfg.optimizer.weight_decay)
    

    scheduler = CosineAnnealingLR(optimizer, 
                                T_max=cfg.training.epochs)

    total_data = 0
    if cfg.training.early_stopping:
        early_stopper = EarlyStopping(cfg.training.patience)

    val_accuracy, val_loss, _ = validate(model, val_loader, device, processor, cfg.dataset.name) # track metrics before first train epoch
    if cfg.logging.wandb.enabled:
        wandb.log({
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        })

    for epoch in range(cfg.training.epochs):
        print(f"train epoch {epoch}")
        train_loss, train_total = train_epoch(model, train_loader, optimizer, device, processor)
        print(f"val epoch {epoch}")
        val_accuracy, val_loss, class_accuracies = validate(model, val_loader, device, processor, cfg.dataset.name)
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
        
        if cfg.training.early_stopping:
            early_stopper(val_loss)
            if early_stopper.stop:
                print("=============Early stopping============")
                break
                
        # IF IT IS TIME TO DO THE MIXTURE TRAINING, THEN ONLY CHANGE THE DATALOADER:    
        if (epoch + 1) % cfg.training.num_epochs_mix == 0:
            _, bad_class_ids = torch.topk(class_accuracies, cfg.training.num_bad_classes)
            print(bad_class_ids.shape)
            train_loader, _ = create_dataloader(dataset=cfg.dataset.name,
                                                data_dir=cfg.dataset.path, 
                                                batch_size=cfg.dataset.batch_size,
                                                val_batch_size=cfg.dataset.val_batch_size,
                                                pin_memory=True,
                                                bad_class_ids=bad_class_ids,
                                                subset_weight=cfg.dataset.subset_weight)
        elif cfg.training.mix_for_only_one and epoch > 0 and epoch % cfg.training.num_epochs_mix == 0:
            # reset train loader to original train loader if we only want to mix for one epoch each time
            train_loader, _ = create_dataloader(dataset=cfg.dataset.name,
                                                data_dir=cfg.dataset.path, 
                                                batch_size=cfg.dataset.batch_size,
                                                val_batch_size=cfg.dataset.val_batch_size,
                                                pin_memory=True,
                                                subset_weight=cfg.dataset.subset_weight)

    if cfg.logging.wandb.enabled:
        wandb.finish()

if __name__ == "__main__":
    train()