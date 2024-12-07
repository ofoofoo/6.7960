import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import hydra
from omegaconf import DictConfig
from tqdm import tqdm
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification, ViTImageProcessor, ViTForImageClassification

from src.dataloader import *
from src.model import create_model

def eval_baseline_imagenet():
    test_hf = load_dataset("imagenet-1k", split='validation', streaming=True)
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224").eval().cuda()
    top1_correct = 0
    total = 0

    for example in tqdm(test_hf):
        image, label = example["image"], example["label"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        print(type(image))
        inputs = processor(images=image, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        preds = torch.softmax(logits, dim=-1)
        top1 = torch.topk(preds, k=1, dim=-1)
        top1_indices = top1.indices[0].cpu().numpy()
        total += 1
        if top1_indices[0] == label:
            top1_correct += 1
        print(top1_correct / total)
    top1_acc = top1_correct / total
    print("Top-1 Accuracy:", top1_acc)

# THIS EVAL IS NOT RIGHT (PROBABLY)
def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            show_image(images[0])
            
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0.0

@hydra.main(version_base=None, config_path="src", config_name="config")
def main(cfg: DictConfig):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model().to(device)
    # train_loader, val_loader, test_loader = create_dataloader(
    #     data_dir=cfg.dataset.path, 
    #     batch_size=cfg.dataset.batch_size,
    #     pin_memory=True
    # )

    # accuracy = evaluate(model, test_loader, device)
    # print(f"Test Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    eval_baseline_imagenet()