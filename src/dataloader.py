import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from datasets import load_dataset

from PIL import Image

import numpy as np

#  HF datasets --> PyTorch datasets
class HFDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.hf_dataset)
    
    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        image = item['image']
        label = item['label']
        image = self.transform(image)
        
        return image, label

class DummyImageNetDataset(Dataset):
    def __init__(self, num_samples=10000, num_classes=1000, image_size=(3, 224, 224)):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.image_size = image_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = torch.randn(*self.image_size)
        label = torch.randint(0, self.num_classes, (1,)).item()
        return image, label

def create_dummy_dataloader(batch_size=64, num_workers=4, pin_memory=True):
    dataset = DummyImageNetDataset(num_samples=10000, num_classes=1000, image_size=(3, 224, 224))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, pin_memory=pin_memory)
    return loader, loader
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset

def create_dataloader(data_dir, batch_size=64, pin_memory=True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = load_dataset("GATE-engine/mini_imagenet", split='train')
    val_dataset = load_dataset("GATE-engine/mini_imagenet", split='validation')
    test_dataset = load_dataset("GATE-engine/mini_imagenet", split='test')
    print("Hi HF to Pytorch takes a long time")
    train_dataset = HFDataset(train_dataset, transform=train_transform)
    val_dataset = HFDataset(val_dataset, transform=val_transform)
    test_dataset = HFDataset(test_dataset, transform=val_transform)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        pin_memory=pin_memory
    )
    return train_loader, val_loader, test_loader