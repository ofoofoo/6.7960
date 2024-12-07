import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from datasets import load_dataset

from PIL import Image

import numpy as np

np.random.seed(42)
torch.manual_seed(42)

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

def create_dataloader(dataset="Food101", data_dir = "data", batch_size=64, val_batch_size=512, pin_memory=True):

    # normalize = transforms.Normalize(mean=image_mean, std=image_std)
    # _train_transforms = transforms.Compose(
    #     [
    #         transforms.RandomResizedCrop(size),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.normalize,
    #     ]
    # )

    # _val_transforms = Compose(
    #     [
    #         transforms.Resize(size),
    #         transforms.CenterCrop(size),
    #         transforms.ToTensor(),
    #         transforms.normalize,
    #     ]
    # )

    if dataset == "Food101": # FOOD101 HAS NO VALIDATION SET, WE NEED TO SPLIT MANUALLY
        train_dataset = datasets.Food101(
            root=data_dir, 
            split='train',
            download=True
        )

        test_dataset = datasets.Food101(
        root=data_dir, 
        split='test',
        download=True
        )
    
    elif dataset == "Flowers102": # FLOWERS102 DOES HAVE A VALIDATION SET
        train_dataset = datasets.Flowers102(
            root=data_dir, 
            split='train',
            download=True
        )

        # val_dataset = datasets.Flowers102(
        #     root=data_dir,
        #     split='val',
        #     download=True
        # )

        test_dataset = datasets.Flowers102(
        root=data_dir, 
        split='test',
        download=True
        )
    
    elif dataset == "Stanford196": 
        train_dataset = datasets.StanfordCars(
            root=data_dir, 
            split='train',
            download=True
        )

        test_dataset = datasets.StanfordCars(
        root=data_dir, 
        split='test',
        download=True
        )
    
    subset_size = int(len(train_dataset) * 1)  # subset the train set
    indices = np.random.choice(len(train_dataset), subset_size, replace=False)  # randomly select points
    train_dataset = Subset(train_dataset, indices)

    subset_size = int(len(test_dataset) * 1)  # subset the test set
    indices = np.random.choice(len(test_dataset), subset_size, replace=False)  # randomly select points
    test_dataset = Subset(test_dataset, indices)


    def collate_fn(batch):
        images = [item[0] for item in batch]
        labels = [item[1] for item in batch] 
        return images, torch.tensor(labels) # keep images raw, convert labels to tensor
    
    train_loader = DataLoader(train_dataset, 
                              batch_size=batch_size, 
                              shuffle=True, 
                              pin_memory=pin_memory,
                              collate_fn=collate_fn)

    test_loader = DataLoader(test_dataset, 
                             batch_size=val_batch_size, 
                             shuffle=True, 
                             pin_memory=pin_memory,
                             collate_fn=collate_fn)
    

    return train_loader, test_loader


