import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Tuple

def create_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    return transform_train, transform_test

def get_dataloaders(data_dir: str = "./data", batch_size: int = 256, num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
    transform_train, transform_test = create_transforms()
    train_dataset = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform_train)
    test_dataset = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader