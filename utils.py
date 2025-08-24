import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
from typing import Tuple, Dict
import random
import numpy as np
from pathlib import Path

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_transforms(train: bool = True):
    if train:
        return T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
    else:
        return T.Compose([
            T.ToTensor(),
            T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])

def get_dataloaders(batch_size: int = 128, workers: int = 2) -> Tuple[DataLoader, DataLoader]:
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=get_transforms(True))
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=get_transforms(False))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    return train_loader, test_loader

def build_optimizer(model: torch.nn.Module, opt: str = 'adam', lr: float = 1e-3, weight_decay: float = 5e-4):
    if opt.lower() == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    else:
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()

def save_checkpoint(state: Dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)

def load_checkpoint(path: str, map_location=None) -> Dict:
    return torch.load(path, map_location=map_location)

def clamp_to_image(x: torch.Tensor) -> torch.Tensor:
    # clamp normalized tensor to valid normalized range by inverse-normalizing to [0,1], clamping, then renormalizing
    mean = torch.tensor(CIFAR10_MEAN, device=x.device).view(1,3,1,1)
    std = torch.tensor(CIFAR10_STD, device=x.device).view(1,3,1,1)
    x_pixel = x * std + mean
    x_pixel = torch.clamp(x_pixel, 0.0, 1.0)
    x_norm = (x_pixel - mean) / std
    return x_norm

def eps_pixel_to_normalized(eps_pixel: float, device) -> torch.Tensor:
    """Convert epsilon defined in pixel space ([0,1]) to normalized space per-channel."""
    std = torch.tensor(CIFAR10_STD, device=device).view(1,3,1,1)
    return torch.tensor(eps_pixel, device=device).view(1,1,1,1) / std
