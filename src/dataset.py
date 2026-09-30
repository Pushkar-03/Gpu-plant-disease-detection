import torch
from pathlib import Path
try:
    from torch.utils.data import DataLoader, random_split, Subset
    from torchvision import datasets, transforms
except ImportError:
    pass
from src import config
from src.device import get_device

def get_transforms(image_size=config.IMAGE_SIZE):
    """Returns torchvision transforms for training and validation."""
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.NORM_MEAN, std=config.NORM_STD)
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((image_size[0] + 32, image_size[1] + 32)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.NORM_MEAN, std=config.NORM_STD)
    ])
    
    return train_transforms, val_transforms

def get_datasets(data_dir, image_size, validation_split, seed):
    """Splits an ImageFolder dataset into deterministic training and validation subsets."""
    train_transform, val_transform = get_transforms(image_size)
    
    # Load dataset twice to apply different transforms to train and val splits safely
    full_dataset_train = datasets.ImageFolder(data_dir, transform=train_transform)
    full_dataset_val = datasets.ImageFolder(data_dir, transform=val_transform)
    
    num_total = len(full_dataset_train)
    num_val = int(validation_split * num_total)
    num_train = num_total - num_val
    
    # Deterministic split
    generator = torch.Generator().manual_seed(seed)
    train_indices, val_indices = random_split(
        range(num_total), [num_train, num_val], generator=generator
    )
    
    train_dataset = Subset(full_dataset_train, train_indices)
    val_dataset = Subset(full_dataset_val, val_indices)
    
    return train_dataset, val_dataset, full_dataset_train.classes

def get_dataloaders(data_dir=config.RAW_DATA_DIR, image_size=config.IMAGE_SIZE, 
                    batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, 
                    validation_split=config.VALIDATION_SPLIT, seed=config.RANDOM_SEED):
    """Creates configured DataLoaders, handling CUDA pin_memory appropriately."""
    train_dataset, val_dataset, classes = get_datasets(data_dir, image_size, validation_split, seed)
    
    device = get_device()
    pin_memory = True if (hasattr(device, 'type') and device.type == 'cuda') or device == 'cuda' else False
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=pin_memory
    )
    
    return train_loader, val_loader, classes
