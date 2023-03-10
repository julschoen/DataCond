import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

def get_train_loader(batch_size):
    
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5), 
            (0.5, 0.5, 0.5))
    ])

    # load the dataset
    train_dataset = datasets.CIFAR10(
        root="../../data",
        train=True,
        download=True,
        transform=transform,
    )

    
    train_idx= torch.load('../../data/train_id.pt')
    train_sampler = SubsetRandomSampler(train_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler
    )


    return train_loader

def get_val_loader(batch_size):
    
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5), 
            (0.5, 0.5, 0.5))
    ])

    
    valid_dataset = datasets.CIFAR10(
        root="../../data",
        train=True,
        download=True,
        transform=transform,
    )
    val_idx = torch.load('../../data/val_id.pt')
    valid_sampler = SubsetRandomSampler(val_idx)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler
    )

    return valid_loader

def get_test_loader(batch_size, shuffle=True):
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5), 
            (0.5, 0.5, 0.5))
    ])

    dataset = datasets.CIFAR10(
        root="../../data", train=False,
        download=True, transform=transform,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle
    )

    return data_loader
