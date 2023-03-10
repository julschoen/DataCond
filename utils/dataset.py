import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler


def get_train_val_loader(batch_size, valid_size=0.1, num_workers=4, pin_memory=False):
    
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

    valid_dataset = datasets.CIFAR10(
        root="../../data",
        train=True,
        download=True,
        transform=transform,
    )

    num_train = len(train_dataset)
    indices = torch.randperm(num_train)
    split = int(torch.floor(valid_size * num_train))

    train_idx, val_idx = indices[split:], indices[:split]

    torch.save(train_idx, '../../data/train_id.pt')
    torch.save(val_idx, '../../data/val_id.pt')

    train_idx= torch.load('../../data/train_id.pt')
    val_idx = torch.load('../../data/val_id.pt')

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return (train_loader, valid_loader)


def get_test_loader(batch_size,
                    shuffle=True,
                    num_workers=4,
                    pin_memory=False):

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
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader