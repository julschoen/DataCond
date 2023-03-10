import argparse
import torch
from torchvision import datasets, transforms
import os
from dataset import get_train_val_loader


log_dir = '../../data'

def save(file_name, data):
        file_name = os.path.join(log_dir, file_name)
        torch.save(data.cpu(), file_name)

def make_data():
    train_loader, _ = get_train_val_loader(5000)

    for c in range(10):
        print(f'#### Class {c} ####')
        data_all = []
        for i, (x,y) in enumerate(train_loader):
            data = x[y == c]
            data_all.append(data)

        data = torch.concat(data_all)
        save(f'data_class_{c}.pt', data)

if __name__ == '__main__':
    make_data()
