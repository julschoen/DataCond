import argparse
import torch
from torchvision import datasets, transforms
import os
sys.path.append("../") 

from utils.dataset import get_train_loader


comp_dir = '../../comparison_synth'

def save(file_name, data):
        file_name = os.path.join(comp_dir, file_name)
        torch.save(data.cpu(), file_name)

def make_random():
    train_loader = get_train_loader(5000)

    if not os.path.isdir(comp_dir):
        os.mkdir(comp_dir)

    data_all = []
    label_all = []

    for i, (x,y) in enumerate(train_loader):
        for c in range(10):
            data = x[y == c]
            perm = torch.randperm(data.shape[0])[:100]
            data, label = data[perm], torch.ones(100)*c

            data_all.append(data)
            label_all.append(label)

        data = torch.concat(data_all)
        label = torch.concat(label_all)
        save(f'rand_x_{i}.pt', data)
        save(f'rand_y_{i}.pt', label)

        if i == 10:
            break

if __name__ == '__main__':
    make_random()
