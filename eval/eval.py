import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.utils.data as data_utils

import sys;
sys.path.append("../") 

from utils.classifiers import ConvNet, ResNet18, SimpleNet
from utils.dataset import get_train_loader, get_val_loader, get_test_loader
from utils.eval_utils import EarlyStopper, train, val, test


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=500)
    parser.add_argument('--test-batch-size', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--save-model', type=bool, default=False)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--train_full', type=bool, default=False)
    parser.add_argument('--log_dir', type=str, default='./log')
    args = parser.parse_args()

    device = args.device

    train_loader = get_train_loader(args.batch_size)
    val_loader = get_val_loader(args.test_batch_size)
    test_loader = get_test_loader(args.test_batch_size)

    if args.train_full:
        model = ConvNet(args).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        es = EarlyStopper()
        for epoch in range(1, args.epochs + 1):
            tl = train(args, model, device, train_loader, optimizer, epoch)
            vl = val(model, device, val_loader)
            es(tl, vl)
            if es.early_stop:
                break
        test(model, device, test_loader, verbose=True)

        model = ResNet18().to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        es = EarlyStopper()
        for epoch in range(1, args.epochs + 1):
            tl = train(args, model, device, train_loader, optimizer, epoch)
            vl = val(model, device, val_loader)
            es(tl, vl)
            if es.early_stop:
                break
        test(model, device, test_loader, verbose=True)

        model = SimpleNet(in_dim=32*32*3).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        es = EarlyStopper()
        for epoch in range(1, args.epochs + 1):
            tl = train(args, model, device, train_loader, optimizer, epoch)
            vl = val(model, device, val_loader)
            es(tl, vl)
            if es.early_stop:
                break
        test(model, device, test_loader, verbose=True)

    chkpt = os.path.join(args.log_dir, 'checkpoints')
    targets = torch.load(os.path.join(chkpt,'labels.pt'))
    features = torch.load(os.path.join(chkpt, 'data.pt'))
    synth = data_utils.TensorDataset(features, targets)
    train_loader = torch.utils.data.DataLoader(synth, batch_size=args.batch_size, shuffle=True)

    model = ConvNet(args).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    es = EarlyStopper()
    for epoch in range(1, 200):
        tl = train(args, model, device, train_loader, optimizer, epoch)
        vl = val(model, device, val_loader)
        es(tl, vl)
        if es.early_stop:
            print(epoch)
            break
    test(model, device, test_loader, verbose=True)

    model = ResNet18().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    es = EarlyStopper()
    for epoch in range(1, 200):
        tl = train(args, model, device, train_loader, optimizer, epoch)
        vl = val(model, device, val_loader)
        es(tl, vl)
        if es.early_stop:
            print(epoch)
            break
    test(model, device, test_loader, verbose=True)

    model = SimpleNet(in_dim=32*32*3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    es = EarlyStopper()
    for epoch in range(1, 200):
        tl = train(args, model, device, train_loader, optimizer, epoch)
        vl = val(model, device, val_loader)
        es(tl, vl)
        if es.early_stop:
            print(epoch)
            break
    test(model, device, test_loader, verbose=True)

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
