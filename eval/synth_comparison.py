import argparse
import os
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np

import sys;
sys.path.append("../") 

from utils.classifiers import ConvNet, ResNet18, SimpleNet

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    return 100. * correct / len(test_loader.dataset)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=500)
    parser.add_argument('--test-batch-size', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_ims', type=int, default=10)
    parser.add_argument('--method', type=str, default='herding')
    args = parser.parse_args()

    device = args.device

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5), 
            (0.5, 0.5, 0.5))
    ])

    test_kwargs = {'batch_size': args.test_batch_size}
    
    dataset2 = datasets.CIFAR10('../../data', train=False,
                       transform=transform)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    conv = []
    res = []
    simple = []
    for i in range(10):
        comp_dir = '../../comparison_synth'
        
        ## random
        targets = torch.load(os.path.join(comp_dir, str(args.num_ims),args.method+f'_y_{i}.pt'))
        features = torch.load(os.path.join(comp_dir, str(args.num_ims), args.method+f'_x_{i}.pt'))

        synth = torch.utils.data.TensorDataset(features, targets)
        train_loader = torch.utils.data.DataLoader(synth, batch_size=args.batch_size, shuffle=True)
        for _ in range(10):
            model = ConvNet(args).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

            for epoch in range(1, 200):
                train(args, model, device, train_loader, optimizer, epoch)
            acc = test(model, device, test_loader)
            conv.append(acc)

            model = ResNet18().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

            for epoch in range(1, 200):
                train(args, model, device, train_loader, optimizer, epoch)
            acc = test(model, device, test_loader)
            res.append(acc)

            model = SimpleNet(in_dim=32*32*3).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

            for epoch in range(1, 200):
                train(args, model, device, train_loader, optimizer, epoch)
            acc = test(model, device, test_loader)
            simple.append(acc)

    conv = np.array(conv)
    res = np.array(res)
    simple = np.array(simple)
    print('Conv {:.2f}\\pm{:.2f}'.format(conv.mean(), conv.std()))
    print('Res {:.2f}\\pm{:.2f}'.format(res.mean(), res.std()))
    print('Simple {:.2f}\\pm{:.2f}'.format(simple.mean(), simple.std()))



if __name__ == '__main__':
    main()