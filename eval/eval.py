import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.utils.data as data_utils

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

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


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
    parser.add_argument('--cifar', type=bool, default=True)
    parser.add_argument('--log_dir', type=str, default='./log')
    args = parser.parse_args()

    device = args.device

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5), 
            (0.5, 0.5, 0.5))
    ])
    if args.cifar:
        dataset1 = datasets.CIFAR10('./', train=True, download=True,
                           transform=transform)
        dataset2 = datasets.CIFAR10('./', train=False,
                           transform=transform)
    else:
        dataset1 = datasets.MNIST('./', train=True, download=True,
                           transform=transform)
        dataset2 = datasets.MNIST('./', train=False,
                           transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, shuffle=True, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    if args.train_full:
        model = ConvNet(args).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

        model = ResNet18().to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

        model = SimpleNet(in_dim=32).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

    chkpt = os.path.join(args.log_dir, 'checkpoints')
    targets = torch.load(os.path.join(chkpt,'labels.pt'))
    features = torch.load(os.path.join(chkpt, 'data.pt'))
    synth = data_utils.TensorDataset(features, targets)
    train_loader = torch.utils.data.DataLoader(synth, batch_size=args.batch_size, shuffle=True)

    model = ConvNet(args).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, 200):
        train(args, model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)

    model = ResNet18(args).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, 200):
        train(args, model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)

    model = SimpleNet(in_dim=32).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, 200):
        train(args, model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
