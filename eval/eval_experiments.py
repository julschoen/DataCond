import argparse
import os
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np

import sys;
sys.path.append("../") 

from utils.classifiers import ConvNet, ResNet18, SimpleNet
from utils.dataset import get_val_loader, get_test_loader
from utils.eval_utils import EarlyStopper, train, val, test


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Evaluate Full DC Runs')
    parser.add_argument('--batch-size', type=int, default=500)
    parser.add_argument('--test-batch-size', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_ims', type=int, default=10)
    parser.add_argument('--method', type=str, default='DM/Base')
    args = parser.parse_args()

    device = args.device

    val_loader = get_val_loader(args.test_batch_size)
    test_loader = get_test_loader(args.test_batch_size)

    conv = []
    res = []
    simple = []
    for i in range(10):
        d = args.method + str(i)        
        ## random
        targets = torch.load(os.path.join(d, 'checkpoints', 'labels.pt'))
        features = torch.load(os.path.join(d, 'checkpoints', 'data.pt'))

        synth = torch.utils.data.TensorDataset(features, targets)
        train_loader = torch.utils.data.DataLoader(synth, batch_size=args.batch_size, shuffle=True)
        for _ in range(10):
            model = ConvNet(args).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            es = EarlyStopper()
            for epoch in range(1, 200):
                tl = train(args, model, device, train_loader, optimizer, epoch)
                vl = val(model, device, val_loader)
                es(tl, vl)
                if es.early_stop:
                    break
            acc = test(model, device, test_loader)
            conv.append(acc)

            model = ResNet18().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            es = EarlyStopper()
            for epoch in range(1, 200):
                tl = train(args, model, device, train_loader, optimizer, epoch)
                vl = val(model, device, val_loader)
                es(tl, vl)
                if es.early_stop:
                    break
            acc = test(model, device, test_loader)
            res.append(acc)

            model = SimpleNet(in_dim=32*32*3).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            es = EarlyStopper()
            for epoch in range(1, 200):
                tl = train(args, model, device, train_loader, optimizer, epoch)
                vl = val(model, device, val_loader)
                es(tl, vl)
                if es.early_stop:
                    break
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
