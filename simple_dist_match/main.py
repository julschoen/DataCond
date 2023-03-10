import argparse
import torch
from torchvision import datasets, transforms
from trainer import Trainer
import numpy as np
from sklearn.decomposition import IncrementalPCA
import matplotlib.pyplot as plt
import os
import sys;
sys.path.append("../") 

from utils.dataset import get_train_loader

def main():
    parser = argparse.ArgumentParser(description='DC-VAE')
    ### General
    parser.add_argument('--batch_size', type=int, default= 512, help='input batch size for training (default: 512)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_dir', type=str, default='./log')
    parser.add_argument('--load_ae', type=bool, default=False)
    parser.add_argument('--load_ims', type=bool, default=False)
    parser.add_argument('--pca', type=bool, default=True)

    ### VAE
    parser.add_argument('--niter_ae', type=int, default=3000)
    parser.add_argument('--lrAE', type=float, default=1e-3)
    parser.add_argument('--z_dim', type=int, default=32)
    parser.add_argument('--filter', type=int, default=64)

    ### Synth Images
    parser.add_argument('--num_ims', type=int, default=10)
    parser.add_argument('--niter_ims', type=int, default=10000)
    parser.add_argument('--init_ims', type=bool,default=False)
    parser.add_argument('--lrIms', type=float, default=1e-2)
    parser.add_argument('--corr', type=bool, default=False)
    parser.add_argument('--corr_coef', type=float, default=0.1)

    args = parser.parse_args()

    train_loader = get_train_loader(args.batch_size)

    trainer = Trainer(args, train_loader)
    trainer.train()

    transformer = IncrementalPCA(n_components=2, batch_size=200)
    with torch.no_grad():
        for i, (x,y) in enumerate(train_loader):
            _, z = trainer.ae(x.to(args.device), y.to(args.device))
            transformer.partial_fit(z.squeeze().detach().cpu().numpy())

    zs = []
    ys = []
    with torch.no_grad():
        for i, (x,y) in enumerate(train_loader):
            _, z = trainer.ae(x.to(args.device), y.to(args.device))

            z = transformer.transform(z.squeeze().detach().cpu().numpy())

            zs.append(z)
            ys.append(y.detach().cpu())

    _, z_ims = trainer.ae(torch.tanh(trainer.ims).to(args.device), trainer.labels.to(args.device))

    z_ims = transformer.transform(z_ims.squeeze().detach().cpu().numpy())
    zs.append(z_ims)
    ys.append(trainer.labels.detach().cpu())

    zs = np.concatenate(zs)
    ys = np.concatenate(ys)

    print(transformer.explained_variance_ratio_)

    X, Y = zs[:-100], ys[:-100]
    x, y = zs[-100:], ys[-100:]

    fig = plt.figure(figsize=(16,14))
    plt.scatter(X[:,0], X[:,1], c=Y, cmap='Set1')
    plt.scatter(x[:,0], x[:,1], cmap='black')
    plt.colorbar()
    plt.savefig(os.path.join(args.log_dir, 'images','all_vae.pdf'), bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure(figsize=(40,30))
    for ind in range(10):
        plt.subplot(3,4,ind+1)
        plt.scatter(X[ind == Y,0], X[ind == Y,1], cmap='b', label='real')
        plt.scatter(x[ind == y,0], x[ind == y,1], cmap='r', label='embedded')
        plt.title(f'Class {ind}')
        plt.legend()
    plt.savefig(os.path.join(args.log_dir, 'images','classes_vae.pdf'), bbox_inches='tight')
    plt.close(fig)
    

if __name__ == '__main__':
    main()