import torch
import torchvision.utils as vutils
from torchvision import datasets, transforms

import os
import random
import argparse

from utils.classifiers import ConvNet

class Trainer():
    def __init__(self, params, train_loader):
        self.p = params

        self.train_loader = train_loader
        self.gen = self.inf_train_gen()

        self.ims = torch.randn(10*self.p.num_ims,3,32,32).to(self.p.device)

        if self.p.init_ims:
            self.init_ims()

        self.ims = torch.nn.Parameter(self.ims)
        self.labels = torch.arange(10, device=self.p.device).repeat(self.p.num_ims,1).T.flatten()
        self.opt_ims = torch.optim.Adam([self.ims], lr=self.p.lr)

        self.models = []
        for _ in range(self.p.num_models):
        	m = ConvNet(cl=False)
            self.moedels.append(m)
        
        ### Make Log Dirs
        if not os.path.isdir(self.p.log_dir):
            os.mkdir(self.p.log_dir)

        path = os.path.join(self.p.log_dir, 'images')
        if not os.path.isdir(path):
            os.mkdir(path)

        path = os.path.join(self.p.log_dir, 'checkpoints')
        if not os.path.isdir(path):
            os.mkdir(path)

    def inf_train_gen(self):
        while True:
            for data in self.train_loader:
                yield data

    def init_ims(self):
        for c in range(10):
            X = torch.load(os.path.join('../data/', f'data_class_{c}.pt'))
            perm = torch.randperm(X.shape[0])[:self.p.num_ims]
            xc = X[perm]
            self.ims[c*self.p.num_ims:(c+1)*self.p.num_ims] = xc


    def log_interpolation(self, step):
        path = os.path.join(self.p.log_dir, 'images/synth')
        if not os.path.isdir(path):
            os.mkdir(path)
        ims = torch.tanh(self.ims)
        vutils.save_image(
            vutils.make_grid(ims, nrow=self.p.num_ims, padding=2, normalize=True)
            , os.path.join(path, f'{step}.png'))


    def save(self):
        path = os.path.join(self.p.log_dir, 'checkpoints')
        if not os.path.isdir(path):
            os.mkdir(path)
        file_name = os.path.join(path, 'data.pt')
        ims = torch.tanh(self.ims)
        torch.save(ims.cpu(), file_name)

        file_name = os.path.join(path, 'labels.pt')
        torch.save(self.labels.cpu(), file_name)

    def load_ims(self):
        path = os.path.join(self.p.log_dir, 'checkpoints', 'data.pt')
        if os.path.exists(path):
            self.ims = torch.load(path)
            self.ims = torch.nn.Parameter(self.ims)
        return os.path.exists(path)

    
    def total_variation_loss(self, img, weight=1, four=True):
        bs_img, c_img, h_img, w_img = img.size()

        tv_h = torch.pow(img[:,:,1:,:]-img[:,:,:-1,:], 2).sum()
        tv_w = torch.pow(img[:,:,:,1:]-img[:,:,:,:-1], 2).sum()

        tv = weight*(tv_h+tv_w)/(bs_img*c_img*h_img*w_img)

        if four:
            tv_h4 = torch.pow(img[:,:,:-1,:]-img[:,:,1:,:], 2).sum()
            tv_w4 = torch.pow(img[:,:,:,:-1]-img[:,:,:,1:], 2).sum()
            tv = tv + weight*(tv_h4+tv_w4)/(bs_img*c_img*h_img*w_img)
            tv = tv/2

        return tv

    def sample_model(self):
    	m = self.models[random.randint(0,self.p.num_models-1)]
    	return m.eval().to(self.p.device)

    def train_ims_cw(self):
        print('############## Training Images ##############',flush=True)
        self.ims.requires_grad = True

        for t in range(self.p.niter):
            loss = torch.tensor(0.0).to(self.p.device)
            data, labels = next(self.gen)
            for c in range(10):
                d_c = data[labels == c].to(self.p.device)
                ims = self.ims[c*self.p.num_ims:(c+1)*self.p.num_ims]
                model = self.sample_model()

                encX = model(d_c).detach()
                encY = model(ims)

                mmd = torch.norm(encX.mean(dim=0)-encY.mean(dim=0))

                if self.p.corr:
                    corr = self.total_variation_loss(torch.tanh(ims))
                else:
                    corr = torch.zeros(1)

                loss = loss + mmd

                if self.p.corr:
                    loss = loss + self.p.corr_coef*corr

            self.opt_ims.zero_grad()
            loss.backward()
            self.opt_ims.step()
        
            if (t%100) == 0:
                s = '[{}|{}] Loss: {:.4f}, MMD: {:.4f}'.format(t, self.p.niter, loss.item(), mmd.item())
                if self.p.corr:
                    s += ', Corr: {:.4f}'.format(corr.item())
                print(s,flush=True)
                self.log_interpolation(t)

        self.save()
        self.ims.requires_grad = False


    def train(self):
        self.ims.requires_grad = False
        self.train_ims_cw()

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Simple Dist Match')
    # General Training
    parser.add_argument('--batch-size', type=int, default= 256)
    parser.add_argument('--niter', type=int, default=20000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_ims', type=int, default=10)
    parser.add_argument('--num_models', type=int, default=20)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_dir', type=str, default='./log')
    parser.add_argument('--corr', type=bool, default=False)
    parser.add_argument('--corr_coef', type=float, default=1)
    parser.add_argument('--init_ims', type=bool, default=False)

    args = parser.parse_args()

    train_kwargs = {'batch_size': args.batch_size, 'shuffle':True}

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5), 
            (0.5, 0.5, 0.5))
        ])
    dataset1 = datasets.CIFAR10('../data/', train=True, download=True,
                        transform=transform)
    

    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)

    trainer = Trainer(args, train_loader)
    trainer.train()
    

if __name__ == '__main__':
    main()