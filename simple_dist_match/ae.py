import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    # initializers
    def __init__(self, params, z_dim, nfilter=128):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(3, nfilter//4, kernel_size=3, stride=2, padding=1)
        self.norm1 = nn.GroupNorm(nfilter//4, nfilter//4, affine=True)
        self.conv2 = nn.Conv2d(nfilter//4, nfilter//2, kernel_size=3, stride=2, padding=1)
        self.norm2 = nn.GroupNorm(nfilter//2, nfilter//2, affine=True)
        self.conv3 = nn.Conv2d(nfilter//2, nfilter, kernel_size=3, stride=2, padding=1)
        self.norm3 = nn.GroupNorm(nfilter, nfilter, affine=True)
        self.conv4 = nn.Conv2d(nfilter, z_dim, kernel_size=3, stride=2, padding=0)

        self.fc1 = nn.Linear(10, z_dim)

    def forward(self, input, label):
        x = self.norm1(self.conv1(input))
        x = F.relu(x)

        x = self.norm2(self.conv2(x))
        x = F.relu(x)

        x = self.norm3(self.conv3(x))
        x = F.relu(x)
        x = self.conv4(x)

        label = F.one_hot(label, num_classes=10).float()
        y = self.fc1(label)

        x = torch.cat((x.squeeze(), y), dim=1)

        return x

class Decoder(nn.Module):
    # initializers
    def __init__(self, z_dim, nfilter=128):
        super(Decoder, self).__init__()
        z_dim = 2*z_dim
        self.conv1 = nn.ConvTranspose2d(self.z_dim, nfilter, kernel_size=4, stride=2, padding=0)
        self.conv2 = nn.ConvTranspose2d(nfilter, nfilter//2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.ConvTranspose2d(nfilter//2, nfilter//4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.ConvTranspose2d(nfilter//4, 3, kernel_size=4, stride=2, padding=1)

        self.norm1 = nn.GroupNorm(nfilter, nfilter, affine=True)
        self.norm2 = nn.GroupNorm(nfilter//2, nfilter//2, affine=True)
        self.norm3 = nn.GroupNorm(nfilter//4, nfilter//4, affine=True)

    # forward method
    def forward(self, input):
        input = input.reshape(-1,input.shape[0], 1, 1)
        x = self.norm1(self.conv1(input))
        x = F.relu(x)

        x = self.norm2(self.conv2(x))
        x = F.relu(x)

        x = self.norm3(self.conv3(x))
        x = F.relu(x)
        return torch.tanh(self.conv4(x))

class AE(nn.Module):
    def __init__(self, params):
        super(AE, self).__init__()
        self.encoder = Encoder(params, z_dim=params.z_dim, nfilter=params.filter)
        self.decoder = Decoder(z_dim=params.z_dim, nfilter=params.filter)

    def forward(self, x, y):
        z = self.encoder(x, y)
        recon_img = self.decoder(z)
        return recon_img, z
