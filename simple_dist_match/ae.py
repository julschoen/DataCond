import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    # initializers
    def __init__(self, params, z_dim, nfilter=128):
        super(Encoder, self).__init__()

        self.z_dim = z_dim

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.norm1 = nn.GroupNorm(16, 16, affine=True)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.norm2 = nn.GroupNorm(32, 32, affine=True)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.norm3 = nn.GroupNorm(64, 64, affine=True)
        self.conv4 = nn.Conv2d(64, z_dim, kernel_size=3, stride=2, padding=1)


        self.fc1 = nn.Linear(10, z_dim)

    def forward(self, input, label):
        x = self.norm1(self.conv1(input))
        x = F.relu(x)

        x = self.norm2(self.conv2(x))
        x = F.relu(x)

        x = self.norm3(self.conv3(x))
        x = F.relu(x)
        x = self.conv4(x)

        x = x.view(-1, self.z_dim)
        label = F.one_hot(label, num_classes=10).float()
        y = self.fc1(label)

        print(x.shape, y.shape)

        x = torch.cat((x, y), dim=1)

        return x

class Decoder(nn.Module):
    # initializers
    def __init__(self, z_dim, nfilter=128):
        super(Decoder, self).__init__()
        self.z_dim = z_dim
        self.conv1 = nn.ConvTranspose2d(z_dim, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1)

        self.norm1 = nn.GroupNorm(64, 64, affine=True)
        self.norm2 = nn.GroupNorm(32, 32, affine=True)
        self.norm3 = nn.GroupNorm(16, 16, affine=True)

    # forward method
    def forward(self, input):
        input = input.reshape(-1,self.z_dim, 1, 1)
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
