import torch
import torch.nn as nn
from .base import BaseAutoencoder


class InputAutoencoder(BaseAutoencoder):
    def __init__(self, input_channel=3, bottleneck_channel=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channel, 32, kernel_size=5), nn.BatchNorm2d(32),
            nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 16, kernel_size=5), nn.BatchNorm2d(16),
            nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 8, kernel_size=4), nn.BatchNorm2d(8),
            nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(8, bottleneck_channel, kernel_size=2), nn.BatchNorm2d(bottleneck_channel), nn.ReLU(inplace=True)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(bottleneck_channel, 6, kernel_size=5, stride=2), nn.BatchNorm2d(6), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(6, 12, kernel_size=5, stride=2), nn.BatchNorm2d(12), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(12, 18, kernel_size=5, stride=2), nn.BatchNorm2d(18), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(18, 24, kernel_size=4), nn.BatchNorm2d(24), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(24, 18, kernel_size=4), nn.BatchNorm2d(18), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(18, 12, kernel_size=3), nn.BatchNorm2d(12), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(12, 6, kernel_size=3), nn.BatchNorm2d(6), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(6, input_channel, kernel_size=2), nn.Sigmoid()
        )
        self.initialize_weights()

    def forward(self, sample_batch):
        # Encoding
        zs = self.encoder(sample_batch)

        # Decoding
        zs = self.decoder(zs)
        return zs


# Referred to https://github.com/sksq96/pytorch-vae/blob/master/vae-cnn.ipynb
class Flatter(nn.Module):
    def forward(self, zs):
        return zs.view(zs.size(0), -1)


class UnFlatter(nn.Module):
    def __init__(self, h_dim):
        super().__init__()
        self.h_dim = h_dim

    def forward(self, zs):
        return zs.view(zs.size(0), self.h_dim, 1, 1)


class Bottleneck(nn.Module):
    def __init__(self, h_dim, z_dim, is_static):
        super().__init__()
        self.flatter = Flatter()
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)
        self.unflatter = UnFlatter(h_dim)
        self.is_static = is_static

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).to(mu.device) if not self.is_static else 1
        z = mu + std * esp
        return z

    def forward(self, hs):
        hs = self.flatter(hs)
        mu, logvar = self.fc1(hs), self.fc2(hs)
        zs = self.reparameterize(mu, logvar)
        zs = self.fc3(zs)
        zs = self.unflatter(zs)
        return (zs, mu, logvar) if self.training else zs


class InputVAE(BaseAutoencoder):
    def __init__(self, input_channel=3, h_dim=18432, z_dim=512, is_static=False):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channel, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2),
            nn.ReLU()
        )
        self.bottleneck = Bottleneck(h_dim, z_dim, is_static)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(h_dim, 512, kernel_size=4, stride=3),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=3),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_channel, kernel_size=2, stride=2),
            nn.Sigmoid()
        )
        self.initialize_weights()

    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

    def loss_function(self, outputs, sample_batch, mu, logvar):
        bce = nn.functional.mse_loss(outputs, sample_batch, reduction='sum')
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return bce + kld

    def forward(self, sample_batch):
        hs = self.encoder(sample_batch)
        if self.training:
            zs, mu, logvar = self.bottleneck(hs)
            zs = self.decoder(zs)
            return zs, self.loss_function(zs, sample_batch, mu, logvar)

        zs = self.bottleneck(hs)
        zs = self.decoder(zs)
        return zs
