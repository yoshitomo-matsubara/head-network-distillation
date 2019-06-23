import torch.nn as nn
from .base import BaseAutoencoder


class MiddleAutoencoder(BaseAutoencoder):
    def __init__(self, input_channel=3, bottleneck_channel=3):
        super().__init__()
        self.sub_encoder1 = nn.Sequential(nn.Conv2d(input_channel, 6, kernel_size=5), nn.ReLU(inplace=True))
        self.maxpool2d1 = nn.MaxPool2d(kernel_size=2, return_indices=True)
        self.sub_encoder2 = nn.Sequential(nn.Conv2d(6, 12, kernel_size=5), nn.ReLU(inplace=True))
        self.maxpool2d2 = nn.MaxPool2d(kernel_size=2, return_indices=True)
        self.sub_encoder3 = nn.Sequential(nn.Conv2d(12, 6, kernel_size=4), nn.ReLU(inplace=True))
        self.maxpool2d3 = nn.MaxPool2d(kernel_size=2, return_indices=True)
        self.sub_encoder4 = nn.Sequential(nn.Conv2d(6, bottleneck_channel, kernel_size=2), nn.ReLU(inplace=True))

        self.sub_decoder4 = nn.ConvTranspose2d(bottleneck_channel, 6, kernel_size=2)
        self.maxunpool2d3 = nn.MaxUnpool2d(kernel_size=2)
        self.sub_decoder3 = nn.ConvTranspose2d(6, 12, kernel_size=4)
        self.maxunpool2d2 = nn.MaxUnpool2d(kernel_size=2)
        self.sub_decoder2 = nn.ConvTranspose2d(12, 6, kernel_size=5)
        self.maxunpool2d1 = nn.MaxUnpool2d(kernel_size=2)
        self.sub_decoder1 = nn.ConvTranspose2d(6, input_channel, kernel_size=5)
        self.initialize_weights()

    def forward(self, sample_batch):
        # Encoding
        zs = self.sub_encoder1(sample_batch)
        zs, indices1 = self.maxpool2d1(zs)
        zs = self.sub_encoder2(zs)
        zs, indices2 = self.maxpool2d2(zs)
        zs = self.sub_encoder3(zs)
        zs, indices3 = self.maxpool2d3(zs)
        zs = self.sub_encoder4(zs)

        # Decoding
        zs = self.sub_decoder4(zs)
        zs = self.maxunpool2d3(zs, indices3)
        zs = self.sub_decoder3(zs)
        zs = self.maxunpool2d2(zs, indices2)
        zs = self.sub_decoder2(zs)
        zs = self.maxunpool2d1(zs, indices1)
        zs = self.sub_decoder1(zs)
        return zs


class Autoencoder4DenseNet(nn.Module):
    def __init__(self, input_channel=128, bottleneck_channel=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channel, 128, kernel_size=2), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 64, kernel_size=2), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 32, kernel_size=2), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, bottleneck_channel, kernel_size=2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(bottleneck_channel, 32, kernel_size=4, stride=3), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 64, kernel_size=4, stride=3), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 128, kernel_size=4, stride=3), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, input_channel, kernel_size=2), nn.Sigmoid()
        )

    def forward(self, sample_batch):
        # Encoding
        zs = self.encoder(sample_batch)

        # Decoding
        zs = self.decoder(zs)
        return zs
