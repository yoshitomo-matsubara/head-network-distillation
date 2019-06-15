import torch.nn as nn
from .base import BaseAutoencoder


class InputAutoencoder(BaseAutoencoder):
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
