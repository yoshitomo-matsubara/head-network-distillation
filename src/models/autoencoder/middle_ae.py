from torch import nn

from models.autoencoder.base import BaseAutoencoder


class MiddleAutoencoder(BaseAutoencoder):
    def __init__(self, input_channel=256, bottleneck_channel=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channel, 128, kernel_size=2, stride=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=2, stride=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=2, stride=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, bottleneck_channel, kernel_size=2, stride=1), nn.BatchNorm2d(bottleneck_channel)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(bottleneck_channel, 32, kernel_size=2, stride=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 64, kernel_size=2, stride=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 128, kernel_size=2, stride=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, input_channel, kernel_size=2, stride=1)
        )
        self.initialize_weights()

    def forward(self, sample_batch):
        # Encoding
        zs = self.encoder(sample_batch)

        # Decoding
        zs = self.decoder(zs)
        return zs
