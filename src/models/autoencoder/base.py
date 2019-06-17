import torch.nn as nn


class BaseAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, sample_batch):
        raise NotImplementedError


class BaseExtendedModel(nn.Module):
    def __init__(self, head_modules, autoencoder, tail_modules):
        super().__init__()
        self.head_model = nn.Sequential(*head_modules)
        self.autoencoder = autoencoder
        self.tail_model = nn.Sequential(*tail_modules[:-1])
        self.linear = tail_modules[-1]

    def forward(self, sample_batch):
        zs = self.head_model(sample_batch)
        zs = self.autoencoder(zs)
        zs = self.tail_model(zs)
        return self.linear(zs.view(zs.size(0), -1))
