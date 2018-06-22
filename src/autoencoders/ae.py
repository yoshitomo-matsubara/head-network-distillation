import torch.nn as nn


class Autoencoder(nn.Module):
    def forward(self, x):
        raise NotImplementedError

    def encode_and_decode(self, x):
        raise NotImplementedError

    def loss_function(self, x):
        raise NotImplementedError


class AETransformer(object):
    def __init__(self, autoencoder):
        self.autoencoder = autoencoder

    def __call__(self, data):
        return self.autoencoder.encode_and_decode(data)

    def __repr__(self):
        return self.__class__.__name__ + '()'
