import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

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
        result = self.autoencoder.encode_and_decode(data.unsqueeze(0))
        return result.squeeze(0)

    def __repr__(self):
        return self.__class__.__name__ + '()'