import torch.nn as nn


class VggMimic(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, zs):
        return zs
