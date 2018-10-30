import torch.nn as nn

from .base import BaseHeadMimic, BaseMimic


def mimic_version1():
    return nn.Sequential(
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 128, kernel_size=2, stride=1, padding=1, bias=False),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
    )


def mimic_version2():
    return nn.Sequential(
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 128, kernel_size=2, stride=2, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 256, kernel_size=1, stride=1, bias=False),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
    )


def mimic_version3():
    return nn.Sequential(
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 128, kernel_size=2, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 256, kernel_size=2, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 512, kernel_size=2, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(512),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 640, kernel_size=2, stride=1, bias=False),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
    )


class DenseNet169HeadMimic(BaseHeadMimic):
    # designed for input image size [3, 224, 224]
    def __init__(self, version):
        super().__init__()
        self.extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        if version == 1:
            self.module_seq = mimic_version1()
        elif version == 2:
            self.module_seq = mimic_version2()
        elif version == 3:
            self.module_seq = mimic_version3()
        else:
            raise ValueError('version `{}` is not expected'.format(version))
        self.initialize_weights()

    def forward(self, sample_batch):
        zs = self.extractor(sample_batch)
        return self.module_seq(zs)


class DenseNetMimic(BaseMimic):
    def __init__(self, modules):
        super().__init__(modules)

    def forward(self, sample_batch):
        return super().forward(sample_batch)
