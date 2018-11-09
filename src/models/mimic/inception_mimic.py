import torch.nn as nn

from .base import BaseHeadMimic, BaseMimic


def mimic_version1(make_bottleneck=False):
    if make_bottleneck:
        return nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=2, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
            nn.Conv2d(3, 64, kernel_size=2, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=2, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 192, kernel_size=2, stride=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
    return nn.Sequential(
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 128, kernel_size=2, stride=1, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 192, kernel_size=2, stride=1, bias=False),
        nn.AvgPool2d(kernel_size=2, stride=2)
    )


def mimic_version2(make_bottleneck=False):
    if make_bottleneck:
        return nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=2, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
            nn.Conv2d(3, 64, kernel_size=2, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=2, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=2, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=2, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 768, kernel_size=2, stride=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
    return nn.Sequential(
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 128, kernel_size=2, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 256, kernel_size=2, stride=1, bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 512, kernel_size=2, stride=1, bias=False),
        nn.BatchNorm2d(512),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 768, kernel_size=2, stride=1, bias=False),
        nn.AvgPool2d(kernel_size=2, stride=2)
    )


def mimic_version3(make_bottleneck=False):
    if make_bottleneck:
        return nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=2, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
            nn.Conv2d(3, 64, kernel_size=2, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=2, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=2, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=2, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 768, kernel_size=2, stride=1, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
            nn.Conv2d(768, 1024, kernel_size=2, stride=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1280, kernel_size=2, stride=1, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU(inplace=True),
            nn.Conv2d(1280, 1536, kernel_size=2, stride=1, bias=False),
            nn.BatchNorm2d(1536),
            nn.ReLU(inplace=True),
            nn.Conv2d(1536, 1280, kernel_size=2, stride=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
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
        nn.Conv2d(512, 768, kernel_size=2, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(768),
        nn.ReLU(inplace=True),
        nn.Conv2d(768, 1024, kernel_size=2, stride=1, bias=False),
        nn.BatchNorm2d(1024),
        nn.ReLU(inplace=True),
        nn.Conv2d(1024, 1280, kernel_size=2, stride=1, bias=False),
        nn.BatchNorm2d(1280),
        nn.ReLU(inplace=True),
        nn.Conv2d(1280, 1536, kernel_size=2, stride=1, bias=False),
        nn.BatchNorm2d(1536),
        nn.ReLU(inplace=True),
        nn.Conv2d(1536, 1280, kernel_size=2, stride=1, bias=False),
        nn.AvgPool2d(kernel_size=2, stride=2)
    )


class InceptionHeadMimic(BaseHeadMimic):
    # designed for input image size [3, 299, 299]
    def __init__(self, version):
        super().__init__()
        self.extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        if version in ['1', '1b']:
            self.module_seq = mimic_version1(version == '1b')
        elif version in ['2', '2b']:
            self.module_seq = mimic_version2(version == '2b')
        elif version in ['3', '3b']:
            self.module_seq = mimic_version3(version == '3b')
        else:
            raise ValueError('version `{}` is not expected'.format(version))
        self.initialize_weights()

    def forward(self, sample_batch):
        zs = self.extractor(sample_batch)
        return self.module_seq(zs)


class InceptionMimic(BaseMimic):
    def __init__(self, modules):
        super().__init__(modules)

    def forward(self, sample_batch):
        return super().forward(sample_batch)
