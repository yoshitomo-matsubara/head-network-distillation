import torch.nn as nn

from .base import BaseHeadMimic, BaseMimic, SeqWithAux


def mimic_version1b_with_aux(bottleneck_channel, aux_output_size=1000):
    modules = [
        nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        nn.Conv2d(64, bottleneck_channel, kernel_size=2, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(bottleneck_channel), nn.ReLU(inplace=True),
        nn.Conv2d(bottleneck_channel, 512, kernel_size=2, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(512), nn.ReLU(inplace=True),
        nn.Conv2d(512, 256, kernel_size=2, stride=1, bias=False),
        nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        nn.Conv2d(256, 192, kernel_size=2, stride=1, bias=False),
        nn.AvgPool2d(kernel_size=2, stride=2)
    ]
    return SeqWithAux(modules, aux_idx=2, aux_input_channel=bottleneck_channel, aux_output_size=aux_output_size)


def mimic_version1(make_bottleneck, bottleneck_channel, use_aux):
    if make_bottleneck:
        return mimic_version1b_with_aux(bottleneck_channel) if use_aux else nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, bottleneck_channel, kernel_size=2, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(bottleneck_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_channel, 256, kernel_size=2, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=2, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=2, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=2, stride=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
    return nn.Sequential(
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 128, kernel_size=2, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 64, kernel_size=2, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 32, kernel_size=2, stride=1, bias=False),
        nn.AvgPool2d(kernel_size=2, stride=1)
    )


def mimic_version2(make_bottleneck, bottleneck_channel):
    if make_bottleneck:
        return nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, bottleneck_channel, kernel_size=2, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(bottleneck_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_channel, 512, kernel_size=2, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=2, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 192, kernel_size=2, stride=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
    return nn.Sequential(
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 128, kernel_size=2, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 256, kernel_size=2, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 192, kernel_size=2, stride=1, bias=False),
        nn.AvgPool2d(kernel_size=2, stride=2)
    )


def mimic_version3(make_bottleneck, bottleneck_channel):
    if make_bottleneck:
        return nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, bottleneck_channel, kernel_size=2, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(bottleneck_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_channel, 512, kernel_size=2, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=2, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=2, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 160, kernel_size=2, stride=1, bias=False),
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
        nn.Conv2d(512, 160, kernel_size=2, stride=1, bias=False),
        nn.AvgPool2d(kernel_size=2, stride=2)
    )


def mimic_version_test1(bottleneck_channel):
    # end_idx: 24
    return nn.Sequential(
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 32, kernel_size=2, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, bottleneck_channel, kernel_size=2, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(bottleneck_channel),
        nn.ReLU(inplace=True),
        nn.Conv2d(bottleneck_channel, 64, kernel_size=2, stride=1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 32, kernel_size=2, stride=1, bias=False)
    )


class MobileNetHeadMimic(BaseHeadMimic):
    # designed for input image size [3, 224, 224]
    def __init__(self, version, bottleneck_channel=3, use_aux=False):
        super().__init__()
        self.extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        if version in ['1', '1b']:
            self.module_seq = mimic_version1(version == '1b', bottleneck_channel, use_aux)
        elif version in ['2', '2b']:
            self.module_seq = mimic_version2(version == '2b', bottleneck_channel)
        elif version in ['3', '3b']:
            self.module_seq = mimic_version3(version == '3b', bottleneck_channel)
        elif version == 'test1':
            self.module_seq = mimic_version_test1(bottleneck_channel)
        else:
            raise ValueError('version `{}` is not expected'.format(version))
        self.initialize_weights()

    def forward(self, sample_batch):
        zs = self.extractor(sample_batch)
        return self.module_seq(zs)


class MobileNetMimic(BaseMimic):
    def __init__(self, student_model, tail_modules):
        super().__init__(student_model, tail_modules)

    def forward(self, sample_batch):
        aux = None
        zs = sample_batch
        if self.student_model is not None:
            zs = self.student_model(zs)
            if isinstance(zs, tuple):
                zs, aux = zs[0], zs[1]

        zs = self.features(zs)
        zs = self.classifier(zs.view(zs.size(0), -1))
        return zs if aux is None else (zs, aux)
