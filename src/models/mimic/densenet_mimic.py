import torch.nn as nn


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
        nn.Conv2d(64, 128, kernel_size=2, stride=2, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 256, kernel_size=2, stride=2, bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 640, kernel_size=2, stride=1, padding=1, bias=False),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
    )


class DenseNet169HeadMimic(nn.Module):
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
        zs = self.extractor(sample_batch)
        return self.module_seq(zs)


class DenseNet169Mimic(nn.Module):
    def __init__(self, modules):
        super().__init__()
        self.features = nn.Sequential(*modules[:-1])
        self.classifier = modules[-1]

    def forward(self, sample_batch):
        features = self.features(sample_batch)
        return self.classifier(features.view(features.size(0), -1))
