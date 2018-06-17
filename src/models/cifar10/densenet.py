from collections import OrderedDict

import torch.nn as nn
from torchvision.models.densenet import _DenseBlock, _Transition


class DenseNet(nn.Module):
    """
    PyTorch based DenseNet implementation for CIFAR-10
    """
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000,
                 conv2d_ksize=7, conv2d_stride=2, conv2d_padding=3,
                 maxpool_2d_ksize=3, maxpool_2d_stride=2, maxpool_2d_padding=1, avg_pool2d_ksize=7):

        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=conv2d_ksize, stride=conv2d_stride,
                                padding=conv2d_padding, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=maxpool_2d_ksize, stride=maxpool_2d_stride, padding=maxpool_2d_padding)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        self.relu = nn.ReLU(inplace=True)
        self.avg_pool2d = nn.AvgPool2d(kernel_size=avg_pool2d_ksize, stride=1)

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = self.relu(features)
        out = self.avg_pool2d(out)
        out = self.classifier(out.view(features.size(0), -1))
        return out
