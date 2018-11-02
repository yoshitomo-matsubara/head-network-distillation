import torch.nn as nn
from torchvision.models.resnet import BasicBlock, Bottleneck


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=101, num_init_features=64,
                 first_conv2d_ksize=7, first_conv2d_stride=2, first_conv2d_padding=3,
                 last_avgpool2d_ksize=7, last_avgpool2d_stride=1):
        self.inplanes = num_init_features
        super().__init__()
        self.conv1 = nn.Conv2d(3, num_init_features, kernel_size=first_conv2d_ksize,
                               stride=first_conv2d_stride, padding=first_conv2d_padding, bias=False)
        self.bn1 = nn.BatchNorm2d(num_init_features)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, num_init_features, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(last_avgpool2d_ksize, stride=last_avgpool2d_stride)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def resnet18(param_config):
    return ResNet(BasicBlock, [2, 2, 2, 2], **param_config)


def resnet34(param_config):
    return ResNet(BasicBlock, [3, 4, 6, 3], **param_config)


def resnet50(param_config):
    return ResNet(Bottleneck, [3, 4, 6, 3], **param_config)


def resnet101(param_config):
    return ResNet(Bottleneck, [3, 4, 23, 3], **param_config)


def resnet152(param_config):
    return ResNet(Bottleneck, [3, 8, 36, 3], **param_config)


def resnet_model(model_type, param_config):
    if model_type == 'resnet18':
        return resnet18(param_config)
    elif model_type == 'resnet34':
        return resnet34(param_config)
    elif model_type == 'resnet50':
        return resnet50(param_config)
    elif model_type == 'resnet101':
        return resnet101(param_config)
    elif model_type == 'resnet152':
        return resnet152(param_config)
    return None
