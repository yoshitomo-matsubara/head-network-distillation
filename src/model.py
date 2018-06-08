import torch.nn as nn


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, 10),
            nn.LogSoftmax(1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 16 * 5 * 5)
        return self.classifier(x)


class YOLOv2(nn.Module):
    def __init__(self):
        super(YOLOv2, self).__init__()
        self.features56 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.Conv2d(128, 64, kernel_size=1),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.features28 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3),
            nn.Conv2d(256, 128, kernel_size=1),
            nn.Conv2d(128, 256, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.features14 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3),
            nn.Conv2d(512, 256, kernel_size=1),
            nn.Conv2d(256, 512, kernel_size=3),
            nn.Conv2d(512, 256, kernel_size=1),
            nn.Conv2d(256, 512, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.features7 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3),
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.Conv2d(512, 1024, kernel_size=3),
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.Conv2d(512, 1024, kernel_size=3)
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(1024, 1000, kernel_size=1),
            nn.AvgPool2d(kernel_size=1),
            nn.Softmax(1)
        )

    def forward(self, x):
        x = self.features56(x)
        x = self.features28(x)
        x = self.features14(x)
        x = self.features7(x)
        return self.classifier(x)


class YOLOv3(nn.Module):
    # Referred https://github.com/xiaochus/YOLOv3
    def __init__(self):
        super(YOLOv3, self).__init__()
        self.conv2d_32 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2d_64 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.stacked_conv2d_64 = self.create_stacked_residual_block(64, 32, 1)
        self.conv2d_128 = nn.Conv2d(64, 128, kernel_size=3, stride=2)
        self.stacked_conv2d_128 = self.create_stacked_residual_block(128, 64, 2)
        self.conv2d_256 = nn.Conv2d(128, 256, kernel_size=3, stride=2)
        self.stacked_conv2d_256 = self.create_stacked_residual_block(256, 128, 8)
        self.conv2d_512 = nn.Conv2d(256, 512, kernel_size=3, stride=2)
        self.stacked_conv2d_512 = self.create_stacked_residual_block(512, 256, 8)
        self.conv2d_1024 = nn.Conv2d(512, 1024, kernel_size=3, stride=2)
        self.stacked_conv2d_1024 = self.create_stacked_residual_block(1024, 512, 4)
        self.classifier = nn.Sequential(
            nn.Conv2d(1024, 1000, kernel_size=1),
            nn.AvgPool2d(kernel_size=1),
            nn.Softmax(1)
        )

    def create_conv2_unit(self, input_filters, output_filters, kernel_size, strides=1):
        sequential = nn.Sequential(
            nn.Conv2d(input_filters, output_filters, kernel_size=kernel_size, stride=strides),
            nn.BatchNorm2d(num_features=output_filters),
            nn.LeakyReLU()
        )
        return sequential

    def create_residual_block(self, input_filters, output_filters):
        sequential = nn.Sequential(
            self.create_conv2_unit(input_filters, output_filters, 1),
            self.create_conv2_unit(output_filters, 2 * output_filters, 3)
        )
        return sequential

    def create_stacked_residual_block(self, input_filters, output_filters, n):
        stacked_list = list()
        for i in range(n):
            stacked_list.append(self.create_residual_block(input_filters, output_filters))
        return nn.Sequential(*stacked_list)

    def forward(self, x):
        x = self.conv2d_32(x)
        x = self.conv2d_64(x)
        x = self.stacked_conv2d_64(x)
        x = self.conv2d_128(x)
        x = self.stacked_conv2d_128(x)
        x = self.conv2d_256(x)
        x = self.stacked_conv2d_256(x)
        x = self.conv2d_512(x)
        x = self.stacked_conv2d_512(x)
        x = self.conv2d_1024(x)
        x = self.stacked_conv2d_1024(x)
        return self.classifier(x)
