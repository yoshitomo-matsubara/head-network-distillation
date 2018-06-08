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
