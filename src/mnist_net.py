import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from utils import net_measurer


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.first_conv_size = 40
        self.second_conv_size = 80
        self.last_mp_kernel_size = 2
        self.feature_size = ((self.last_mp_kernel_size ** 2) ** 2) * self.second_conv_size
        self.features = nn.Sequential(
            nn.Conv2d(1, self.first_conv_size, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(self.first_conv_size, self.second_conv_size, kernel_size=5),
            nn.MaxPool2d(kernel_size=self.last_mp_kernel_size)
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_size, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(100, 50),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(50, 10),
            nn.LogSoftmax(1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.feature_size)
        return self.classifier(x)


def get_argparser():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('-data', default='./data/', help='MNIST data dir path')
    parser.add_argument('-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('-test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('-epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('-lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('-momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('-no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('-seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('-log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    return parser


def get_data_loaders(use_cuda, args):
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(args.data, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(args.data, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
    return train_loader, test_loader


def train(model, train_loader, epoch, device, args):
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def run(args):
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if use_cuda else 'cpu')
    model = MyNet().to(device)
    train_loader, test_loader = get_data_loaders(use_cuda, args)
    input_shape = train_loader.dataset[0][0].size()
    net_measurer.calc_model_flops_and_size(model, list(input_shape))
    for epoch in range(1, args.epochs + 1):
        train(model, train_loader, epoch, device, args)
        test(model, test_loader, device)


if __name__ == '__main__':
    argparser = get_argparser()
    run(argparser.parse_args())
