import argparse

import torch.utils.data

from autoencoder import *
from utils import cifar10_util


# Referred to https://github.com/wanglouis49/pytorch-autoencoders
def get_argparser():
    parser = argparse.ArgumentParser(description='VAE CIFAR-10')
    parser.add_argument('-data', default='./resource/data/', help='CIFAR-10 data dir path')
    parser.add_argument('-epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('-no_cuda', action='store_true', default=False, help='enables CUDA training')
    parser.add_argument('-seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('-hsize', type=int, default=20, metavar='N', help='how big is z')
    parser.add_argument('-vrate', type=float, default=0.1, help='validation rate')
    parser.add_argument('-inter_size', type=int, default=128, metavar='N', help='how big is linear around z')
    return parser


def train(model, train_loader, optimizer, epoch, args):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        if args.cuda:
            data = data.cuda()
        optimizer.zero_grad()
        loss = model.loss_function(data)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))


def test(model, test_loader, args):
    model.eval()
    test_loss = 0
    for i, (data, _) in enumerate(test_loader):
        if args.cuda:
            data = data.cuda()
        test_loss += model.loss_function(data).item()

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


def run(args):
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    data_dir_path = args.data

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    train_loader, valid_loader, test_loader =\
        cifar10_util.get_data_loaders(data_dir_path, args.vrate)

    model = VAE(args.inter_size, args.hsize)
    if args.cuda:
        model.cuda()

    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)
    for epoch in range(1, args.epochs + 1):
        train(model, train_loader, optimizer, epoch, args)
        test(model, test_loader, args)


if __name__ == '__main__':
    parser = get_argparser()
    run(parser.parse_args())
