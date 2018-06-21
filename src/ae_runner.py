import argparse
import os

import torch.utils.data
import yaml

from autoencoder import *
from utils import cifar10_util, file_util


# Referred to https://github.com/wanglouis49/pytorch-autoencoders
def get_argparser():
    parser = argparse.ArgumentParser(description='VAE CIFAR-10')
    parser.add_argument('-data', default='./resource/data/', help='CIFAR-10 data dir path')
    parser.add_argument('-config', required=True, help='yaml file path')
    parser.add_argument('-ckpt', default='./resource/ckpt/', help='checkpoint dir path')
    parser.add_argument('-epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('-lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('-no_cuda', action='store_true', default=False, help='enables CUDA training')
    parser.add_argument('-seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('-vrate', type=float, default=0.1, help='validation rate')
    parser.add_argument('-init', action='store_true', help='overwrite checkpoint')
    return parser


def get_autoencoder(cuda_available, config):
    ae_config = config['autoencoder']
    ae_type = ae_config['type']
    if ae_type == 'vae':
        model = VAE(**ae_config['params'])
    else:
        model = None

    if cuda_available:
        model = torch.nn.DataParallel(model)
        model.cuda()
    return model


def resume_from_ckpt(ae, config, args):
    ckpt_file_path = os.path.join(args.ckpt, config['experiment_name'])
    if args.init or not os.path.exists(ckpt_file_path):
        return config['autoencoder']['type'], 0, 1, ckpt_file_path

    print('Resuming from checkpoint..')
    checkpoint = torch.load(ckpt_file_path)
    ae.load_state_dict(checkpoint['autoencoder'])
    ae_type = checkpoint['type']
    best_loss = checkpoint['loss']
    start_epoch = checkpoint['epoch']
    return ae_type, best_loss, start_epoch, ckpt_file_path


def train(model, train_loader, optimizer, epoch, cuda_available, args):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        if cuda_available:
            data = data.cuda()

        optimizer.zero_grad()
        loss = model.loss_function(data)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.sampler),
                100.0 * batch_idx / len(train_loader), loss / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.sampler)))


def save_ckpt(ae, loss, epoch, ckpt_file_path, ae_type):
    print('Saving..')
    state = {
        'type': ae_type,
        'model': ae.state_dict(),
        'loss': loss,
        'epoch': epoch,
    }
    file_util.make_parent_dirs(ckpt_file_path)
    torch.save(state, ckpt_file_path)


def test(model, test_loader, cuda_available, data_type='Test'):
    model.eval()
    test_loss = 0
    for i, (data, _) in enumerate(test_loader):
        if cuda_available:
            data = data.cuda()
        test_loss += model.loss_function(data).item()

    test_loss /= len(test_loader.sampler)
    print('{} set loss: {:.4f}'.format(data_type, test_loss))
    return test_loss


def validate(ae, valid_loader, cuda_available, epoch, best_loss, ckpt_file_path, ae_type):
    loss = test(ae, valid_loader, cuda_available, data_type='Validation')
    if loss < best_loss:
        save_ckpt(ae, loss, epoch, ckpt_file_path, ae_type)
        best_loss = loss
    return best_loss


def run(args):
    cuda_available = not args.no_cuda and torch.cuda.is_available()
    data_dir_path = args.data
    with open(args.config, 'r') as fp:
        config = yaml.load(fp)

    torch.manual_seed(args.seed)
    if cuda_available:
        torch.cuda.manual_seed(args.seed)

    train_loader, valid_loader, test_loader =\
        cifar10_util.get_data_loaders(data_dir_path, args.vrate)
    ae = get_autoencoder(cuda_available, config)
    ae_type, best_loss, start_epoch, ckpt_file_path = resume_from_ckpt(ae, config, args)
    optimizer = torch.optim.RMSprop(ae.parameters(), lr=args.lr)
    for epoch in range(start_epoch, start_epoch + args.epochs):
        train(ae, train_loader, optimizer, epoch, args)
        best_loss = validate(ae, valid_loader, cuda_available, epoch, best_loss, ckpt_file_path, ae_type)
    test(ae, test_loader, cuda_available)


if __name__ == '__main__':
    parser = get_argparser()
    run(parser.parse_args())
