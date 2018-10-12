import argparse
import os

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml

import ae_runner
from utils import file_util, module_util
from utils.dataset import caltech_util, cifar_util


def get_argparser():
    parser = argparse.ArgumentParser(description='PyTorch image classifier')
    parser.add_argument('--data', default='./resource/data/', help='data dir path')
    parser.add_argument('--dataset', default='caltech', help='dataset type')
    parser.add_argument('-cifar100', action='store_true', help='option to use CIFAR-100 instead of CIFAR-10')
    parser.add_argument('-caltech256', action='store_true', help='option to use Caltech101 instead of Caltech256')
    parser.add_argument('--config', required=True, help='yaml file path')
    parser.add_argument('--ckpt', default='./resource/ckpt/', help='checkpoint dir path')
    parser.add_argument('--bsize', type=int, default=128, help='number of samples per a batch')
    parser.add_argument('--epoch', type=int, default=100, help='number of epochs for training')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--vrate', type=float, default=0.1, help='validation rate')
    parser.add_argument('--interval', type=int, default=50, help='logging training status ')
    parser.add_argument('--jquality', type=int, default=0, help='JPEG compression quality (1 - 95)')
    parser.add_argument('--ctype', help='compression type')
    parser.add_argument('--csize', help='compression size')
    parser.add_argument('--ae', help='autoencoder yaml file path')
    parser.add_argument('-init', action='store_true', help='overwrite checkpoint')
    parser.add_argument('-evaluate', action='store_true', help='evaluation option')
    return parser


def resume_from_ckpt(model, config, args):
    ckpt_file_path = os.path.join(args.ckpt, config['experiment_name'])
    if args.init or not os.path.exists(ckpt_file_path):
        return config['model']['type'], 0, 1, ckpt_file_path

    print('Resuming from checkpoint..')
    checkpoint = torch.load(ckpt_file_path)
    model.load_state_dict(checkpoint['model'])
    model_type = checkpoint['type']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    return model_type, best_acc, start_epoch, ckpt_file_path


def load_autoencoder(ae_config_file_path, ckpt_dir_path):
    if ae_config_file_path is None or ckpt_dir_path is None:
        return None

    with open(ae_config_file_path, 'r') as fp:
        ae_config = yaml.load(fp)

    ae = module_util.get_autoencoder(False, ae_config)
    ae_runner.resume_from_ckpt(ae, ae_config, ckpt_dir_path, False)
    return ae


def get_data_loaders(args, ae, config):
    dataset_type = args.dataset
    if dataset_type == 'caltech':
        return caltech_util.get_data_loaders(args.data, args.bsize, args.ctype, args.csize, args.vrate,
                                             is_caltech256=args.caltech256, ae=ae,
                                             reshape_size=tuple(config['input_shape'][1:3]),
                                             compression_quality=args.jquality)
    elif dataset_type == 'cifar':
        return cifar_util.get_data_loaders(args.data, args.bsize, args.ctype, args.csize, args.vrate,
                                           is_cifar100=args.cifar100, ae=ae)
    raise ValueError('dataset_type `{}` is not expected'.format(dataset_type))


def get_criterion_optimizer(model, args, momentum=0.9, weight_decay=5e-4):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=momentum, weight_decay=weight_decay)
    return criterion, optimizer


def train(model, train_loader, optimizer, criterion, epoch, device, interval):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if batch_idx > 0 and batch_idx % interval == 0:
            progress = 100.0 * batch_idx / len(train_loader)
            train_accuracy = correct / total
            print('[{}/{} ({:.0f}%)]\tLoss: {:.4f}\tTraining Accuracy: {:.4f}'.format(batch_idx * len(inputs),
                                                                                      len(train_loader.sampler),
                                                                                      progress, loss.item(),
                                                                                      train_accuracy))


def save_ckpt(model, acc, epoch, ckpt_file_path, model_type):
    print('Saving..')
    state = {
        'type': model_type,
        'model': model.state_dict(),
        'acc': acc,
        'epoch': epoch,
    }
    file_util.make_parent_dirs(ckpt_file_path)
    torch.save(state, ckpt_file_path)


def test(model, test_loader, criterion, device, data_type='Test'):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100.0 * correct / total
    print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        data_type, test_loss, correct, total, acc))
    return acc


def validate(model, valid_loader, criterion, epoch, device, best_acc, ckpt_file_path, model_type):
    acc = test(model, valid_loader, criterion, device, 'Validation')
    if acc > best_acc:
        save_ckpt(model, acc, epoch, ckpt_file_path, model_type)
        best_acc = acc
    return best_acc


def run(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        cudnn.benchmark = True

    with open(args.config, 'r') as fp:
        config = yaml.load(fp)

    ae = load_autoencoder(args.ae, args.ckpt)
    train_loader, valid_loader, test_loader = get_data_loaders(args, ae, config)
    model = module_util.get_model(device, config)
    model_type, best_acc, start_epoch, ckpt_file_path = resume_from_ckpt(model, config, args)
    criterion, optimizer = get_criterion_optimizer(model, args)
    if not args.evaluate:
        for epoch in range(start_epoch, start_epoch + args.epoch):
            train(model, train_loader, optimizer, criterion, epoch, device, args.interval)
            best_acc = validate(model, valid_loader, criterion, epoch, device, best_acc, ckpt_file_path, model_type)
    test(model, test_loader, criterion, device)


if __name__ == '__main__':
    parser = get_argparser()
    run(parser.parse_args())
