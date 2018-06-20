import argparse
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import yaml
from torch.utils.data.sampler import SubsetRandomSampler

from models.cifar10 import *
from utils import file_util


# Referred to https://github.com/kuangliu/pytorch-cifar
def get_argparser():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10')
    parser.add_argument('-data', default='./resource/data/', help='CIFAR-10 data dir path')
    parser.add_argument('-config', required=True, help='yaml file path')
    parser.add_argument('-ckpt', default='./resource/ckpt/', help='checkpoint dir path')
    parser.add_argument('-epoch', type=int, default=100, help='model id')
    parser.add_argument('-lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('-val', type=float, default=0.1, help='validation rate')
    parser.add_argument('-interval', type=int, default=50, help='logging training status ')
    parser.add_argument('-ctype', help='compression type')
    parser.add_argument('-csize', help='compression size')
    parser.add_argument('-init', action='store_true', help='overwrite checkpoint')
    parser.add_argument('-evaluate', action='store_true', help='evaluation option')
    return parser


def get_train_and_valid_loaders(data_dir_path, batch_size, normalizer, random_seed=1, valid_rate=0.1, shuffle=True):
    valid_transformer = transforms.Compose([transforms.ToTensor(), normalizer])
    train_transformer = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalizer
    ])

    train_dataset = torchvision.datasets.CIFAR10(root=data_dir_path, train=True,
                                                 download=True, transform=train_transformer)
    valid_dataset = torchvision.datasets.CIFAR10(root=data_dir_path, train=True,
                                                 download=True, transform=valid_transformer)
    org_train_size = len(train_dataset)
    indices = list(range(org_train_size))
    train_end_idx = int(np.floor((1 - valid_rate) * org_train_size))
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_indices, valid_indices = indices[:train_end_idx], indices[train_end_idx:]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(valid_indices)
    pin_memory = torch.cuda.is_available()
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                                               num_workers=2, pin_memory=pin_memory)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, sampler=valid_sampler,
                                               num_workers=2, pin_memory=pin_memory)
    return train_loader, valid_loader


def get_test_transformer(normalizer, compression_type, compressed_size_str, org_size=(32, 32)):
    normal_transformer = transforms.Compose([transforms.ToTensor(), normalizer])
    if compression_type is None or compressed_size_str is None:
        return normal_transformer

    hw = compressed_size_str.split(',')
    compressed_size = (int(hw[0]), int(hw[1]))
    if compression_type == 'base':
        return transforms.Compose([
            transforms.Resize(compressed_size),
            transforms.Resize(org_size),
            transforms.ToTensor(),
            normalizer
        ])
    return normal_transformer


def get_data_loaders(data_dir_path, compression_type, compressed_size_str):
    normalizer = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    train_loader, valid_loader = get_train_and_valid_loaders(data_dir_path, batch_size=128, normalizer=normalizer)
    test_transformer = get_test_transformer(normalizer, compression_type, compressed_size_str)
    test_set = torchvision.datasets.CIFAR10(root=data_dir_path, train=False, download=True, transform=test_transformer)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=2)
    return train_loader, valid_loader, test_loader


def get_model(device, config):
    model_config = config['model']
    model_type = model_config['type']
    if model_type == 'alexnet':
        model = AlexNet(**model_config['params'])
    elif model_type == 'densenet':
        model = DenseNet(**model_config['params'])
    elif model_type == 'lenet5':
        model = LeNet5(**model_config['params'])
    elif model_type.startswith('resnet'):
        model = resnet_model(model_type, model_config['params'])
    else:
        model = None
    model = model.to(device)
    if device == 'cuda':
        model = torch.nn.DataParallel(model)
    return model


def resume_from_ckpt(model, config, args):
    ckpt_file_path = os.path.join(args.ckpt, config['experiment_name'])
    if args.init or not os.path.exists(ckpt_file_path):
        return config['model']['type'], 0, 1, ckpt_file_path

    print('Resuming from checkpoint..')
    assert os.path.isdir(args.ckpt), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(ckpt_file_path)
    model.load_state_dict(checkpoint['model'])
    model_type = checkpoint['type']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    return model_type, best_acc, start_epoch, ckpt_file_path


def get_criterion_optimizer(model, args):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
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
            print('[{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(batch_idx * len(inputs), len(train_loader.dataset),
                                                           100.0 * batch_idx / len(train_loader), loss.item()))


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
        data_type, test_loss, correct, len(test_loader.dataset), acc))
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

    train_loader, valid_loader, test_loader = get_data_loaders(args.data, args.ctype, args.csize)
    model = get_model(device, config)
    model_type, best_acc, start_epoch, ckpt_file_path = resume_from_ckpt(model, config, args)
    criterion, optimizer = get_criterion_optimizer(model, args)
    if not args.evaluate:
        for epoch in range(start_epoch, start_epoch + args.epoch):
            train(model, train_loader, optimizer, criterion, epoch, device, args.interval)
            best_acc = validate(model, valid_loader, criterion, epoch, device, best_acc, ckpt_file_path, model_type)
    test(model, test_loader, criterion, start_epoch, device, best_acc, ckpt_file_path, model_type)


if __name__ == '__main__':
    parser = get_argparser()
    run(parser.parse_args())
