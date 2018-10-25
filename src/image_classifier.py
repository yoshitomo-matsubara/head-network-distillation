import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim

from myutils.common import file_util, yaml_util
from myutils.pytorch import func_util
from utils import module_util
from utils.dataset import general_util, cifar_util


def get_argparser():
    argparser = argparse.ArgumentParser(description='PyTorch image classifier')
    argparser.add_argument('--config', required=True, help='yaml file path')
    argparser.add_argument('-init', action='store_true', help='overwrite checkpoint')
    argparser.add_argument('-evaluate', action='store_true', help='evaluation option')
    return argparser


def resume_from_ckpt(model, model_config, init):
    ckpt_file_path = model_config['ckpt']
    if init or not file_util.check_if_exists(ckpt_file_path):
        return model_config['type'], 0, 1, ckpt_file_path

    print('Resuming from checkpoint..')
    checkpoint = torch.load(ckpt_file_path)
    model.load_state_dict(checkpoint['model'])
    model_type = checkpoint['type']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    return model_type, best_acc, start_epoch, ckpt_file_path


def get_data_loaders(config):
    dataset_config = config['dataset']
    train_config = config['train']
    test_config = config['test']
    compress_config = test_config['compression']
    dataset_name = dataset_config['name']
    if dataset_name.startswith('caltech'):
        return general_util.get_data_loaders(dataset_config['data'], train_config['batch_size'],
                                             compress_config['type'], compress_config['size'], ae_model=None,
                                             rough_size=train_config['rough_size'],
                                             reshape_size=config['input_shape'][1:3],
                                             compression_quality=test_config['jquality'])
    elif dataset_name.startswith('cifar'):
        return cifar_util.get_data_loaders(dataset_config['data'], train_config['batch_size'],
                                           compress_config['type'], compress_config['size'], train_config['valid_rate'],
                                           is_cifar100=dataset_name == 'cifar100', ae_model=None)
    raise ValueError('dataset_name `{}` is not expected'.format(dataset_name))


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
    print('\n{} set: Loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
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

    config = yaml_util.load_yaml_file(args.config)
    train_loader, valid_loader, test_loader = get_data_loaders(config)
    model = module_util.get_model(config, device)
    model_type, best_acc, start_epoch, ckpt_file_path = resume_from_ckpt(model, config['model'], args.init)
    train_config = config['train']
    criterion_config = train_config['criterion']
    criterion = func_util.get_loss(criterion_config['type'], criterion_config['params'])
    if not args.evaluate:
        optim_config = train_config['optimizer']
        optimizer = func_util.get_optimizer(model, optim_config['type'], optim_config['params'])
        interval = train_config['interval']
        for epoch in range(start_epoch, start_epoch + train_config['epoch']):
            train(model, train_loader, optimizer, criterion, epoch, device, interval)
            best_acc = validate(model, valid_loader, criterion, epoch, device, best_acc, ckpt_file_path, model_type)
    test(model, test_loader, criterion, device)


if __name__ == '__main__':
    parser = get_argparser()
    run(parser.parse_args())
