import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml

import ae_runner
from utils import caltech_util, file_util, misc_util, module_util, module_wrap_util


def get_argparser():
    parser = argparse.ArgumentParser(description='Compression Analyzer')
    parser.add_argument('--data', default='./resource/data/', help='Caltech data dir path')
    parser.add_argument('-caltech256', action='store_true', help='option to use Caltech101 instead of Caltech256')
    parser.add_argument('--config', required=True, help='yaml file path')
    parser.add_argument('--ckpt', default='./resource/ckpt/', help='checkpoint dir path')
    parser.add_argument('--bsize', type=int, default=100, help='number of samples per a batch')
    parser.add_argument('--epoch', type=int, default=100, help='number of epochs for training')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--vrate', type=float, default=0.1, help='validation rate')
    parser.add_argument('--interval', type=int, default=50, help='logging training status ')
    parser.add_argument('--ctype', help='compression type')
    parser.add_argument('--csize', help='compression size')
    parser.add_argument('--ae', help='autoencoder yaml file path')
    parser.add_argument('-init', action='store_true', help='overwrite checkpoint')
    parser.add_argument('-evaluate', action='store_true', help='evaluation option')
    parser.add_argument('--mode', default='comp_rate', help='evaluation option')
    parser.add_argument('--comp_layer', type=int, default=-1, help='index of layer to compress its input (starts from 1)')
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
            print('[{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(batch_idx * len(inputs), len(train_loader.sampler),
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


def test(model, test_loader, device, data_type='Test'):
    model.eval()
    correct = 0
    total = 0
    bandwidth = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            bandwidth += inputs.clone().cpu().detach().numpy().nbytes
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100.0 * correct / total
    print('\n{} set: Accuracy: {}/{} ({:.0f}%)\n'.format(data_type, correct, total, acc))
    return acc, bandwidth / total


def validate(model, valid_loader, criterion, epoch, device, best_acc, ckpt_file_path, model_type):
    acc, _ = test(model, valid_loader, criterion, device, 'Validation')
    if acc > best_acc:
        save_ckpt(model, acc, epoch, ckpt_file_path, model_type)
        best_acc = acc
    return best_acc


def extract_compression_rates(parent_module, org_bandwidth_list, compressed_bandwidth_list, name_list):
    for name, child_module in parent_module.named_children():
        if list(child_module.children()) and not isinstance(child_module, module_wrap_util.CompressionWrapper):
            extract_compression_rates(child_module, org_bandwidth_list, compressed_bandwidth_list, name_list)
        else:
            org_bandwidth_list.append(child_module.get_average_org_bandwidth())
            compressed_bandwidth_list.append(child_module.get_average_compressed_bandwidth())
            name_list.append(type(child_module.org_module).__name__)


def plot_compression_rates(model, avg_input_bandwidth):
    org_bandwidth_list = list()
    compressed_bandwidth_list = list()
    name_list = list()
    extract_compression_rates(model, org_bandwidth_list, compressed_bandwidth_list, name_list)
    xs = list(range(len(org_bandwidth_list)))
    if not misc_util.check_if_plottable():
        print('Layer\tOriginal Bandwidth\tCompressed Bandwidth')
        for i in range(len(xs)):
            print('{}\t{}\t{}'.format(name_list[i], org_bandwidth_list[i], compressed_bandwidth_list[i]))
        return

    plt.plot(xs, [avg_input_bandwidth for _ in range(len(name_list))], label='Input')
    plt.plot(xs, org_bandwidth_list, label='Original')
    plt.plot(xs, compressed_bandwidth_list, label='Compressed')
    plt.xticks(xs, name_list, rotation=90)
    plt.xlabel('Layer')
    plt.ylabel('Average Bandwidth [Bytes]')
    plt.legend()
    plt.show()


def analyze_compression_rate(model, test_loader, device):
    module_wrap_util.wrap_all_child_modules(model, module_wrap_util.CompressionWrapper)
    _, avg_input_bandwidth = test(model, test_loader, device)
    plot_compression_rates(model, avg_input_bandwidth)


def extract_running_times(wrapped_modules):
    num_samples = len(wrapped_modules[0].get_timestamps())
    start_times = np.array([wrapped_module.start_timestamp for wrapped_module in wrapped_modules])
    time_mat = np.zeros((num_samples, len(wrapped_modules)))
    comp_time_mat = np.zeros_like(time_mat)
    for i, wrapped_module in enumerate(wrapped_modules):
        target_times = np.array(wrapped_module.get_compression_timestamps() if wrapped_module.is_compressed
                                else wrapped_module.get_timestamps())
        time_mat[:, i] = (target_times - start_times).reshape(1, start_times.size)
        if wrapped_module.is_compressed:
            comp_time_mat[:, i] = np.array(wrapped_module.get_compression_time_list()).reshape(1, start_times.size)
    return time_mat, comp_time_mat


def plot_running_time(wrapped_modules):
    name_list = ['{}{}: {}'.format(type(wrapped_module.org_module).__name__,
                                   '*' if wrapped_module.is_compressed else '', i + 1)
                 for i, wrapped_module in enumerate(wrapped_modules)]
    time_mat, comp_time_mat = extract_running_times(wrapped_modules)
    mean_times = time_mat.mean(axis=0)
    mean_comp_times = comp_time_mat.mean(axis=0)
    xs = list(range(len(name_list)))
    if not misc_util.check_if_plottable():
        print('Layer\tAverage Accumulated Elapsed Time\tAverage Elapsed Time for Compression')
        for i in range(len(xs)):
            print('{}\t{}\t{}'.format(name_list[i], mean_times[i], mean_comp_times[i]))
        return

    fig, ax1 = plt.subplots()
    ax1.plot(xs, mean_comp_times, '-')
    ax1.set_xticks(xs)
    ax1.set_xticklabels(name_list)
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Average Elapsed Time for Compression [sec]', color='b')
    for tick in ax1.get_xticklabels():
        tick.set_rotation(90)

    ax2 = ax1.twinx()
    ax2.plot(xs, mean_times, 'r--')
    ax2.set_ylabel('Average Accumulated Elapsed Time [sec]', color='r')
    plt.tight_layout()
    plt.show()


def analyze_running_time(model, comp_layer_idx, test_loader, device):
    wrapped_modules = list()
    module_wrap_util.wrap_all_child_modules(model, module_wrap_util.RunTimeWrapper, wrapped_list=wrapped_modules)
    wrapped_modules[0].is_first = True
    if comp_layer_idx < 1:
        for wrapped_module in wrapped_modules:
            wrapped_module.is_compressed = True
    elif comp_layer_idx <= len(wrapped_modules):
        wrapped_modules[comp_layer_idx - 1].is_compressed = True

    _, avg_input_bandwidth = test(model, test_loader, device)
    plot_running_time(wrapped_modules)


def run(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        cudnn.benchmark = True

    with open(args.config, 'r') as fp:
        config = yaml.load(fp)

    ae = load_autoencoder(args.ae, args.ckpt)
    train_loader, valid_loader, test_loader =\
        caltech_util.get_data_loaders(args.data, args.bsize, args.ctype, args.csize, args.vrate,
                                      is_caltech256=args.caltech256, ae=ae, reshape_size=tuple(config['input_shape'][1:3]))
    model = module_util.get_model(device, config)
    model_type, best_acc, start_epoch, ckpt_file_path = resume_from_ckpt(model, config, args)
    criterion, optimizer = get_criterion_optimizer(model, args)
    if not args.evaluate:
        for epoch in range(start_epoch, start_epoch + args.epoch):
            train(model, train_loader, optimizer, criterion, epoch, device, args.interval)
            best_acc = validate(model, valid_loader, criterion, epoch, device, best_acc, ckpt_file_path, model_type)

    analysis_mode = args.mode
    if analysis_mode == 'comp_rate':
        analyze_compression_rate(model, test_loader, device)
    elif analysis_mode == 'run_time':
        analyze_running_time(model, args.comp_layer, test_loader, device)
    else:
        raise ValueError('mode argument `{}` is not expected'.format(analysis_mode))


if __name__ == '__main__':
    parser = get_argparser()
    run(parser.parse_args())
