import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn

import ae_runner
from myutils.common import file_util, yaml_util
from myutils.pytorch import func_util
from structure.wrapper import *
from utils import misc_util, module_util, module_wrap_util
from utils.dataset import general_util


def get_argparser():
    parser = argparse.ArgumentParser(description='Compression Analyzer')
    parser.add_argument('--config', required=True, help='yaml file path')
    parser.add_argument('--pkl', help='model pickle file path')
    parser.add_argument('-init', action='store_true', help='overwrite checkpoint')
    parser.add_argument('-evaluate', action='store_true', help='evaluation option')
    parser.add_argument('--mode', default='comp_rate', help='evaluation option')
    parser.add_argument('--comp_layer', type=int, default=-1, help='index of layer to compress its input'
                                                                   ' (starts from 1, no compression if 0 is given)')
    return parser


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


def load_autoencoder(ae_config_file_path, device):
    if not file_util.check_if_exists(ae_config_file_path):
        return None

    ae_config = yaml_util.load_yaml_file(ae_config_file_path)
    ae_model = module_util.get_autoencoder(ae_config, device)
    ae_runner.resume_from_ckpt(ae_model, ae_config, False)
    return ae_model


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
    compressed_bandwidth = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            np_input = inputs.clone().cpu().detach().numpy()
            bandwidth += np_input.nbytes
            compressed_input = zlib.compress(np_input, 9)
            compressed_bandwidth += sys.getsizeof(compressed_input)
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100.0 * correct / total
    print('\n{} set: Accuracy: {}/{} ({:.0f}%)\n'.format(data_type, correct, total, acc))
    return acc, bandwidth / total, compressed_bandwidth / total


def validate(model, valid_loader, epoch, device, best_acc, ckpt_file_path, model_type):
    acc, _, _ = test(model, valid_loader, device, 'Validation')
    if acc > best_acc:
        save_ckpt(model, acc, epoch, ckpt_file_path, model_type)
        best_acc = acc
    return best_acc


def extract_compression_rates(parent_module, org_bandwidth_list, compressed_bandwidth_list, name_list):
    for name, child_module in parent_module.named_children():
        if isinstance(child_module, CompressionWrapper):
            org_bandwidth_list.append(child_module.get_average_org_bandwidth())
            compressed_bandwidth_list.append(child_module.get_average_compressed_bandwidth())
            name_list.append(type(child_module.org_module).__name__)
        elif list(child_module.children()):
            extract_compression_rates(child_module, org_bandwidth_list, compressed_bandwidth_list, name_list)
        else:
            print('CompressionWrapper is missing for {}: {}'.format(name, type(child_module).__name__))


def plot_compression_rates(model, avg_input_bandwidth, avg_compressed_input_bandwidth):
    org_bandwidth_list = list()
    compressed_bandwidth_list = list()
    name_list = list()
    org_bandwidth_list.append(avg_input_bandwidth)
    compressed_bandwidth_list.append(avg_compressed_input_bandwidth)
    name_list.append('Input')
    extract_compression_rates(model, org_bandwidth_list, compressed_bandwidth_list, name_list)
    xs = list(range(len(org_bandwidth_list)))
    if not misc_util.check_if_plottable():
        print('Average Input Bandwidth: {}\tCompressed: {}'.format(avg_input_bandwidth,avg_compressed_input_bandwidth))
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


def analyze_compression_rate(model, input_shape, test_loader, device):
    input_batch = torch.rand(input_shape).unsqueeze(0).to(device)
    module_wrap_util.wrap_decomposable_modules(model, CompressionWrapper, input_batch)
    _, avg_input_bandwidth, avg_compressed_input_bandwidth = test(model, test_loader, device)
    plot_compression_rates(model, avg_input_bandwidth, avg_compressed_input_bandwidth)


def extract_running_times(wrapped_modules):
    num_samples = len(wrapped_modules[0].get_timestamps())
    start_times = np.array(wrapped_modules[0].start_timestamp_list)
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


def analyze_running_time(model, input_shape, comp_layer_idx, test_loader, device):
    wrapped_modules = list()
    input_batch = torch.rand(input_shape).unsqueeze(0).to(device)
    module_wrap_util.wrap_decomposable_modules(model, RunTimeWrapper, input_batch,
                                               wrapped_list=wrapped_modules)
    wrapped_modules[0].is_first = True
    if comp_layer_idx < 0:
        for wrapped_module in wrapped_modules:
            wrapped_module.is_compressed = True
    elif 0 < comp_layer_idx <= len(wrapped_modules):
        wrapped_modules[comp_layer_idx - 1].is_compressed = True

    _, avg_input_bandwidth, _ = test(model, test_loader, device)
    plot_running_time(wrapped_modules)


def run(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        cudnn.benchmark = True

    config = yaml_util.load_yaml_file(args.config)
    dataset_config = config['dataset']
    train_config = config['train']
    test_config = config['test']
    compress_config = test_config['compression']
    ae_model = load_autoencoder(test_config['autoencoder'], device)
    input_shape = config['input_shape']
    train_loader, valid_loader, test_loader =\
        general_util.get_data_loaders(dataset_config['data'], train_config['batch_size'],
                                      compress_config['type'], compress_config['size'], ae_model=ae_model,
                                      reshape_size=input_shape[1:3], compression_quality=test_config['jquality'])

    pickle_file_path = args.pkl
    if not file_util.check_if_exists(pickle_file_path):
        model = module_util.get_model(config, device)
        model_type, best_acc, start_epoch, ckpt_file_path = resume_from_ckpt(model, config['model'], args.init)
        if not args.evaluate:
            criterion_config = train_config['criterion']
            criterion = func_util.get_loss(criterion_config['type'], criterion_config['params'])
            optim_config = train_config['optimizer']
            optimizer = func_util.get_optimizer(model, optim_config['type'], optim_config['params'])
            interval = train_config['interval']
            for epoch in range(start_epoch, start_epoch + train_config['epoch']):
                train(model, train_loader, optimizer, criterion, epoch, device, interval)
                best_acc = validate(model, valid_loader, epoch, device, best_acc, ckpt_file_path, model_type)
    else:
        model = file_util.load_pickle(pickle_file_path).to(device)

    analysis_mode = args.mode
    if analysis_mode == 'comp_rate':
        analyze_compression_rate(model, input_shape, test_loader, device)
    elif analysis_mode == 'run_time':
        analyze_running_time(model, input_shape, args.comp_layer, test_loader, device)
    else:
        raise ValueError('mode argument `{}` is not expected'.format(analysis_mode))


if __name__ == '__main__':
    parser = get_argparser()
    run(parser.parse_args())
