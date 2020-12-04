import argparse
import time

import numpy as np
import torch
from torch import nn
from torch.backends import cudnn
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel

from model_distiller import load_ckpt
from myutils.common import file_util, yaml_util
from myutils.pytorch import tensor_util
from utils import mimic_util, module_util, dataset_util


def get_argparser():
    argparser = argparse.ArgumentParser(description='PyTorch deployment helper')
    argparser.add_argument('--config', required=True, help='yaml file path')
    argparser.add_argument('--partition', type=int, default=-1, help='partition index (starts from 0)')
    argparser.add_argument('--head', help='output file path for head network pickle')
    argparser.add_argument('--tail', help='output file path for tail network pickle')
    argparser.add_argument('--model', help='output file path for original network pickle')
    argparser.add_argument('--device', help='device for original network pickle')
    argparser.add_argument('--spbit', help='casting or quantization at splitting point: '
                                           '`8bits`, `16bits` or None (32 bits)')
    argparser.add_argument('-org', action='store_true', help='option to split an original DNN model')
    argparser.add_argument('-mimic', action='store_true', help='option to split a mimic DNN model')
    argparser.add_argument('-scpu', action='store_true', help='option to make sensor-side model runnable without cuda')
    argparser.add_argument('-ecpu', action='store_true', help='option to make edge-server model runnable without cuda')
    argparser.add_argument('-test', action='store_true', help='option to check if performance changes after splitting')
    return argparser


def predict(preds, targets):
    loss = nn.functional.cross_entropy(preds, targets)
    _, pred_labels = preds.max(1)
    correct_count = pred_labels.eq(targets).sum().item()
    return correct_count, loss.item()


def test_split_model(model, head_network, tail_network, sensor_device, edge_device, spbit, config):
    dataset_config = config['dataset']
    _, _, test_loader =\
        dataset_util.get_data_loaders(dataset_config, batch_size=config['train']['batch_size'],
                                      rough_size=config['train']['rough_size'],
                                      reshape_size=tuple(config['input_shape'][1:3]),
                                      test_batch_size=config['test']['batch_size'], jpeg_quality=-1)
    print('Testing..')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        cudnn.benchmark = True

    head_network = module_util.use_multiple_gpus_if_available(head_network, sensor_device)
    tail_network = module_util.use_multiple_gpus_if_available(tail_network, edge_device)
    model.to(device)
    head_network.to(sensor_device)
    tail_network.to(edge_device)
    head_network.eval()
    tail_network.eval()
    model.eval()
    split_correct_count = 0
    split_test_loss = 0
    org_correct_count = 0
    org_test_loss = 0
    total = 0
    file_size_list = list()
    head_proc_time_list = list()
    tail_proc_time_list = list()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            total += targets.size(0)
            inputs, targets = inputs.to(sensor_device), targets.to(edge_device)
            head_start_time = time.time()
            zs = head_network(inputs)
            if spbit in ['8bits', '16bits']:
                if spbit == '8bits':
                    # Quantization and dequantization
                    qzs = tensor_util.quantize_tensor(zs)
                    head_end_time = time.time()
                    file_size_list.append(file_util.get_binary_object_size(qzs))
                    zs = tensor_util.dequantize_tensor(qzs)
                else:
                    # Casting and recasting
                    zs = zs.half()
                    head_end_time = time.time()
                    file_size_list.append(file_util.get_binary_object_size(zs))
                    zs = zs.float()
            else:
                head_end_time = time.time()
                file_size_list.append(file_util.get_binary_object_size(zs))

            preds = tail_network(zs.to(edge_device))
            tail_end_time = time.time()
            sub_correct_count, sub_test_loss = predict(preds, targets)
            split_correct_count += sub_correct_count
            split_test_loss += sub_test_loss
            inputs, targets = inputs.to(device), targets.to(device)
            preds = model(inputs)
            sub_correct_count, sub_test_loss = predict(preds, targets)
            org_correct_count += sub_correct_count
            org_test_loss += sub_test_loss
            head_proc_time_list.append(head_end_time - head_start_time)
            tail_proc_time_list.append(tail_end_time - head_end_time)

    org_acc = 100.0 * org_correct_count / total
    print('[Before splitting]\tAverage Loss: {:.4f}, Accuracy: {}/{} [{:.4f}%]\n'.format(
        org_test_loss / total, org_correct_count, total, org_acc))
    split_acc = 100.0 * split_correct_count / total
    print('[After splitting]\tAverage Loss: {:.4f}, Accuracy: {}/{} [{:.4f}%]\n'.format(
        split_test_loss / total, split_correct_count, total, split_acc))
    print('Output file size at splitting point [KB]: {} +- {}'.format(
        np.average(file_size_list), np.std(file_size_list)))
    print('Local processing time [sec]: {} +- {}'.format(np.average(head_proc_time_list), np.std(head_proc_time_list)))
    print('Edge processing time [sec]: {} +- {}'.format(np.average(tail_proc_time_list), np.std(tail_proc_time_list)))


def split_original_model(model, input_shape, device, config, sensor_device, edge_device, partition_idx,
                         head_output_file_path, tail_output_file_path, require_test, spbit):
    print('Splitting an original DNN model')
    modules = list()
    z = torch.rand(1, *input_shape).to(device)
    module_util.extract_decomposable_modules(model, z, modules)
    head_module_list = list()
    tail_module_list = list()
    if partition_idx < 0:
        teacher_model_config = config['teacher_model']
        start_idx = teacher_model_config['start_idx']
        end_idx = teacher_model_config['end_idx']
        head_module_list.extend(modules[start_idx:end_idx])
        tail_module_list.extend(modules[end_idx:])
    else:
        head_module_list.extend(modules[:partition_idx])
        tail_module_list.extend(modules[partition_idx:])

    head_network = nn.Sequential(*head_module_list)
    tail_network = mimic_util.get_tail_network(config, tail_module_list)
    file_util.save_pickle(head_network.to(sensor_device), head_output_file_path)
    file_util.save_pickle(tail_network.to(edge_device), tail_output_file_path)
    if require_test:
        test_split_model(model, head_network, tail_network, sensor_device, edge_device, spbit, config)


def split_within_student_model(model, input_shape, device, config, teacher_model_type, sensor_device, edge_device,
                               partition_idx, head_output_file_path, tail_output_file_path, require_test, spbit):
    print('Splitting within a student DNN model')
    org_modules = list()
    z = torch.rand(1, *input_shape).to(device)
    plain_model = model.module if isinstance(model, (DataParallel, DistributedDataParallel)) else model
    module_util.extract_decomposable_modules(plain_model, z, org_modules)
    student_model = mimic_util.load_student_model(config, teacher_model_type, device)
    student_modules = list()
    module_util.extract_decomposable_modules(student_model, z, student_modules)
    head_module_list = list()
    tail_module_list = list()
    teacher_model_config = config['teacher_model']
    end_idx = teacher_model_config['end_idx']
    if partition_idx < 0:
        head_module_list.extend(student_modules)
    else:
        head_module_list.extend(student_modules[:partition_idx])
        tail_module_list.extend(student_modules[partition_idx:])

    tail_module_list.extend(org_modules[end_idx:])
    head_network = nn.Sequential(*head_module_list)
    tail_network = mimic_util.get_tail_network(config, tail_module_list)
    file_util.save_pickle(head_network.to(sensor_device), head_output_file_path)
    file_util.save_pickle(tail_network.to(edge_device), tail_output_file_path)
    if require_test:
        device = torch.device('cuda' if next(model.parameters()).is_cuda else 'cpu')
        mimic_model = mimic_util.get_mimic_model(config, model, teacher_model_type, teacher_model_config, device)
        test_split_model(mimic_model, head_network, tail_network, sensor_device, edge_device, spbit, config)


def convert_model(model, device, output_file_path):
    if device.type == 'cpu' and isinstance(model, nn.parallel.DataParallel):
        model = model.module

    for module in model.modules():
        module.to(device)
    file_util.save_pickle(model, output_file_path)


def run(args):
    print(args)
    config = yaml_util.load_yaml_file(args.config)
    sensor_device = torch.device('cpu' if args.scpu else 'cuda')
    edge_device = torch.device('cpu' if args.ecpu else 'cuda')
    partition_idx = args.partition
    head_output_file_path = args.head
    tail_output_file_path = args.tail
    input_shape = config['input_shape']
    if 'teacher_model' not in config:
        model = module_util.get_model(config, torch.device('cuda') if torch.cuda.is_available() else None)
        module_util.resume_from_ckpt(model, config['model'], False)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model, teacher_model_type =\
            mimic_util.get_org_model(config['teacher_model'], device)
        if args.org and head_output_file_path is not None and tail_output_file_path is not None:
            split_original_model(model, input_shape, device, config, sensor_device, edge_device, partition_idx,
                                 head_output_file_path, tail_output_file_path, args.test, args.spbit)
        elif args.mimic:
            model = mimic_util.get_mimic_model_easily(config, sensor_device)
            student_model_config = config['mimic_model']
            load_ckpt(student_model_config['ckpt'], model=model, strict=True)
            split_original_model(model, input_shape, device, config, sensor_device, edge_device, partition_idx,
                                 head_output_file_path, tail_output_file_path, args.test, args.spbit)
        elif head_output_file_path is not None and tail_output_file_path is not None:
            split_within_student_model(model, input_shape, device, config, teacher_model_type,
                                       sensor_device, edge_device, partition_idx,
                                       head_output_file_path, tail_output_file_path, args.test, args.spbit)

    if args.model is not None and args.device is not None:
        convert_model(model, torch.device(args.device), args.model)


if __name__ == '__main__':
    parser = get_argparser()
    run(parser.parse_args())
