import argparse

import torch
import torch.nn as nn

from myutils.common import file_util, yaml_util
from utils import mimic_util, module_util


def get_argparser():
    argparser = argparse.ArgumentParser(description='PyTorch deployment helper')
    argparser.add_argument('--config', required=True, help='yaml file path')
    argparser.add_argument('--partition', type=int, default=-1, help='partition index (starts from 0)')
    argparser.add_argument('--head', help='output file path for head network pickle')
    argparser.add_argument('--tail', help='output file path for tail network pickle')
    argparser.add_argument('--model', help='output file path for original network pickle')
    argparser.add_argument('--device', help='device for original network pickle')
    argparser.add_argument('-org', action='store_true', help='option to split an original DNN model')
    argparser.add_argument('-scpu', action='store_true', help='option to make sensor-side model runnable without cuda')
    argparser.add_argument('-ecpu', action='store_true', help='option to make edge-server model runnable without cuda')
    return argparser


def split_original_model(model, input_shape, config, sensor_device, edge_device, partition_idx,
                         head_output_file_path, tail_output_file_path):
    print('Splitting an original DNN model')
    modules = list()
    z = torch.rand(1, *input_shape).to('cuda')
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

    for head_module in head_module_list:
        z = head_module(z)
        head_module.to(sensor_device)

    for tail_module in tail_module_list:
        z = tail_module(z)
        tail_module.to(edge_device)

    head_network = nn.Sequential(*head_module_list)
    tail_network = mimic_util.get_tail_network(config, tail_module_list)
    file_util.save_pickle(head_network, head_output_file_path)
    file_util.save_pickle(tail_network, tail_output_file_path)


def split_within_student_model(model, input_shape, config, teacher_model_type, sensor_device, edge_device,
                               partition_idx, head_output_file_path, tail_output_file_path):
    print('Splitting within a student DNN model')
    org_modules = list()
    z = torch.rand(1, *input_shape).to('cuda')
    module_util.extract_decomposable_modules(model, z, org_modules)
    student_model = mimic_util.load_student_model(config, teacher_model_type, 'cuda')
    student_modules = list()
    module_util.extract_decomposable_modules(student_model, z, student_modules)
    head_module_list = list()
    tail_module_list = list()
    end_idx = config['teacher_model']['end_idx']
    if partition_idx < 0:
        head_module_list.extend(student_modules)
    else:
        head_module_list.extend(student_modules[:partition_idx])
        tail_module_list.extend(student_modules[partition_idx:])

    tail_module_list.extend(org_modules[end_idx:])
    for head_module in head_module_list:
        z = head_module(z)
        head_module.to(sensor_device)

    for tail_module in tail_module_list:
        z = tail_module(z)
        tail_module.to(edge_device)

    head_network = nn.Sequential(*head_module_list)
    tail_network = mimic_util.get_tail_network(config, tail_module_list)
    file_util.save_pickle(head_network, head_output_file_path)
    file_util.save_pickle(tail_network, tail_output_file_path)


def convert_model(model, device, output_file_path):
    if device == 'cpu' and isinstance(model, nn.parallel.DataParallel):
        model = model.module

    for module in model.modules():
        module.to(device)
    file_util.save_pickle(model, output_file_path)


def run(args):
    config = yaml_util.load_yaml_file(args.config)
    sensor_device = 'cpu' if args.scpu else 'cuda'
    edge_device = 'cpu' if args.ecpu else 'cuda'
    partition_idx = args.partition
    head_output_file_path = args.head
    tail_output_file_path = args.tail
    input_shape = config['input_shape']
    if 'teacher_model' not in config:
        model = module_util.get_model(config)
    else:
        model, teacher_model_type = mimic_util.get_org_model(config['teacher_model'], 'cuda')
        if args.org and head_output_file_path is not None and tail_output_file_path is not None:
            split_original_model(model, input_shape, config, sensor_device, edge_device, partition_idx,
                                 head_output_file_path, tail_output_file_path)
        elif head_output_file_path is not None and tail_output_file_path is not None:
            split_within_student_model(model, input_shape, config, teacher_model_type, sensor_device, edge_device,
                                       partition_idx, head_output_file_path, tail_output_file_path)

    if args.model is not None and args.device is not None:
        convert_model(model, args.device, args.model)


if __name__ == '__main__':
    parser = get_argparser()
    run(parser.parse_args())
