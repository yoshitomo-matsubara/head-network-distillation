import argparse

import torch
import torch.nn as nn

import mimic_tester
from myutils.common import file_util, yaml_util
from utils import module_util


def get_argparser():
    argparser = argparse.ArgumentParser(description='PyTorch deployment helper')
    argparser.add_argument('--config', required=True, help='yaml file path')
    argparser.add_argument('--partition', type=int, default=-1, help='partition index (starts from 0)')
    argparser.add_argument('--head', required=True, help='output file path for head network pickle')
    argparser.add_argument('--tail', required=True, help='output file path for tail network pickle')
    argparser.add_argument('-org', action='store_true', help='option to split an original DNN model')
    argparser.add_argument('-cpu', action='store_true', help='option to make a model runnable without cuda')
    return argparser


def split_original_model(model, input_shape, teacher_model_config, device, partition_idx,
                         head_output_file_path, tail_output_file_path):
    print('Splitting an original DNN model')
    modules = list()
    module_util.extract_decomposable_modules(model, torch.rand(1, *input_shape).to(device), modules)
    if partition_idx < 0:
        start_idx = teacher_model_config['start_idx']
        end_idx = teacher_model_config['end_idx']
        head_network = nn.Sequential(*modules[start_idx:end_idx])
        tail_network = nn.Sequential(*modules[end_idx:])
    else:
        head_network = nn.Sequential(*modules[:partition_idx])
        tail_network = nn.Sequential(*modules[partition_idx:])

    file_util.save_pickle(head_network, head_output_file_path)
    file_util.save_pickle(tail_network, tail_output_file_path)


def split_within_student_model(model, input_shape, config, teacher_model_type, device, partition_idx,
                               head_output_file_path, tail_output_file_path):
    print('Splitting within a student DNN model')
    org_modules = list()
    module_util.extract_decomposable_modules(model, torch.rand(1, *input_shape).to(device), org_modules)
    student_model = mimic_tester.load_student_model(config, teacher_model_type, 'cuda')
    student_modules = list()
    module_util.extract_decomposable_modules(student_model, torch.rand(1, *input_shape).to(device), student_modules)
    head_network = nn.Sequential(*student_modules[:partition_idx])
    end_idx = config['teacher_model']['end_idx']
    tail_network = nn.Sequential(*student_modules[partition_idx:], *org_modules[end_idx:])
    file_util.save_pickle(head_network, head_output_file_path)
    file_util.save_pickle(tail_network, tail_output_file_path)


def run(args):
    config = yaml_util.load_yaml_file(args.config)
    device = 'cpu' if args.cpu else 'cuda'
    partition_idx = args.partition
    head_output_file_path = args.head
    tail_output_file_path = args.tail
    input_shape = config['input_shape']
    teacher_model_config = config['teacher_model']
    model, teacher_model_type = mimic_tester.get_org_model(teacher_model_config, 'cuda')
    if args.org:
        split_original_model(model, input_shape, teacher_model_config, device, partition_idx,
                             head_output_file_path, tail_output_file_path)
    else:
        split_within_student_model(model, input_shape, config, teacher_model_type, device, partition_idx,
                                   head_output_file_path, tail_output_file_path)


if __name__ == '__main__':
    parser = get_argparser()
    run(parser.parse_args())
