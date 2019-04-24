import argparse

import torch
import torch.nn as nn

from myutils.common import file_util, yaml_util
from utils import mimic_util, module_util
from utils.dataset import general_util


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
    argparser.add_argument('-test', action='store_true', help='option to check if performance changes after splitting')
    return argparser


def predict(preds, targets):
    loss = nn.functional.cross_entropy(preds, targets)
    _, pred_labels = preds.max(1)
    correct_count = pred_labels.eq(targets).sum().item()
    return correct_count, loss.item()


def test_split_model(model, head_network, tail_network, sensor_device, edge_device, config):
    dataset_config = config['dataset']
    _, _, test_loader =\
        general_util.get_data_loaders(dataset_config, batch_size=config['test']['batch_size'],
                                      rough_size=config['train']['rough_size'],
                                      reshape_size=tuple(config['input_shape'][1:3]), jpeg_quality=-1)
    print('Testing..')
    device = 'cuda' if next(model.parameters()).is_cuda else 'cpu'
    head_network.eval()
    tail_network.eval()
    model.eval()
    split_correct_count = 0
    split_test_loss = 0
    org_correct_count = 0
    org_test_loss = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            total += targets.size(0)
            inputs, targets = inputs.to(sensor_device), targets.to(edge_device)
            zs = head_network(inputs)
            preds = tail_network(zs.to(edge_device))
            sub_correct_count, sub_test_loss = predict(preds, targets)
            split_correct_count += sub_correct_count
            split_test_loss += sub_test_loss
            inputs, targets = inputs.to(device), targets.to(device)
            preds = model(inputs)
            sub_correct_count, sub_test_loss = predict(preds, targets)
            org_correct_count += sub_correct_count
            org_test_loss += sub_test_loss

    org_acc = 100.0 * org_correct_count / total
    print('[Before splitting]\tAverage Loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        org_test_loss / total, org_correct_count, total, org_acc))
    split_acc = 100.0 * split_correct_count / total
    print('[After splitting]\tAverage Loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        split_test_loss / total, split_correct_count, total, split_acc))


def split_original_model(model, input_shape, config, sensor_device, edge_device, partition_idx,
                         head_output_file_path, tail_output_file_path, require_test):
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

    head_network = nn.Sequential(*head_module_list)
    tail_network = mimic_util.get_tail_network(config, tail_module_list)
    file_util.save_pickle(head_network.to(sensor_device), head_output_file_path)
    file_util.save_pickle(tail_network.to(edge_device), tail_output_file_path)
    if require_test:
        test_split_model(model, head_network, tail_network, sensor_device, edge_device, config)


def split_within_student_model(model, input_shape, config, teacher_model_type, sensor_device, edge_device,
                               partition_idx, head_output_file_path, tail_output_file_path, require_test):
    print('Splitting within a student DNN model')
    org_modules = list()
    z = torch.rand(1, *input_shape).to('cuda')
    module_util.extract_decomposable_modules(model, z, org_modules)
    student_model = mimic_util.load_student_model(config, teacher_model_type, 'cuda')
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
        device = 'cuda' if next(model.parameters()).is_cuda else 'cpu'
        mimic_model = mimic_util.get_mimic_model(config, model, teacher_model_type, teacher_model_config, device)
        test_split_model(mimic_model, head_network, tail_network, sensor_device, edge_device, config)


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
        model = module_util.get_model(config, 'cuda' if torch.cuda.is_available() else None)
        module_util.resume_from_ckpt(model, config['model'], False)
    else:
        model, teacher_model_type = mimic_util.get_org_model(config['teacher_model'], 'cuda')
        if args.org and head_output_file_path is not None and tail_output_file_path is not None:
            split_original_model(model, input_shape, config, sensor_device, edge_device, partition_idx,
                                 head_output_file_path, tail_output_file_path, args.test)
        elif head_output_file_path is not None and tail_output_file_path is not None:
            split_within_student_model(model, input_shape, config, teacher_model_type, sensor_device, edge_device,
                                       partition_idx, head_output_file_path, tail_output_file_path, args.test)

    if args.model is not None and args.device is not None:
        convert_model(model, args.device, args.model)


if __name__ == '__main__':
    parser = get_argparser()
    run(parser.parse_args())
