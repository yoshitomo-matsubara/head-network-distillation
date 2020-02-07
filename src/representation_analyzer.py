import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.backends import cudnn
from torch.nn import DataParallel

from myutils.common import yaml_util
from structure.wrapper import RepresentationWrapper
from utils import main_util, mimic_util, module_util, module_wrap_util


def get_argparser():
    argparser = argparse.ArgumentParser(description='Layer-wise representation analyzer')
    argparser.add_argument('--config', required=True, help='yaml file path')
    argparser.add_argument('--split', default='train', help='dataset split')
    argparser.add_argument('--method', default='tsne', help='representation method')
    argparser.add_argument('-cpu', action='store_true', help='use CPU')
    return argparser


def extract_transformed_outputs(parent_module, transformed_output_list, name_list):
    for name, child_module in parent_module.named_children():
        if isinstance(child_module, RepresentationWrapper):
            transformed_output_list.append(child_module.get_transformed_list())
            name_list.append(type(child_module.org_module).__name__)
        elif list(child_module.children()):
            extract_transformed_outputs(child_module, transformed_output_list, name_list)
        else:
            print('RepresentationWrapper is missing for {}: {}'.format(name, type(child_module).__name__))


def assess_discriminabilities(transformed_outputs):
    value_list = list()
    for transformed_output in transformed_outputs:
        dist_list = list()
        for i, transformed_output_x in enumerate(transformed_output):
            for transformed_output_y in transformed_outputs[i + 1:]:
                dist = np.linalg.norm(transformed_output_x - transformed_output_y)
                dist_list.append(dist)
        value_list.append(np.mean(dist_list))
    return value_list


def analyze_with_mean_inputs(model, input_shape, data_loader, device, split_name, method):
    model = model.module if isinstance(model, DataParallel) else model
    input_batch = torch.rand(input_shape).unsqueeze(0).to(device)
    module_wrap_util.wrap_decomposable_modules(model, RepresentationWrapper, input_batch, method=method)
    if device.type == 'cuda':
        model = DataParallel(model)

    model.eval()
    accumulated_tensor_dict = dict()
    with torch.no_grad():
        for batch_idx, (sample_batch, targets) in enumerate(data_loader):
            for x, y in zip(sample_batch, targets):
                class_label = y.item()
                if class_label not in accumulated_tensor_dict:
                    accumulated_tensor_dict[class_label] = [x, 1]
                else:
                    accumulated_tensor_dict[class_label][0] += x
                    accumulated_tensor_dict[class_label][1] += 1

        mean_input_list = list()
        for y, (x, num_samples) in accumulated_tensor_dict.items():
            mean_x = x / num_samples
            mean_input_list.append(mean_x)
        mean_batch = torch.stack(mean_input_list)
        preds = model(mean_batch)

    transformed_output_list = list()
    name_list = list()
    extract_transformed_outputs(model, transformed_output_list, name_list)
    xs = list(range(len(name_list)))
    discriminabilities = assess_discriminabilities(transformed_output_list)
    plt.plot(xs, discriminabilities, label=method)
    plt.xticks(xs, name_list, rotation=90)
    plt.xlabel('Layer')
    plt.ylabel('Discriminability')
    plt.title(split_name)
    plt.legend()
    plt.show()


def run(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    if device.type == 'cuda':
        cudnn.benchmark = True

    config = yaml_util.load_yaml_file(args.config)
    train_loader, valid_loader, test_loader = main_util.get_data_loaders(config, False)
    input_shape = config['input_shape']
    if 'mimic_model' in config:
        model = mimic_util.get_mimic_model_easily(config, device)
        model_config = config['mimic_model']
    else:
        model = module_util.get_model(config, device)
        model_config = config['model']

    model_type, _, _, _ = module_util.resume_from_ckpt(model, model_config, args.init)
    split_name = args.split
    data_loader = train_loader if split_name == 'train' else valid_loader if split_name == 'valid' else test_loader
    analyze_with_mean_inputs(model, input_shape, data_loader, device, split_name, args.method)


if __name__ == '__main__':
    parser = get_argparser()
    run(parser.parse_args())
