import argparse
import os

import torchvision

import mimic_tester
from models.classification import *
from models.mock import *
from myutils.common import file_util, yaml_util
from utils import data_util, module_util, net_measure_util


def get_argparser():
    parser = argparse.ArgumentParser(description=os.path.basename(__file__))
    parser.add_argument('--isize', default='3,224,224', help='input data shape (delimited by comma)')
    parser.add_argument('--model', default='AlexNet', help='network model (default: alexnet)')
    parser.add_argument('--config', nargs='+', help='yaml file path')
    parser.add_argument('--pkl', help='pickle file path')
    parser.add_argument('-scale', action='store_true', help='bandwidth scaling option')
    parser.add_argument('-submodule', action='store_true', help='submodule extraction option')
    return parser


def get_model(model_type):
    lower_model_type = model_type.lower()
    if lower_model_type == 'mnist':
        return MnistLeNet5()
    elif lower_model_type == 'yolov2':
        return YOLOv2()
    elif lower_model_type == 'yolov3':
        return YOLOv3()
    elif model_type in torchvision.models.__dict__:
        return torchvision.models.__dict__[model_type]()
    raise ValueError('model_type `{}` is not expected'.format(model_type))


def read_config(config_file_path):
    config = yaml_util.load_yaml_file(config_file_path)
    dataset_name = config['dataset']['name']
    if not dataset_name.startswith('cifar') and not dataset_name.startswith('caltech'):
        return None, None, None

    model = module_util.get_model(config, 'cpu')
    model_type = config['model']['type']
    input_shape = config['input_shape']
    return model, model_type, input_shape


def analyze(model, input_shape, model_type, scaled, submoduled, plot):
    if submoduled:
        return net_measure_util.compute_model_complexity_and_bandwidth(model, model_type, input_shape,
                                                                       scaled=scaled, plot=plot)
    return net_measure_util.compute_layerwise_complexity_and_bandwidth(model, model_type, input_shape,
                                                                       scaled=scaled, plot=plot)


def analyze_single_model(config_file_path, args, plot=True):
    if file_util.check_if_exists(config_file_path):
        config = yaml_util.load_yaml_file(config_file_path)
        if 'teacher_model' in config:
            teacher_model_config = config['teacher_model']
            org_model, teacher_model_type = mimic_tester.get_org_model(teacher_model_config, 'cpu')
            model = mimic_tester.get_mimic_model(config, org_model, teacher_model_type, teacher_model_config, 'cpu')
            model_type = config['mimic_model']['type']
            input_shape = config['input_shape']
        else:
            model, model_type, input_shape = read_config(config_file_path)
    else:
        pickle_file_path = args.pkl
        model_type = args.model
        input_shape = list(data_util.convert2type_list(args.isize, ',', int))
        model = file_util.load_pickle(pickle_file_path) if file_util.check_if_exists(pickle_file_path)\
            else get_model(model_type)
    op_counts, bandwidths, accum_complexities =\
        analyze(model, input_shape, model_type, args.scale, args.submodule, plot)
    return op_counts, bandwidths, accum_complexities, model_type


def analyze_multiple_models(config_file_paths, args):
    op_counts_list = list()
    bandwidths_list = list()
    accum_complexities_list = list()
    model_type_list = list()
    for config_file_path in config_file_paths:
        op_counts, bandwidths, accum_complexities, model_type = analyze_single_model(config_file_path, args, False)
        op_counts_list.append(op_counts)
        bandwidths_list.append(bandwidths)
        accum_complexities_list.append(accum_complexities)
        model_type_list.append(model_type)

    net_measure_util.plot_model_complexities(op_counts_list, model_type_list)
    net_measure_util.plot_accumulated_model_complexities(accum_complexities_list, model_type_list)
    net_measure_util.plot_model_bandwidths(bandwidths_list, args.scale, model_type_list)


def run(args):
    config_file_paths = args.config
    if len(config_file_paths) <= 1:
        analyze_single_model(None if len(config_file_paths) == 0 else config_file_paths[0], args)
    else:
        analyze_multiple_models(config_file_paths, args)


if __name__ == '__main__':
    argparser = get_argparser()
    run(argparser.parse_args())
