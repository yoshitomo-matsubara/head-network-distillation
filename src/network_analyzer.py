import argparse
import os

import numpy as np
import torchvision

from models.classification.lenet5 import MnistLeNet5
from myutils.common import file_util, yaml_util
from utils import data_util, mimic_util, module_util, net_measure_util


def get_argparser():
    parser = argparse.ArgumentParser(description=os.path.basename(__file__))
    parser.add_argument('--isize', default='3,224,224', help='input data shape (delimited by comma)')
    parser.add_argument('--model', default='alexnet', help='network model (default: alexnet)')
    parser.add_argument('--config', nargs='+', help='yaml file path')
    parser.add_argument('--pkl', help='pickle file path')
    parser.add_argument('-scale', action='store_true', help='data size scaling option')
    parser.add_argument('-submodule', action='store_true', help='submodule extraction option')
    parser.add_argument('-ts', action='store_true', help='teacher-student models option')
    return parser


def get_model(model_type):
    lower_model_type = model_type.lower()
    if lower_model_type == 'mnist':
        return MnistLeNet5()
    elif model_type in torchvision.models.__dict__:
        return torchvision.models.__dict__[model_type]()
    raise ValueError('model_type `{}` is not expected'.format(model_type))


def read_config(config_file_path):
    config = yaml_util.load_yaml_file(config_file_path)
    if config['model']['type'] == 'inception_v3':
        config['model']['params']['aux_logits'] = False

    model = module_util.get_model(config, 'cpu')
    model_type = config['model']['type']
    input_shape = config['input_shape']
    return model, model_type, input_shape


def analyze(model, input_shape, model_type, scaled, submoduled, plot):
    if submoduled:
        return net_measure_util.compute_model_complexity_and_data_size(model, model_type, input_shape,
                                                                       scaled=scaled, plot=plot)
    return net_measure_util.compute_layerwise_complexity_and_data_size(model, model_type, input_shape,
                                                                       scaled=scaled, plot=plot)


def analyze_single_model(config_file_path, args, plot=True):
    if file_util.check_if_exists(config_file_path):
        config = yaml_util.load_yaml_file(config_file_path)
        if 'teacher_model' in config:
            teacher_model_config = config['teacher_model']
            org_model, teacher_model_type = mimic_util.get_org_model(teacher_model_config, 'cpu')
            model = mimic_util.get_mimic_model(config, org_model, teacher_model_type, teacher_model_config, 'cpu')
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
    op_counts, data_sizes, accum_complexities =\
        analyze(model, input_shape, model_type, args.scale, args.submodule, plot)
    return op_counts, data_sizes, accum_complexities, model_type


def analyze_multiple_models(config_file_paths, args):
    op_counts_list = list()
    data_sizes_list = list()
    accum_complexities_list = list()
    model_type_list = list()
    for config_file_path in config_file_paths:
        op_counts, data_sizes, accum_complexities, model_type = analyze_single_model(config_file_path, args, False)
        op_counts_list.append(op_counts)
        data_sizes_list.append(data_sizes)
        accum_complexities_list.append(accum_complexities)
        model_type_list.append(model_type)

    net_measure_util.plot_model_complexities(op_counts_list, model_type_list)
    net_measure_util.plot_accumulated_model_complexities(accum_complexities_list, model_type_list)
    net_measure_util.plot_model_data_sizes(data_sizes_list, args.scale, model_type_list)


def get_teacher_and_student_models(mimic_config, input_shape):
    teacher_model_config = mimic_config['teacher_model']
    teacher_model, teacher_model_type = mimic_util.get_teacher_model(teacher_model_config, input_shape, 'cpu')
    student_model = mimic_util.get_student_model(teacher_model_type, mimic_config['student_model'])
    return teacher_model_type, teacher_model, student_model


def analyze_teacher_student_models(mimic_config_file_paths, args):
    scaled = args.scale
    submoduled = args.submodule
    model_type_list = list()
    teacher_complexity_list = list()
    student_complexity_list = list()
    teacher_data_size_list = list()
    student_data_size_list = list()
    for mimic_config_file_path in mimic_config_file_paths:
        mimic_config = yaml_util.load_yaml_file(mimic_config_file_path)
        input_shape = mimic_config['input_shape']
        teacher_model_type, teacher_model, student_model = get_teacher_and_student_models(mimic_config, input_shape)
        _, teacher_data_sizes, teacher_accum_complexities = analyze(teacher_model, input_shape, None, scaled=scaled,
                                                                    submoduled=submoduled, plot=False)
        _, student_data_sizes, student_accum_complexities = analyze(student_model, input_shape, None, scaled=scaled,
                                                                    submoduled=submoduled, plot=False)
        student_model_config = mimic_config['student_model']
        student_model_version = student_model_config['version']
        made_bottleneck = student_model_version.endswith('b')
        model_type_list.append('Ver.{}'.format(student_model_version))
        teacher_complexity_list.append(teacher_accum_complexities[-1])
        teacher_data_size_list.append(teacher_data_sizes[-1] / teacher_data_sizes[0])
        bottleneck_idx = np.argmin(student_data_sizes) if made_bottleneck else -1
        if student_data_sizes[bottleneck_idx] >= student_data_sizes[0] or not made_bottleneck:
            student_complexity_list.append(student_accum_complexities[-1])
            student_data_size_list.append(student_data_sizes[-1] / student_data_sizes[0])
        else:
            student_complexity_list.append(student_accum_complexities[bottleneck_idx - 1])
            student_data_size_list.append(student_data_sizes[bottleneck_idx] / student_data_sizes[0])

    net_measure_util.plot_teacher_and_student_complexities(teacher_complexity_list, student_complexity_list,
                                                           model_type_list)
    net_measure_util.plot_bottleneck_data_size_vs_complexity(teacher_data_size_list, teacher_complexity_list,
                                                             student_data_size_list, student_complexity_list,
                                                             model_type_list)


def run(args):
    config_file_paths = args.config
    if config_file_paths is None or (len(config_file_paths) <= 1 and not args.ts):
        analyze_single_model(None if config_file_paths is None else config_file_paths[0], args)
    elif not args.ts:
        analyze_multiple_models(config_file_paths, args)
    else:
        analyze_teacher_student_models(config_file_paths, args)


if __name__ == '__main__':
    argparser = get_argparser()
    run(argparser.parse_args())
