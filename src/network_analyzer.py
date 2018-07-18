import argparse
import os

import yaml
import torchvision

from models.cifar import *
from models.coco import *
from models.mnist import *
from utils import cifar_util, data_util, net_measure_util


def get_argparser():
    parser = argparse.ArgumentParser(description=os.path.basename(__file__))
    parser.add_argument('-isize', default='3,224,224', help='input data shape (delimited by comma)')
    parser.add_argument('-model', default='AlexNet', help='network model (default: alexnet)')
    parser.add_argument('-config', help='yaml file path')
    parser.add_argument('-scale', action='store_true', help='bandwidth scaling option')
    return parser


def get_model_and_input_shape(model_type, input_shape_str):
    if model_type == 'alexnet':
        return torchvision.models.AlexNet(), (3, 224, 224)
    elif model_type == 'vgg':
        return torchvision.models.vgg16(), (3, 224, 224)
    elif model_type == 'mnist':
        return LeNet5(), (1, 32, 32)
    elif model_type == 'yolov2':
        return YOLOv2(), (3, 448, 448)
    elif model_type == 'yolov3':
        return YOLOv3(), (3, 896, 896)

    input_shape = list(data_util.convert2type_list(input_shape_str, ',', int))
    return None, input_shape


def read_config(config_file_path):
    with open(config_file_path, 'r') as fp:
        config = yaml.load(fp)

    if config['dataset'].startswith('cifar'):
        model = cifar_util.get_model('cpu', config)
        model_type = config['model']['type']
        input_shape = config['input_shape']
    else:
        return None, None, None
    return model, model_type, input_shape


def run(args):
    if args.config is not None:
        model, model_type, input_shape = read_config(args.config)
    else:
        model_type = args.model
        model, input_shape = get_model_and_input_shape(model_type.lower(), args.isize)

    net_measure_util.calc_model_complexity_and_bandwidth(model, input_shape, scaled=args.scale, model_name=model_type)


if __name__ == '__main__':
    argparser = get_argparser()
    run(argparser.parse_args())
