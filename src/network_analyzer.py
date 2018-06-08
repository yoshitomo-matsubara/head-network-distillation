import argparse
import os

from torchvision.models import *
from utils import data_util, net_measure_util
from model import *


def get_argparser():
    parser = argparse.ArgumentParser(description=os.path.basename(__file__))
    parser.add_argument('-isize', default='3,224,224', help='input data shape (delimited by comma)')
    parser.add_argument('-model', default='AlexNet', help='network model (default: alexnet)')
    return parser


def get_model_and_input_shape(model_type, input_shape_str):
    if model_type == 'alexnet':
        return AlexNet(), (3, 224, 224)
    elif model_type == 'vgg':
        return vgg16(), (3, 224, 224)
    elif model_type == 'mnist':
        return LeNet5(), (1, 32, 32)

    input_shape = list(data_util.convert2type_list(input_shape_str, ',', int))
    return None, input_shape


def run(args):
    model_type = args.model
    model, input_shape = get_model_and_input_shape(model_type.lower(), args.isize)
    net_measure_util.calc_model_complexity_and_bandwidth(model, input_shape, model_name=model_type)


if __name__ == '__main__':
    argparser = get_argparser()
    run(argparser.parse_args())
