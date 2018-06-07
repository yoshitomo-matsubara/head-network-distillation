import argparse
import os

from torchvision.models import *
from utils import data_util, net_measurer
from model import *


def get_argparser():
    parser = argparse.ArgumentParser(description=os.path.basename(__file__))
    parser.add_argument('-isize', default='3,224,224', help='input data shape (delimited by comma)')
    parser.add_argument('-model', default='AlexNet', help='network model (default: alexnet)')
    return parser


def get_model(model_type):
    if model_type == 'alexnet':
        return AlexNet()
    elif model_type == 'vgg':
        return VGG()
    elif model_type == 'mnist':
        return LeNet5()
    return None


def run(args):
    input_shape = data_util.convert2type_list(args.isize, ',', int)
    model_type = args.model
    model = get_model(model_type.lower())
    net_measurer.calc_model_complexity_and_bandwidth(model, list(input_shape), model_name=model_type)


if __name__ == '__main__':
    argparser = get_argparser()
    run(argparser.parse_args())
