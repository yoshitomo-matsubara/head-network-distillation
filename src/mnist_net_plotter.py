import argparse
import matplotlib.pyplot as plt
import os
from mnist_net import MyNet
from utils import data_util, net_measurer


def get_argparser():
    parser = argparse.ArgumentParser(description=os.path.basename(__file__))
    parser.add_argument('-input', default='1,28,28', help='input shape')
    parser.add_argument('-range1', default='5:55:15', help='channel range for 1st convolution layer')
    parser.add_argument('-range2', default='10:80:20', help='channel range for 2nd convolution layer')
    return parser


def plot_bandwidth_vs_complexity(bandwidths_list, accumulated_op_counts_list, label_list):
    for i in range(len(bandwidths_list)):
        plt.scatter(bandwidths_list[i][1:], accumulated_op_counts_list[i], label=label_list[i])
    plt.legend()
    plt.xlabel('Bandwidth [kB]')
    plt.ylabel('Accumulated Complexity')
    plt.show()


def plot(input_shape, first_param_range, second_param_range):
    op_counts_list, bandwidths_list, accumulated_op_counts_list, label_list = list(), list(), list(), list()
    for first_param in first_param_range:
        for second_param in second_param_range:
            model = MyNet(first_conv_size=first_param, second_conv_size=second_param)
            op_counts, bandwidths, accumulated_op_counts =\
                net_measurer.calc_model_complexity_and_bandwidth(model, input_shape, False)
            op_counts_list.append(op_counts)
            bandwidths_list.append(bandwidths)
            accumulated_op_counts_list.append(accumulated_op_counts)
            label_list.append('1st: ' + str(first_param) + ', 2nd: ' + str(second_param))
    plot_bandwidth_vs_complexity(bandwidths_list, accumulated_op_counts_list, label_list)


def run(args):
    input_shape = data_util.convert2type_list(args.input, ',', int)
    first_param_range = data_util.convert2type_range(args.range1, ':', int)
    second_param_range = data_util.convert2type_range(args.range2, ':', int)
    plot(input_shape, first_param_range, second_param_range)


if __name__ == '__main__':
    argparser = get_argparser()
    run(argparser.parse_args())
