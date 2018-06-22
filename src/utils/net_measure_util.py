import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable


def calc_sequential_feature_size(sequential, input_shape):
    input_data = Variable(torch.rand(input_shape).unsqueeze(0), requires_grad=True)
    return np.prod(sequential(input_data).unsqueeze(0).size())


def convert2kb(bandwidth_list):
    return np.array(bandwidth_list) / (8 * 1024)


def convert2accumulated(op_count_list):
    return np.array([sum(op_count_list[0:i]) for i in range(len(op_count_list))])


def plot_model_complexity(xs, op_count_list, layer_list, model_name):
    plt.semilogy(xs[1:], op_count_list, label=model_name)
    plt.xticks(xs[1:], layer_list[1:])
    plt.xlabel('Layer')
    plt.ylabel('Complexity')
    plt.legend()
    plt.show()


def plot_accumulated_model_complexity(xs, accumulated_op_counts, layer_list, model_name):
    plt.plot(xs[1:], accumulated_op_counts, label=model_name)
    plt.xticks(xs[1:], layer_list[1:])
    plt.xlabel('Layer')
    plt.ylabel('Accumulated Complexity')
    plt.legend()
    plt.show()


def plot_model_bandwidth(xs, bandwidths, layer_list, bandwidth_label, model_name):
    plt.semilogy(xs, bandwidths, label=model_name)
    plt.semilogy(xs, [bandwidths[0] for x in xs], '-', label='Input')
    plt.xticks(xs, layer_list)
    plt.xlabel('Layer')
    plt.ylabel(bandwidth_label)
    plt.legend()
    plt.show()


def plot_bandwidth_vs_model_complexity(bandwidths, op_count_list, bandwidth_label, model_name):
    plt.scatter(bandwidths[1:], op_count_list, label=model_name)
    plt.yscale('log')
    plt.xlabel(bandwidth_label)
    plt.ylabel('Complexity')
    plt.legend()
    plt.show()


def plot_accumulated_model_complexity_vs_bandwidth(accumulated_op_counts, bandwidths, bandwidth_label, model_name):
    plt.plot(accumulated_op_counts, bandwidths[1:], marker='o', label=model_name)
    plt.plot(accumulated_op_counts, [bandwidths[0] for x in accumulated_op_counts], '-', label='Input')
    plt.xlabel('Accumulated Complexity')
    plt.ylabel(bandwidth_label)
    plt.legend()
    plt.show()


def plot_model_complexity_and_bandwidth(op_count_list, accumulated_op_counts, bandwidths,
                                        layer_list, bandwidth_label, model_name):
    print('Number of Operations: %.5fM' % (sum(op_count_list) / 1e6))
    xs = np.arange(len(layer_list))
    plot_model_complexity(xs, op_count_list, layer_list, model_name)
    plot_accumulated_model_complexity(xs, accumulated_op_counts, layer_list, model_name)
    plot_model_bandwidth(xs, bandwidths, layer_list, bandwidth_label, model_name)
    plot_bandwidth_vs_model_complexity(bandwidths, op_count_list, bandwidth_label, model_name)
    plot_accumulated_model_complexity_vs_bandwidth(accumulated_op_counts, bandwidths, bandwidth_label, model_name)


def calc_model_complexity_and_bandwidth(model, input_shape, scaling=False, plot=True, model_name='network'):
    # Referred to https://zhuanlan.zhihu.com/p/33992733
    multiply_adds = False
    op_count_list = list()
    bandwidth_list = list()
    layer_list = list()

    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()
        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups)\
                     * (2 if multiply_adds else 1)
        bias_ops = 1 if self.bias is not None else 0
        params = output_channels * (kernel_ops + bias_ops)
        op_size = batch_size * params * output_height * output_width
        op_count_list.append(op_size)
        bandwidth_list.append(np.prod(output[0].size()))
        layer_list.append('Conv2d')

    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1
        weight_ops = self.weight.nelement() * (2 if multiply_adds else 1)
        bias_ops = self.bias.nelement()
        op_size = batch_size * (weight_ops + bias_ops)
        op_count_list.append(op_size)
        bandwidth_list.append(np.prod(output[0].size()))
        layer_list.append('Linear')

    def bn_hook(self, input, output):
        op_size = input[0].nelement()
        op_count_list.append(op_size)
        bandwidth_list.append(np.prod(output[0].size()))
        layer_list.append('BatchNorm2d')

    def relu_hook(self, input, output):
        op_size = input[0].nelement()
        op_count_list.append(op_size)
        bandwidth_list.append(np.prod(output[0].size()))
        layer_list.append('ReLU')

    def leaky_relu_hook(self, input, output):
        op_size = input[0].nelement()
        op_count_list.append(op_size)
        bandwidth_list.append(np.prod(output[0].size()))
        layer_list.append('LeakyReLU')

    def dropout_hook(self, input, output):
        op_size = input[0].nelement()
        op_count_list.append(op_size)
        bandwidth_list.append(np.prod(output[0].size()))
        layer_list.append('Dropout')

    def pooling_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()
        kernel_ops = self.kernel_size * self.kernel_size
        params = output_channels * kernel_ops
        op_size = batch_size * params * output_height * output_width
        op_count_list.append(op_size)
        bandwidth_list.append(np.prod(output[0].size()))
        layer_list.append('MaxPool2d')

    def move_next_layer(net):
        children = list(net.children())
        if not children:
            if isinstance(net, torch.nn.Conv2d):
                net.register_forward_hook(conv_hook)
            elif isinstance(net, torch.nn.Linear):
                net.register_forward_hook(linear_hook)
            elif isinstance(net, torch.nn.BatchNorm2d):
                net.register_forward_hook(bn_hook)
            elif isinstance(net, torch.nn.ReLU):
                net.register_forward_hook(relu_hook)
            elif isinstance(net, torch.nn.LeakyReLU):
                net.register_forward_hook(leaky_relu_hook)
            elif isinstance(net, torch.nn.Dropout):
                net.register_forward_hook(dropout_hook)
            elif isinstance(net, torch.nn.MaxPool2d) or isinstance(net, torch.nn.AvgPool2d):
                net.register_forward_hook(pooling_hook)
            else:
                if plot:
                    print('Non-registered instance:', type(net))
            return

        for child in children:
            move_next_layer(child)

    move_next_layer(model)
    bandwidth_list.append(np.prod(input_shape))
    layer_list.append('Input')
    input = torch.rand(input_shape).unsqueeze(0)
    output = model(input)
    bandwidths = convert2kb(bandwidth_list)
    bandwidth_label = 'Bandwidth [kB]'
    if scaling:
        bandwidths /= bandwidths[0]
        bandwidth_label = 'Scaled Bandwidth'

    accumulated_op_counts = convert2accumulated(op_count_list)
    if plot:
        plot_model_complexity_and_bandwidth(np.array(op_count_list), accumulated_op_counts,
                                            bandwidths, layer_list, bandwidth_label, model_name)
    return op_count_list, bandwidths, accumulated_op_counts
