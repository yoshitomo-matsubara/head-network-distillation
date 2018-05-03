import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable


def plot_model_flops_and_size(total_flops, all_flops, all_bandwidth, all_layers):
    print('Number of FLOPs: %.5fM' % (total_flops / 1e6))
    print(all_flops)
    print(all_bandwidth)
    print(all_layers)
    xs = list(range(len(all_layers)))
    plt.semilogy(xs[1:], all_flops, label='network')
    plt.xticks(xs, all_layers)
    plt.xlabel('Layer')
    plt.ylabel('FLOPS')
    plt.legend()
    plt.show()

    all_bandwidth = all_bandwidth / (8 * (1024 ** 1))
    plt.semilogy(xs, all_bandwidth, label='network')
    plt.semilogy(xs, [all_bandwidth[0] for x in xs], '-', label='input')
    plt.xticks(xs, all_layers)
    plt.xlabel('Layer')
    plt.ylabel('Size [KB]')
    plt.legend()
    plt.show()


def calc_model_flops_and_size(model, input_shape):
    multiply_adds = False
    list_conv = list()
    list_linear = list()
    list_bn = list()
    list_relu = list()
    list_pooling = list()
    all_flops = list()
    all_bandwidth = list()
    all_layers = list()

    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()
        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups) * (
            2 if multiply_adds else 1)
        bias_ops = 1 if self.bias is not None else 0
        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width
        list_conv.append(flops)
        all_flops.append(flops)
        all_bandwidth.append(np.prod(output[0].size()))
        all_layers.append('Conv2d')

    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1
        weight_ops = self.weight.nelement() * (2 if multiply_adds else 1)
        bias_ops = self.bias.nelement()
        flops = batch_size * (weight_ops + bias_ops)
        list_linear.append(flops)
        all_flops.append(flops)
        all_bandwidth.append(np.prod(output[0].size()))
        all_layers.append('Linear')

    def bn_hook(self, input, output):
        flops = input[0].nelement()
        list_bn.append(flops)
        all_flops.append(flops)
        all_bandwidth.append(np.prod(output[0].size()))
        all_layers.append('BatchNorm2d')

    def relu_hook(self, input, output):
        flops = input[0].nelement()
        list_relu.append(flops)
        all_flops.append(flops)
        all_bandwidth.append(np.prod(output[0].size()))
        all_layers.append('ReLU')

    def pooling_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()
        kernel_ops = self.kernel_size * self.kernel_size
        bias_ops = 0
        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width
        list_pooling.append(flops)
        all_flops.append(flops)
        all_bandwidth.append(np.prod(output[0].size()))
        all_layers.append('MaxPool2d')

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
            elif isinstance(net, torch.nn.MaxPool2d) or isinstance(net, torch.nn.AvgPool2d):
                net.register_forward_hook(pooling_hook)
            else:
                print('Non-registered instance:', type(net))
            return

        for child in children:
            move_next_layer(child)

    move_next_layer(model)
    all_bandwidth.append(np.prod(input_shape))
    all_layers.append('Input')
    input = Variable(torch.rand(input_shape).unsqueeze(0), requires_grad=True)
    output = model(input)
    total_flops = sum(all_flops)
    plot_model_flops_and_size(total_flops, np.array(all_flops), np.array(all_bandwidth), all_layers)
    return total_flops, all_flops, all_bandwidth, all_layers
