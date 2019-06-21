import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from utils import module_util


def calc_sequential_feature_size(sequential, input_shape):
    input_data = torch.rand(input_shape).unsqueeze(0)
    return np.prod(sequential(input_data).unsqueeze(0).size())


def convert2kb(data_size_list, bit=32):
    return np.array(data_size_list) * bit / (8 * 1024)


def convert2accumulated(op_count_list):
    return np.array([sum(op_count_list[0:i + 1]) for i in range(len(op_count_list))])


def format_metrics(data_size_list, op_count_list, scaled):
    data_sizes = convert2kb(data_size_list)
    data_size_label = 'Data Size [kB]'
    accum_complexities = convert2accumulated(op_count_list)
    accum_complexity_label = 'Accumulated Complexity'
    if scaled:
        data_sizes /= data_sizes[0]
        data_size_label = 'Scaled Data Size'
        accum_complexities /= accum_complexities[-1]
        accum_complexity_label = 'Scaled Accumulated Complexity'
    return data_sizes, accum_complexities, data_size_label, accum_complexity_label


def plot_model_complexity(xs, op_count_list, layer_list, model_name):
    plt.semilogy(xs[1:], op_count_list, label=model_name)
    plt.xticks(xs[1:], layer_list[1:], rotation=90, fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlim(xs[1] - 1, xs[-1] + 1)
    plt.xlabel('Layer', fontsize=14)
    plt.ylabel('Complexity', fontsize=14)
    plt.legend(fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_accumulated_model_complexity(xs, accumulated_op_counts, layer_list, accum_complexity_label, model_name):
    plt.plot(xs[1:], accumulated_op_counts, label=model_name)
    plt.xticks(xs[1:], layer_list[1:], rotation=90, fontsize=12)
    plt.xlim(xs[1] - 1, xs[-1] + 1)
    plt.xlabel('Layer', fontsize=14)
    plt.yticks(fontsize=12)
    plt.ylabel(accum_complexity_label, fontsize=14)
    plt.legend(fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_model_data_size(xs, data_sizes, layer_list, data_size_label, model_name):
    plt.semilogy(xs, data_sizes, label=model_name)
    plt.semilogy(xs, [data_sizes[0] for x in xs], '-', label='Input')
    plt.xticks(xs, layer_list, rotation=90, fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlim(xs[0] - 1, xs[-1] + 1)
    plt.xlabel('Layer', fontsize=14)
    plt.ylabel(data_size_label, fontsize=14)
    plt.legend(fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_data_size_vs_model_complexity(data_sizes, op_count_list, data_size_label, model_name):
    plt.scatter(data_sizes[1:], op_count_list, label=model_name)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.yscale('log')
    plt.xlabel(data_size_label, fontsize=14)
    plt.ylabel('Complexity', fontsize=14)
    plt.legend(fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_accumulated_model_complexity_vs_data_size(accumulated_op_counts, data_sizes,
                                                   data_size_label, accum_complexity_label, model_name):
    plt.plot(accumulated_op_counts, data_sizes[1:], marker='o', label=model_name)
    plt.plot(accumulated_op_counts, [data_sizes[0] for x in accumulated_op_counts], '-', label='Input')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel(accum_complexity_label, fontsize=14)
    plt.ylabel(data_size_label, fontsize=14)
    plt.legend(fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_accumulated_model_complexity_and_data_size(xs, accumulated_op_counts, data_sizes, layer_list,
                                                    data_size_label, accum_complexity_label):
    fig, ax1 = plt.subplots()
    ax1.semilogy(xs, data_sizes, '-')
    ax1.set_xticks(xs)
    ax1.set_xlim(xs[0] - 1, xs[-1] + 1)
    ax1.set_xticklabels(layer_list, fontsize=12)
    ax1.set_xlabel('Layer', fontsize=14)
    ax1.set_ylabel(data_size_label, color='b', fontsize=14)
    for tick in ax1.get_xticklabels():
        tick.set_rotation(90)
        tick.set_fontsize(12)

    for tick in ax1.get_yticklabels():
        tick.set_fontsize(12)

    ax2 = ax1.twinx()
    ax2.plot(xs[1:], accumulated_op_counts, 'r--')
    for tick in ax2.get_yticklabels():
        tick.set_fontsize(12)

    ax2.set_ylabel(accum_complexity_label, color='r', fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_model_complexity_and_data_size(op_count_list, accum_complexities, data_sizes, layer_list,
                                        data_size_label, accum_complexity_label, model_name):
    print('Number of Operations: {:.5f}M'.format(sum(op_count_list) / 1e6))
    xs = np.arange(len(layer_list))
    plot_model_complexity(xs, op_count_list, layer_list, model_name)
    plot_accumulated_model_complexity(xs, accum_complexities, layer_list, accum_complexity_label, model_name)
    plot_model_data_size(xs, data_sizes, layer_list, data_size_label, model_name)
    plot_data_size_vs_model_complexity(data_sizes, op_count_list, data_size_label, model_name)
    plot_accumulated_model_complexity_vs_data_size(accum_complexities, data_sizes,
                                                   data_size_label, accum_complexity_label, model_name)
    plot_accumulated_model_complexity_and_data_size(xs, accum_complexities, data_sizes, layer_list,
                                                    data_size_label, accum_complexity_label)


def compute_layerwise_complexity_and_data_size(model, model_name, input_shape, scaled=False, plot=True):
    # Referred to https://zhuanlan.zhihu.com/p/33992733
    #  and https://github.com/sovrasov/flops-counter.pytorch/blob/master/ptflops/flops_counter.py
    multiply_adds = False
    op_count_list = list()
    data_size_list = list()
    layer_list = list()

    def conv_hook(self, input_batch, output_batch):
        batch_size, input_channels, input_height, input_width = input_batch[0].size()
        output_channels, output_height, output_width = output_batch[0].size()
        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups)\
                     * (2 if multiply_adds else 1)
        bias_ops = 1 if self.bias is not None else 0
        params = output_channels * (kernel_ops + bias_ops)
        op_size = batch_size * params * output_height * output_width
        op_count_list.append(op_size)
        data_size_list.append(np.prod(output_batch[0].size()))
        layer_list.append('{}: {}'.format(type(self).__name__, len(layer_list)))

    def deconv_hook(self, input_batch, output_batch):
        batch_size, input_channels, input_height, input_width = input_batch[0].size()
        kernel_height, kernel_width = self.kernel_size
        in_channels = self.in_channels
        out_channels = self.out_channels
        groups = self.groups
        filters_per_channel = out_channels // groups
        conv_per_position_flops = kernel_height * kernel_width * in_channels * filters_per_channel
        active_elements_count = batch_size * input_height * input_width
        overall_conv_flops = conv_per_position_flops * active_elements_count
        bias_flops = 0
        if self.bias is not None:
            output_height, output_width = output_batch.shape[2:]
            bias_flops = out_channels * batch_size * output_height * output_height

        op_size = overall_conv_flops + bias_flops
        op_count_list.append(op_size)
        data_size_list.append(np.prod(output_batch[0].size()))
        layer_list.append('{}: {}'.format(type(self).__name__, len(layer_list)))

    def linear_hook(self, input_batch, output_batch):
        batch_size = input_batch[0].size(0) if input_batch[0].dim() == 2 else 1
        weight_ops = self.weight.nelement() * (2 if multiply_adds else 1)
        bias_ops = self.bias.nelement()
        op_size = batch_size * (weight_ops + bias_ops)
        op_count_list.append(op_size)
        data_size_list.append(np.prod(output_batch[0].size()))
        layer_list.append('{}: {}'.format(type(self).__name__, len(layer_list)))

    def pooling_hook(self, input_batch, output_batch):
        batch_size, input_channels, input_height, input_width = input_batch[0].size()
        output_channels, output_height, output_width = output_batch[0].size()
        kernel_ops = self.kernel_size * self.kernel_size
        params = output_channels * kernel_ops
        op_size = batch_size * params * output_height * output_width
        op_count_list.append(op_size)
        data_size_list.append(np.prod(output_batch[0].size()))
        layer_list.append('{}: {}'.format(type(self).__name__, len(layer_list)))

    def simple_hook(self, input_batch, output_batch):
        op_size = input_batch[0].nelement()
        op_count_list.append(op_size)
        data_size_list.append(np.prod(output_batch[0].size()))
        layer_list.append('{}: {}'.format(type(self).__name__, len(layer_list)))

    def move_next_layer(net):
        children = list(net.children())
        if not children:
            if isinstance(net, nn.Conv2d):
                net.register_forward_hook(conv_hook)
            elif isinstance(net, nn.ConvTranspose2d):
                net.register_forward_hook(deconv_hook)
            elif isinstance(net, nn.Linear):
                net.register_forward_hook(linear_hook)
            elif isinstance(net, (nn.MaxPool2d, nn.AvgPool2d)):
                net.register_forward_hook(pooling_hook)
            elif isinstance(net, (nn.BatchNorm2d, nn.ReLU, nn.Sigmoid,
                                  nn.LeakyReLU, nn.Dropout, nn.Softmax, nn.LogSoftmax)):
                net.register_forward_hook(simple_hook)
            else:
                print('Non-registered instance:', type(net))
            return

        for child in children:
            move_next_layer(child)

    move_next_layer(model)
    data_size_list.append(np.prod(input_shape))
    layer_list.append('Input')
    rand_input = torch.rand(input_shape).unsqueeze(0)
    model(rand_input)
    data_sizes, accum_complexities, data_size_label, accum_complexity_label =\
        format_metrics(data_size_list, op_count_list, scaled)
    if plot:
        plot_model_complexity_and_data_size(np.array(op_count_list), accum_complexities, data_sizes, layer_list,
                                            data_size_label, accum_complexity_label, model_name)
    return op_count_list, data_sizes, accum_complexities


def compute_model_complexity_and_data_size(model, model_name, input_shape, scaled=False, plot=True, **kwargs):
    submodules = list()
    output_sizes = list()
    module_util.extract_decomposable_modules(model, torch.rand(1, *input_shape), submodules, output_sizes, **kwargs)
    layer_list = ['Input']
    op_count_list = list()
    data_size_list = [np.prod(input_shape)]
    for i, submodule in enumerate(submodules):
        input_shape = input_shape if i == 0 else output_sizes[i - 1][1:]\
            if not isinstance(submodule, nn.Linear) else output_sizes[i - 1][1]
        module_name = '{}: {}'.format(type(submodule).__name__, i + 1).replace('_', ' ')
        layer_list.append(module_name)
        sub_op_counts, sub_data_sizes, _ =\
            compute_layerwise_complexity_and_data_size(submodule, module_name, input_shape, scaled=False, plot=False)
        op_count_list.append(sum(sub_op_counts))
        data_size_list.append(np.prod(output_sizes[i][1:]))

    data_sizes, accum_complexities, data_size_label, accum_complexity_label =\
        format_metrics(data_size_list, op_count_list, scaled)
    if plot:
        plot_model_complexity_and_data_size(np.array(op_count_list), accum_complexities, data_sizes, layer_list,
                                            data_size_label, accum_complexity_label, model_name)
    return op_count_list, data_sizes, accum_complexities


def plot_model_complexities(op_counts_list, model_type_list):
    for i in range(len(op_counts_list)):
        op_counts = op_counts_list[i]
        plt.semilogy(list(range(len(op_counts))), op_counts, label=model_type_list[i])

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Layer', fontsize=14)
    plt.ylabel('Complexity', fontsize=14)
    plt.legend(fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_accumulated_model_complexities(accum_complexities_list, model_type_list):
    for i in range(len(accum_complexities_list)):
        accum_complexities = accum_complexities_list[i]
        plt.plot(list(range(len(accum_complexities))), accum_complexities, label=model_type_list[i])

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Layer', fontsize=14)
    plt.ylabel('Accumulated Complexity', fontsize=14)
    plt.legend(fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_model_data_sizes(data_sizes_list, scaled, model_type_list):
    max_length = 0
    for i in range(len(data_sizes_list)):
        data_sizes = data_sizes_list[i]
        plt.semilogy(list(range(len(data_sizes))), data_sizes, label=model_type_list[i])
        if len(data_sizes) > max_length:
            max_length = len(data_sizes)

    xs = list(range(max_length))
    plt.semilogy(xs, [data_sizes[0] for _ in xs], '-', label='Input')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Layer', fontsize=14)
    plt.ylabel('Scaled Data Size' if scaled else 'Data Size [kB]', fontsize=14)
    plt.legend(fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_teacher_and_student_complexities(teacher_complexities, student_complexities, names=None):
    xs = np.array(list(range(len(teacher_complexities))))
    plt.bar(xs - 0.125, teacher_complexities, width=0.25, label='Teacher')
    plt.bar(xs + 0.125, student_complexities, width=0.25, label='Student')
    plt.xticks(xs, ['Ver.{}'.format(x + 1) for x in xs] if names is None else names, fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylabel('Total complexity', fontsize=14)
    for i in range(len(teacher_complexities)):
        txt = '{:.1f}x\nfaster'.format(teacher_complexities[i] / student_complexities[i])
        plt.annotate(txt, (xs[i] + 0.05, student_complexities[i] + 10 ** int(np.log10(student_complexities[i])) / 5))

    plt.legend(fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_bottleneck_data_size_vs_complexity(teacher_data_sizes, teacher_complexities,
                                            student_data_sizes, student_complexities, names=None):
    plt.scatter(teacher_data_sizes, teacher_complexities, label='Teacher')
    plt.scatter(student_data_sizes, student_complexities, label='Student')
    for i in range(len(teacher_data_sizes)):
        plt.arrow(teacher_data_sizes[i], teacher_complexities[i], student_data_sizes[i] - teacher_data_sizes[i],
                  student_complexities[i] - teacher_complexities[i], color='black', length_includes_head=True,
                  head_length=10 ** int(np.log10(student_complexities[i])), head_width=0.005, zorder=0)
        if names is not None:
            plt.annotate(names[i], (teacher_data_sizes[i] - 0.005,
                                    teacher_complexities[i] + 10 ** int(np.log10(teacher_complexities[i]) - 1)))

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Scaled Bottleneck Data Size', fontsize=14)
    plt.ylabel('Total complexity', fontsize=14)
    plt.legend(fontsize=13)
    plt.tight_layout()
    plt.show()
