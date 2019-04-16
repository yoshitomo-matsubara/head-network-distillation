from collections import namedtuple

import torchvision.transforms as transforms

QuantizedTensor = namedtuple('QuantizedTensor', ['tensor', 'scale', 'zero_point'])


def convert2type_list(str_var, delimiter, var_type):
    return list(map(var_type, str_var.split(delimiter)))


def convert2type_range(str_var, delimiter, var_type):
    return range(*convert2type_list(str_var, delimiter, var_type))


def build_normalizer(dataset, mean=None, std=None):
    if mean is not None and std is not None:
        return transforms.Normalize(mean=mean, std=std)
    return transforms.Normalize(mean=dataset.mean(axis=(0, 1, 2)) / 255, std=dataset.std(axis=(0, 1, 2)) / 255)


# Referred to https://github.com/eladhoffer/utils.pytorch/blob/master/quantize.py
def quantize_tensor(x, num_bits=8):
    qmin = 0.0
    qmax = 2.0 ** num_bits - 1.0
    min_val, max_val = x.min(), x.max()
    scale = (max_val - min_val) / (qmax - qmin)
    initial_zero_point = qmin - min_val / scale
    zero_point = qmin if initial_zero_point < qmin else qmax if initial_zero_point > qmax else initial_zero_point
    zero_point = int(zero_point)
    qx = zero_point + x / scale
    qx.clamp_(qmin, qmax).round_()
    qx = qx.round().byte()
    return QuantizedTensor(tensor=qx, scale=scale, zero_point=zero_point)


def dequantize_tensor(q_x):
    return q_x.scale * (q_x.tensor.float() - q_x.zero_point)
