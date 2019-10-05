import pickle
import sys

import numpy as np
import torch.nn as nn

from utils import data_util, module_util


class BaseAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, sample_batch):
        raise NotImplementedError('forward function must be implemented')


class BaseExtendedModel(nn.Module):
    def __init__(self, head_modules, autoencoder, tail_modules):
        super().__init__()
        self.head_model = nn.Sequential(*head_modules)
        self.autoencoder = autoencoder
        self.tail_model = nn.Sequential(*tail_modules[:-1])
        self.linear = tail_modules[-1]

    def forward(self, sample_batch):
        zs = self.head_model(sample_batch)
        zs = self.autoencoder(zs)
        zs = self.tail_model(zs)
        return self.linear(zs.view(zs.size(0), -1))

    def compute_ae_bottleneck_size(self, x, print_info=False):
        z = self.head_model(x)
        modules = list()
        module_util.extract_decomposable_modules(self.autoencoder, z, modules)
        modules = [module.to(x.device) for module in modules]
        org_size = np.prod(x.size())
        min_rate = None
        bo = None
        bqo = None
        for i in range(len(modules)):
            if isinstance(modules[i], nn.Linear):
                z = z.view(z.size(0), -1)

            z = modules[i](z)
            rate = np.prod(z.size()) / org_size
            if min_rate is None or rate < min_rate:
                min_rate = rate
                bo = pickle.dumps(z)
                bqo = pickle.dumps(data_util.quantize_tensor(z))

        output_data_size = sys.getsizeof(bo) / 1024
        quantized_output_data_size = sys.getsizeof(bqo) / 1024
        if print_info:
            print('[Autoencoder bottleneck]')
            print('Scaled output size: {} [%]\tOutput data size: {} [KB]\tQuantized output data size: {} [KB]'.format(
                min_rate * 100.0, output_data_size, quantized_output_data_size)
            )
        # Scaled output size, Output data size [KB], Quantized output data size [KB]
        return min_rate, output_data_size, quantized_output_data_size
