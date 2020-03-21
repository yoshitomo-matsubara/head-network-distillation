import torch
from torch import nn
from torch.nn import DataParallel
from torch.nn.parallel.distributed import DistributedDataParallel

from models.autoencoder.base import BaseExtendedModel
from models.autoencoder.input_ae import InputAutoencoder, InputVAE
from models.autoencoder.middle_ae import MiddleAutoencoder
from myutils.common import yaml_util
from utils import module_util


def get_autoencoder(config, device=None, is_static=False):
    autoencoder = None
    ae_config = config['autoencoder']
    ae_type = ae_config['type']
    if ae_type == 'input_ae':
        autoencoder = InputAutoencoder(**ae_config['params'])
    elif ae_type == 'input_vae':
        autoencoder = InputVAE(**ae_config['params'], is_static=is_static)
    elif ae_type == 'middle_ae':
        autoencoder = MiddleAutoencoder(**ae_config['params'])

    if autoencoder is None:
        raise ValueError('ae_type `{}` is not expected'.format(ae_type))

    if device is None:
        return autoencoder, ae_type

    autoencoder = autoencoder.to(device)
    return autoencoder, ae_type


def extract_head_model(model, input_shape, device, partition_idx):
    if partition_idx is None or partition_idx == 0:
        return nn.Sequential()

    modules = list()
    module = model.module if isinstance(model, (DataParallel, DistributedDataParallel)) else model
    module_util.extract_decomposable_modules(module, torch.rand(1, *input_shape).to(device), modules)
    return nn.Sequential(*modules[:partition_idx]).to(device)


def get_head_model(config, input_shape, device):
    org_model_config = config['org_model']
    model_config = yaml_util.load_yaml_file(org_model_config['config'])
    sub_model_config = model_config['model']
    if sub_model_config['type'] == 'inception_v3':
        sub_model_config['params']['aux_logits'] = False

    model = module_util.get_model(model_config, device)
    module_util.resume_from_ckpt(model, sub_model_config, False)
    return extract_head_model(model, input_shape, device, org_model_config['partition_idx'])


def extend_model(autoencoder, model, input_shape, device, partition_idx, skip_bottleneck_size):
    if partition_idx is None or partition_idx == 0:
        return nn.Sequential(autoencoder, model)

    modules = list()
    module = model.module if isinstance(model, (DataParallel, DistributedDataParallel)) else model
    x = torch.rand(1, *input_shape).to(device)
    module_util.extract_decomposable_modules(module, x, modules)
    extended_model = BaseExtendedModel(modules[:partition_idx], autoencoder, modules[partition_idx:]).to(device)
    if not skip_bottleneck_size:
        extended_model.compute_ae_bottleneck_size(x, True)
    return extended_model


def get_extended_model(autoencoder, config, input_shape, device, skip_bottleneck_size=False):
    org_model_config = config['org_model']
    model_config = yaml_util.load_yaml_file(org_model_config['config'])
    sub_model_config = model_config['model']
    if sub_model_config['type'] == 'inception_v3':
        sub_model_config['params']['aux_logits'] = False

    model = module_util.get_model(model_config, device)
    module_util.resume_from_ckpt(model, sub_model_config, False)
    return extend_model(autoencoder, model, input_shape, device,
                        org_model_config['partition_idx'], skip_bottleneck_size), model
