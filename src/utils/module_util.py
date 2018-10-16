import torchvision

from autoencoders.ae import *
from models.classification import *


def get_model(config, device):
    model_config = config['model']
    model_type = model_config['type']
    if model_type == 'alexnet':
        model = AlexNet(**model_config['params'])
    elif model_type == 'densenet':
        model = DenseNet(**model_config['params'])
    elif model_type == 'lenet5':
        model = LeNet5(**model_config['params'])
    elif model_type.startswith('resnet'):
        model = resnet_model(model_type, model_config['params'])
    elif model_type == 'vgg16':
        model = torchvision.models.vgg16()
    else:
        ValueError('model_type `{}` is not expected'.format(model_type))

    model = model.to(device)
    if device == 'cuda':
        model = torch.nn.DataParallel(model)
    return model


def get_autoencoder(config, device):
    ae_config = config['autoencoder']
    ae_type = ae_config['type']
    if ae_type == 'vae':
        ae_model = VAE(**ae_config['params'])
    else:
        ValueError('ae_type `{}` is not expected'.format(ae_type))

    ae_model = ae_model.to(device)
    return ae_model


def extract_all_child_modules(parent_module, module_list, extract_designed_module=True):
    child_modules = list(parent_module.children())
    if not child_modules or (not extract_designed_module and len(module_list) > 0 and
                             type(parent_module) != nn.Sequential):
        module_list.append(parent_module)
        return

    for child_module in child_modules:
        extract_all_child_modules(child_module, module_list, extract_designed_module)
