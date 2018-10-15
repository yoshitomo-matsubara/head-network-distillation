import torchvision

from autoencoders import *
from models.classification import *


def get_model(device, config):
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


def get_autoencoder(cuda_available, config):
    ae_config = config['autoencoder']
    ae_type = ae_config['type']
    if ae_type == 'vae':
        ae = VAE(**ae_config['params'])
    else:
        ValueError('ae_type `{}` is not expected'.format(ae_type))

    if cuda_available:
        ae.cuda()
    return ae


def extract_all_child_modules(parent_module, module_list):
    child_models = list(parent_module.children())
    if not child_models:
        module_list.append(parent_module)
        return

    for child_module in child_models:
        extract_all_child_modules(child_module, module_list)
