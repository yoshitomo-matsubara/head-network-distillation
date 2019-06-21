import torchvision

from models.classification import *
from myutils.common import file_util


def use_multiple_gpus_if_available(model, device):
    return nn.DataParallel(model) if device.startswith('cuda') else model


def get_model(config, device=None):
    model = None
    model_config = config['model']
    model_type = model_config['type']
    if model_type == 'alexnet':
        model = AlexNet(**model_config['params'])
    elif model_type.startswith('densenet'):
        model = densenet_model(model_type, model_config['params'], model_config['pretrained'])
    elif model_type == 'lenet5':
        model = LeNet5(**model_config['params'])
    elif model_type.startswith('resnet'):
        model = resnet_model(model_type, model_config['params'], model_config['pretrained'])
    elif model_type.startswith('mobilenet'):
        model = mobilenet_model(model_type, model_config['params'])
    elif model_type.startswith('inception_v3'):
        model = inception_v3(model_config['pretrained'], **model_config['params'])
    elif model_type in torchvision.models.__dict__:
        model = torchvision.models.__dict__[model_type](**model_config['params'])

    if model is None:
        raise ValueError('model_type `{}` is not expected'.format(model_type))
    elif device is None:
        return model

    model = model.to(device)
    return use_multiple_gpus_if_available(model, device)


def resume_from_ckpt(model, model_config, init):
    ckpt_file_path = model_config['ckpt']
    if init or not file_util.check_if_exists(ckpt_file_path):
        return model_config['type'], 0, 1, ckpt_file_path

    print('Resuming from checkpoint..')
    checkpoint = torch.load(ckpt_file_path)
    model.load_state_dict(checkpoint['model'])
    model_type = checkpoint['type']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    return model_type, best_acc, start_epoch, ckpt_file_path


def extract_target_modules(parent_module, target_class, module_list):
    if isinstance(parent_module, target_class):
        module_list.append(parent_module)

    child_modules = list(parent_module.children())
    for child_module in child_modules:
        extract_target_modules(child_module, target_class, module_list)


def extract_all_child_modules(parent_module, module_list, extract_designed_module=True):
    child_modules = list(parent_module.children())
    if not child_modules or (not extract_designed_module and len(module_list) > 0 and
                             type(parent_module) != nn.Sequential):
        module_list.append(parent_module)
        return

    for child_module in child_modules:
        extract_all_child_modules(child_module, module_list, extract_designed_module)


def extract_decomposable_modules(parent_module, z, module_list, output_size_list=list(), first=True, exception_size=-1):
    parent_module.eval()
    child_modules = list(parent_module.children())
    if not child_modules:
        module_list.append(parent_module)
        try:
            z = parent_module(z)
            output_size_list.append([*z.size()])
            return z, True
        except (RuntimeError, ValueError):
            try:
                z = parent_module(z.view(z.size(0), exception_size))
                output_size_list.append([*z.size()])
                return z, True
            except RuntimeError:
                ValueError('Error\t', type(parent_module).__name__)
        return z, False

    try:
        expected_z = parent_module(z)
    except (RuntimeError, ValueError):
        try:
            resized_z = z.view(z.size(0), exception_size)
            expected_z = parent_module(resized_z)
            z = resized_z
        except RuntimeError:
            ValueError('Error\t', type(parent_module).__name__)
            return z, False

    submodule_list = list()
    sub_output_size_list = list()
    decomposable = True
    for child_module in child_modules:
        z, decomposable = extract_decomposable_modules(child_module, z, submodule_list, sub_output_size_list, False)
        if not decomposable:
            break

    if decomposable and expected_z.size() == z.size() and expected_z.isclose(z).all().item() == 1:
        module_list.extend(submodule_list)
        output_size_list.extend(sub_output_size_list)
        return expected_z, True

    if not first:
        module_list.append(parent_module)
        output_size_list.append([*expected_z.size()])
    return expected_z, True


def count_params(model):
    return sum(param.numel() for param in model.parameters())
