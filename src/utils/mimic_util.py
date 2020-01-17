import torch
from torch import nn

from models.classification.inception import Inception3
from models.mimic.densenet_mimic import DenseNetHeadMimic, DenseNetMimic
from models.mimic.inception_mimic import InceptionHeadMimic, InceptionMimic
from models.mimic.mobilenet_mimic import MobileNetHeadMimic, MobileNetMimic
from models.mimic.resnet_mimic import ResNet152HeadMimic, ResNetMimic
from myutils.common import file_util, yaml_util
from utils import mimic_util, module_util


def resume_from_ckpt(ckpt_file_path, model, is_student=False):
    if not file_util.check_if_exists(ckpt_file_path):
        print('{} checkpoint was not found at {}'.format("Student" if is_student else "Teacher", ckpt_file_path))
        if is_student:
            return 1, None
        return 1

    print('Resuming from checkpoint..')
    ckpt = torch.load(ckpt_file_path)
    state_dict = ckpt['model']
    if not is_student and isinstance(model, Inception3) or\
            (hasattr(model, 'module') and isinstance(model.module, Inception3)):
        for key in list(state_dict.keys()):
            if key.startswith('AuxLogits') or key.startswith('module.AuxLogits'):
                state_dict.pop(key)

    model.load_state_dict(state_dict)
    start_epoch = ckpt['epoch']
    if is_student:
        return start_epoch, ckpt['best_avg_loss'] if 'best_avg_loss' in ckpt else ckpt['best_valid_value']
    return start_epoch


def extract_teacher_model(model, input_shape, device, teacher_model_config):
    modules = list()
    module = model.module if isinstance(model, nn.DataParallel) else model
    module_util.extract_decomposable_modules(module, torch.rand(1, *input_shape).to(device), modules)
    start_idx = teacher_model_config['start_idx']
    end_idx = teacher_model_config['end_idx']
    return nn.Sequential(*modules[start_idx:end_idx]).to(device)


def get_teacher_model(teacher_model_config, input_shape, device):
    teacher_config = yaml_util.load_yaml_file(teacher_model_config['config'])
    model_config = teacher_config['model']
    if model_config['type'] == 'inception_v3':
        model_config['params']['aux_logits'] = False

    model = module_util.get_model(teacher_config, device)
    resume_from_ckpt(model_config['ckpt'], model)
    return extract_teacher_model(model, input_shape, device, teacher_model_config), model_config['type']


def get_student_model(teacher_model_type, student_model_config, dataset_name):
    student_model_type = student_model_config['type']
    student_model_version = student_model_config['version']
    params_config = student_model_config['params']
    if teacher_model_type.startswith('densenet')\
            and student_model_type in ['densenet169_head_mimic', 'densenet201_head_mimic']:
        return DenseNetHeadMimic(teacher_model_type, student_model_version, dataset_name, **params_config)
    elif teacher_model_type == 'inception_v3' and student_model_type == 'inception_v3_head_mimic':
        return InceptionHeadMimic(student_model_version, dataset_name, **params_config)
    elif teacher_model_type.startswith('resnet') and student_model_type == 'resnet152_head_mimic':
        return ResNet152HeadMimic(student_model_version, dataset_name, **params_config)
    elif teacher_model_type == 'mobilenet_v2' and student_model_type == 'mobilenet_v2_head_mimic':
        return MobileNetHeadMimic(student_model_version, **params_config)
    raise ValueError('teacher_model_type `{}` is not expected'.format(teacher_model_type))


def load_student_model(config, teacher_model_type, device):
    student_model_config = config['student_model']
    student_model = get_student_model(teacher_model_type, student_model_config, config['dataset']['name'])
    student_model = student_model.to(device)
    resume_from_ckpt(student_model_config['ckpt'], student_model, True)
    return student_model


def get_org_model(teacher_model_config, device):
    teacher_config = yaml_util.load_yaml_file(teacher_model_config['config'])
    if teacher_config['model']['type'] == 'inception_v3':
        teacher_config['model']['params']['aux_logits'] = False

    model = module_util.get_model(teacher_config, device)
    model_config = teacher_config['model']
    resume_from_ckpt(model_config['ckpt'], model)
    return model, model_config['type']


def get_tail_network(config, tail_modules):
    mimic_model_config = config['mimic_model']
    mimic_type = mimic_model_config['type']
    if mimic_type.startswith('densenet'):
        return DenseNetMimic(None, tail_modules)
    elif mimic_type.startswith('inception'):
        return InceptionMimic(None, tail_modules)
    elif mimic_type.startswith('resnet'):
        return ResNetMimic(None, tail_modules)
    elif mimic_type.startswith('mobilenet'):
        return MobileNetMimic(None, tail_modules)
    raise ValueError('mimic_type `{}` is not expected'.format(mimic_type))


def get_mimic_model(config, org_model, teacher_model_type, teacher_model_config, device, head_model=None):
    target_model = org_model.module if isinstance(org_model, nn.DataParallel) else org_model
    student_model =\
        load_student_model(config, teacher_model_type, device) if head_model is None else head_model.to(device)
    org_modules = list()
    input_batch = torch.rand(config['input_shape']).unsqueeze(0).to(device)
    module_util.extract_decomposable_modules(target_model, input_batch, org_modules)
    end_idx = teacher_model_config['end_idx']
    tail_modules = org_modules[end_idx:]
    mimic_model_config = config['mimic_model']
    mimic_type = mimic_model_config['type']
    if mimic_type.startswith('densenet'):
        mimic_model = DenseNetMimic(student_model, tail_modules)
    elif mimic_type.startswith('inception'):
        mimic_model = InceptionMimic(student_model, tail_modules)
    elif mimic_type.startswith('resnet'):
        mimic_model = ResNetMimic(student_model, tail_modules)
    elif mimic_type.startswith('mobilenet'):
        mimic_model = MobileNetMimic(student_model, tail_modules)
    else:
        raise ValueError('mimic_type `{}` is not expected'.format(mimic_type))
    return mimic_model.to(device)


def get_mimic_model_easily(config, device=torch.device('cpu')):
    teacher_model_config = config['teacher_model']
    org_model, teacher_model_type = get_org_model(teacher_model_config, device)
    return mimic_util.get_mimic_model(config, org_model, teacher_model_type, teacher_model_config, device)
