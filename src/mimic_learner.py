import argparse
import os

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import yaml

from utils import file_util, module_util
from utils.dataset import caltech_util
from models.mimic.densenet_mimic import *
from models.mimic.vgg_mimic import *


def get_argparser():
    argparser = argparse.ArgumentParser(description='Mimic Learner')
    argparser.add_argument('--config', required=True, help='yaml file path')
    argparser.add_argument('-init', action='store_true', help='overwrite checkpoint')
    return argparser


def resume_from_ckpt(ckpt_file_path, model):
    if not os.path.exists(ckpt_file_path):
        return 1

    print('Resuming from checkpoint..')
    checkpoint = torch.load(ckpt_file_path)
    model.load_state_dict(checkpoint['model'])
    start_epoch = checkpoint['epoch']
    return start_epoch


def extract_teacher_model(model, teacher_model_config):
    modules = list()
    module_util.extract_all_child_modules(model, modules)
    start_idx = teacher_model_config['start_idx']
    end_idx = teacher_model_config['end_idx']
    return nn.Sequential(*modules[start_idx:end_idx + 1])


def get_teacher_model(teacher_model_config, device):
    with open(teacher_model_config['config'], 'r') as fp:
        config = yaml.load(fp)

    model = module_util.get_model(device, config)
    model_config = config['model']
    resume_from_ckpt(model_config['ckpt'], model)
    return extract_teacher_model(model, teacher_model_config), model_config['type']


def get_student_model(teacher_model_type, teacher_model, student_model_config):
    student_model_type = student_model_config['type']
    if teacher_model_type == 'vgg' and student_model_type == 'vgg16_head_mimic':
        return Vgg16HeadMimic()
    elif teacher_model_type == 'densenet' and student_model_type == 'densenet121_head_mimic':
        return DenseNet121HeadMimic()
    raise ValueError('teacher_model_type `{}` is not expected'.format(teacher_model_type))


def get_criterion(criterion_config):
    criterion_type = criterion_config['type']
    params_config = criterion_config['params']
    if criterion_type == 'mse':
        return nn.MSELoss(**params_config)
    raise ValueError('criterion_type `{}` is not expected'.format(criterion_type))


def get_optimizer(optimizer_config, model):
    optimizer_type = optimizer_config['type']
    params_config = optimizer_config['params']
    if optimizer_type == 'sgd':
        return optim.SGD(model.parameters(), **params_config)
    elif optimizer_type == 'adam':
        return optim.Adam(model.parameters(), **params_config)
    elif optimizer_type == 'adagrad':
        return optim.Adagrad(model.parameters(), **params_config)
    raise ValueError('optimizer_type `{}` is not expected'.format(optimizer_type))


def train(student_model, teacher_model, train_loader, optimizer, criterion, epoch, device, interval):
    print('\nEpoch: %d' % epoch)
    student_model.train()
    teacher_model.eval()
    train_loss = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        student_outputs = student_model(inputs)
        teacher_outputs = teacher_model(inputs)
        loss = criterion(student_outputs, teacher_outputs)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        total += targets.size(0)
        if batch_idx > 0 and batch_idx % interval == 0:
            print('[{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(batch_idx * len(inputs), len(train_loader.sampler),
                                                           100.0 * batch_idx / len(train_loader), loss.item()))


def save_ckpt(student_model, epoch, ckpt_file_path, teacher_model_type):
    print('Saving..')
    state = {
        'type': teacher_model_type,
        'model': student_model.state_dict(),
        'epoch': epoch + 1,
        'student': True
    }
    file_util.make_parent_dirs(ckpt_file_path)
    torch.save(state, ckpt_file_path)


def run(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        cudnn.benchmark = True

    with open(args.config, 'r') as fp:
        student_config = yaml.load(fp)

    teacher_model_config = student_config['teacher_model']
    teacher_model, teacher_model_type = get_teacher_model(teacher_model_config, device)
    student_model_config = student_config['student_model']
    student_model = get_student_model(teacher_model_type, teacher_model, student_model_config)
    student_model = student_model.to(device)
    start_epoch = resume_from_ckpt(student_model_config['ckpt'], student_model, )
    train_config = student_config['train']
    dataset_config = student_config['dataset']
    train_loader, _, _ =\
        caltech_util.get_data_loaders(dataset_config['train'], batch_size=train_config['batch_size'], valid_rate=0,
                                      is_caltech256=dataset_config['name'] == 'caltech256', ae=None,
                                      reshape_size=tuple(student_config['input_shape'][1:3]), compression_quality=-1)
    criterion = get_criterion(train_config['criterion'])
    optimizer = get_optimizer(train_config['optimizer'], student_model)
    interval = train_config['interval']
    ckpt_file_path = student_model_config['ckpt']
    for epoch in range(start_epoch, train_config['epoch'] + 1):
        train(student_model, teacher_model, train_loader, optimizer, criterion, epoch, device, interval)
        save_ckpt(student_model, epoch, ckpt_file_path, teacher_model_type)


if __name__ == '__main__':
    parser = get_argparser()
    run(parser.parse_args())
