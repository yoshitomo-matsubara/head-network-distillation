import argparse
import os

import torch
import torch.backends.cudnn as cudnn

from models.mimic.densenet_mimic import *
from models.mimic.inception_mimic import *
from models.mimic.resnet_mimic import *
from models.mimic.vgg_mimic import *
from models.classification.inception import Inception3
from myutils.common import file_util, yaml_util
from myutils.pytorch import func_util
from utils import module_util
from utils.dataset import general_util


def get_argparser():
    argparser = argparse.ArgumentParser(description='Mimic Learner')
    argparser.add_argument('--config', required=True, help='yaml file path')
    argparser.add_argument('--epoch', type=int, help='epoch (higher priority than config if set)')
    argparser.add_argument('--lr', type=float, help='learning rate (higher priority than config if set)')
    argparser.add_argument('-init', action='store_true', help='overwrite checkpoint')
    return argparser


def resume_from_ckpt(ckpt_file_path, model, is_student=False):
    if not os.path.exists(ckpt_file_path):
        print('{} checkpoint was not found at {}'.format("Student" if is_student else "Teacher", ckpt_file_path))
        if is_student:
            return 1, 1e60
        return 1

    print('Resuming from checkpoint..')
    checkpoint = torch.load(ckpt_file_path)
    state_dict = checkpoint['model']
    if isinstance(model.module, Inception3):
        for key in list(state_dict.keys()):
            if key.startswith('module.AuxLogits'):
                state_dict.pop(key)

    model.load_state_dict(state_dict)
    start_epoch = checkpoint['epoch']
    if is_student:
        return start_epoch, checkpoint['best_avg_loss']
    return start_epoch


def extract_teacher_model(model, input_shape, teacher_model_config):
    modules = list()
    module_util.extract_decomposable_modules(model, torch.rand(input_shape).unsqueeze(0), modules)
    start_idx = teacher_model_config['start_idx']
    end_idx = teacher_model_config['end_idx']
    return nn.Sequential(*modules[start_idx:end_idx])


def get_teacher_model(teacher_model_config, input_shape, device):
    teacher_config = yaml_util.load_yaml_file(teacher_model_config['config'])
    if teacher_config['model']['type'] == 'inception_v3':
        teacher_config['model']['params']['aux_logits'] = False

    model = module_util.get_model(teacher_config, device)
    model_config = teacher_config['model']
    resume_from_ckpt(model_config['ckpt'], model)
    return extract_teacher_model(model, input_shape, teacher_model_config), model_config['type']


def get_student_model(teacher_model_type, student_model_config):
    student_model_type = student_model_config['type']
    if teacher_model_type.startswith('densenet')\
            and student_model_type in ['densenet169_head_mimic', 'densenet201_head_mimic']:
        return DenseNetHeadMimic(teacher_model_type, student_model_config['version'])
    elif teacher_model_type == 'inception_v3' and student_model_type == 'inception_v3_head_mimic':
        return InceptionHeadMimic(student_model_config['version'])
    elif teacher_model_type.startswith('resnet') and student_model_type == 'resnet152_head_mimic':
        return ResNet152HeadMimic(student_model_config['version'])
    elif teacher_model_type == 'vgg' and student_model_type == 'vgg16_head_mimic':
        return Vgg16HeadMimic()
    raise ValueError('teacher_model_type `{}` is not expected'.format(teacher_model_type))


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
            print('[{}/{} ({:.0f}%)]\tAvg Loss: {:.6f}'.format(batch_idx * len(inputs), len(train_loader.sampler),
                                                               100.0 * batch_idx / len(train_loader),
                                                               loss.item() / targets.size(0)))


def validate(student_model, teacher_model, valid_loader, criterion, device):
    student_model.eval()
    teacher_model.eval()
    valid_loss = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valid_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            student_outputs = student_model(inputs)
            teacher_outputs = teacher_model(inputs)
            loss = criterion(student_outputs, teacher_outputs)
            valid_loss += loss.item()
            total += targets.size(0)

    avg_valid_loss = valid_loss / total
    print('Validation Loss: {:.6f}\tAvg Loss: {:.6f}'.format(valid_loss, avg_valid_loss))
    return avg_valid_loss


def save_ckpt(student_model, epoch, best_avg_loss, ckpt_file_path, teacher_model_type):
    print('Saving..')
    state = {
        'type': teacher_model_type,
        'model': student_model.state_dict(),
        'epoch': epoch + 1,
        'best_avg_loss': best_avg_loss,
        'student': True
    }
    file_util.make_parent_dirs(ckpt_file_path)
    torch.save(state, ckpt_file_path)


def run(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        cudnn.benchmark = True

    config = yaml_util.load_yaml_file(args.config)
    input_shape = config['input_shape']
    teacher_model_config = config['teacher_model']
    teacher_model, teacher_model_type = get_teacher_model(teacher_model_config, input_shape, device)
    student_model_config = config['student_model']
    student_model = get_student_model(teacher_model_type, student_model_config)
    student_model = student_model.to(device)
    start_epoch, best_avg_loss = resume_from_ckpt(student_model_config['ckpt'], student_model, is_student=True)
    train_config = config['train']
    dataset_config = config['dataset']
    train_loader, valid_loader, _ =\
        general_util.get_data_loaders(dataset_config['data'], batch_size=train_config['batch_size'], ae_model=None,
                                      reshape_size=input_shape[1:3], compression_quality=-1)
    criterion_config = train_config['criterion']
    criterion = func_util.get_loss(criterion_config['type'], criterion_config['params'])
    optim_config = train_config['optimizer']
    if args.lr is not None:
        optim_config['params']['lr'] = args.lr

    optimizer = func_util.get_optimizer(student_model, optim_config['type'], optim_config['params'])
    interval = train_config['interval']
    ckpt_file_path = student_model_config['ckpt']
    end_epoch = start_epoch + train_config['epoch'] if args.epoch is None else start_epoch + args.epoch
    for epoch in range(start_epoch, end_epoch):
        train(student_model, teacher_model, train_loader, optimizer, criterion, epoch, device, interval)
        avg_valid_loss = validate(student_model, teacher_model, valid_loader, criterion, device)
        if avg_valid_loss < best_avg_loss:
            best_avg_loss = avg_valid_loss
            save_ckpt(student_model, epoch, best_avg_loss, ckpt_file_path, teacher_model_type)


if __name__ == '__main__':
    parser = get_argparser()
    run(parser.parse_args())
