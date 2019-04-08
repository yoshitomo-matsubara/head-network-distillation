import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

from myutils.common import file_util, yaml_util
from myutils.pytorch import func_util
from utils import mimic_util
from utils.dataset import general_util


def get_argparser():
    argparser = argparse.ArgumentParser(description='Mimic Learner')
    argparser.add_argument('--config', required=True, help='yaml file path')
    argparser.add_argument('--epoch', type=int, help='epoch (higher priority than config if set)')
    argparser.add_argument('--lr', type=float, help='learning rate (higher priority than config if set)')
    argparser.add_argument('--gpu', type=int, help='gpu number')
    argparser.add_argument('-init', action='store_true', help='overwrite checkpoint')
    return argparser


def train(student_model, teacher_model, train_loader, optimizer, criterion, epoch, device, interval, aux_weight=100.0):
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
        if isinstance(student_outputs, tuple):
            student_outputs, aux = student_outputs[0], student_outputs[1]
            loss = criterion(student_outputs, teacher_outputs) + aux_weight * nn.functional.cross_entropy(aux, targets)
        else:
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
    module = student_model.module if isinstance(student_model, nn.DataParallel) else student_model
    state = {
        'type': teacher_model_type,
        'model': module.state_dict(),
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
        gpu_number = args.gpu
        if gpu_number is not None and gpu_number >= 0:
            device += ':' + str(gpu_number)

    config = yaml_util.load_yaml_file(args.config)
    dataset_config = config['dataset']
    input_shape = config['input_shape']
    teacher_model_config = config['teacher_model']
    teacher_model, teacher_model_type = mimic_util.get_teacher_model(teacher_model_config, input_shape, device)
    student_model_config = config['student_model']
    student_model = mimic_util.get_student_model(teacher_model_type, student_model_config, dataset_config['name'])
    student_model = student_model.to(device)
    start_epoch, best_avg_loss = mimic_util.resume_from_ckpt(student_model_config['ckpt'], student_model,
                                                             is_student=True)
    if device == 'cuda':
        teacher_model = nn.DataParallel(teacher_model)
        student_model = nn.DataParallel(student_model)

    train_config = config['train']
    train_loader, valid_loader, _ =\
        general_util.get_data_loaders(dataset_config, batch_size=train_config['batch_size'], ae_model=None,
                                      reshape_size=input_shape[1:3], jpeg_quality=-1)
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
