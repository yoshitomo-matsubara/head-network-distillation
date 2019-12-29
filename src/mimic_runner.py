import argparse
import time

import torch
import torch.backends.cudnn as cudnn
from torch import nn
from torch.nn import DataParallel
from torch.nn.parallel.distributed import DistributedDataParallel

from myutils.common import file_util, yaml_util
from myutils.pytorch import func_util, module_util
from structure.logger import MetricLogger, SmoothedValue
from utils import main_util, mimic_util
from utils.dataset import general_util


def get_argparser():
    argparser = argparse.ArgumentParser(description='Mimic Learner')
    argparser.add_argument('--config', required=True, help='yaml file path')
    argparser.add_argument('--device', default='cuda', help='device')
    argparser.add_argument('--aux', type=float, default=100.0, help='auxiliary weight')
    argparser.add_argument('-test_only', action='store_true', help='only test model')
    argparser.add_argument('-student_only', action='store_true', help='test student model only')
    # distributed training parameters
    argparser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    argparser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return argparser


def distill_one_epoch(student_model, teacher_model, train_loader, optimizer, criterion,
                      epoch, device, interval, aux_weight):
    metric_logger = MetricLogger(delimiter='  ')
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('img/s', SmoothedValue(window_size=10, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)
    for sample_batch, targets in metric_logger.log_every(train_loader, interval, header):
        start_time = time.time()
        sample_batch, targets = sample_batch.to(device), targets.to(device)
        optimizer.zero_grad()
        student_outputs = student_model(sample_batch)
        teacher_outputs = teacher_model(sample_batch)
        if isinstance(student_outputs, tuple):
            student_outputs, aux = student_outputs[0], student_outputs[1]
            loss = criterion(student_outputs, teacher_outputs) + aux_weight * nn.functional.cross_entropy(aux, targets)
        else:
            loss = criterion(student_outputs, teacher_outputs)

        loss.backward()
        optimizer.step()
        batch_size = sample_batch.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])
        metric_logger.meters['img/s'].update(batch_size / (time.time() - start_time))


def compute_accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target[None])
    acc_list = []
    for k in topk:
        correct_k = correct[:k].flatten().sum(dtype=torch.float32)
        acc_list.append(correct_k * (100.0 / batch_size))
    return acc_list


@torch.no_grad()
def evaluate(model, data_loader, device, interval=1000, split_name='Test', title=None):
    if title is not None:
        print(title)

    num_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    model.eval()
    metric_logger = MetricLogger(delimiter='  ')
    header = '{}:'.format(split_name)
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, interval, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)

            acc1, acc5 = compute_accuracy(output, target, topk=(1, 5))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    top1_accuracy = metric_logger.acc1.global_avg
    top5_accuracy = metric_logger.acc5.global_avg
    print(' * Acc@1 {:.4f}\tAcc@5 {:.4f}\n'.format(top1_accuracy, top5_accuracy))
    torch.set_num_threads(num_threads)
    return metric_logger.acc1.global_avg


def validate(student_model_without_ddp, data_loader, config, device, distributed, device_ids):
    teacher_model_config = config['teacher_model']
    org_model, teacher_model_type = mimic_util.get_org_model(teacher_model_config, device)
    mimic_model = mimic_util.get_mimic_model(config, org_model, teacher_model_type, teacher_model_config,
                                             device, head_model=student_model_without_ddp)
    mimic_model_without_dp = mimic_model.module if isinstance(mimic_model, DataParallel) else mimic_model
    if distributed:
        mimic_model = DistributedDataParallel(mimic_model_without_dp, device_ids=device_ids)
    return evaluate(mimic_model, data_loader, device, split_name='Validation')


def save_ckpt(student_model, epoch, best_valid_value, ckpt_file_path, teacher_model_type):
    print('Saving..')
    module =\
        student_model.module if isinstance(student_model, (DataParallel, DistributedDataParallel)) else student_model
    state = {
        'type': teacher_model_type,
        'model': module.state_dict(),
        'epoch': epoch + 1,
        'best_valid_value': best_valid_value,
        'student': True
    }
    file_util.make_parent_dirs(ckpt_file_path)
    torch.save(state, ckpt_file_path)


def distill(train_loader, valid_loader, input_shape, aux_weight, config, device, distributed, device_ids):
    teacher_model_config = config['teacher_model']
    teacher_model, teacher_model_type = mimic_util.get_teacher_model(teacher_model_config, input_shape, device)
    module_util.freeze_module_params(teacher_model)
    student_model_config = config['student_model']
    student_model = mimic_util.get_student_model(teacher_model_type, student_model_config)
    student_model = student_model.to(device)
    start_epoch, best_valid_acc = mimic_util.resume_from_ckpt(student_model_config['ckpt'], student_model,
                                                              is_student=True)
    if best_valid_acc is None:
        best_valid_acc = 0.0

    train_config = config['train']
    criterion_config = train_config['criterion']
    criterion = func_util.get_loss(criterion_config['type'], criterion_config['params'])
    optim_config = train_config['optimizer']

    optimizer = func_util.get_optimizer(student_model, optim_config['type'], optim_config['params'])
    scheduler_config = train_config['scheduler']
    scheduler = func_util.get_scheduler(optimizer, scheduler_config['type'], scheduler_config['params'])
    interval = train_config['interval']
    if interval <= 0:
        num_batches = len(train_loader)
        interval = num_batches // 20 if num_batches >= 20 else 1

    student_model_without_ddp = student_model
    if distributed:
        teacher_model = DataParallel(teacher_model, device_ids=device_ids)
        student_model = DistributedDataParallel(student_model, device_ids=device_ids)
        student_model_without_ddp = student_model.module

    ckpt_file_path = student_model_config['ckpt']
    end_epoch = start_epoch + train_config['epoch']
    for epoch in range(start_epoch, end_epoch):
        distill_one_epoch(student_model, teacher_model, train_loader, optimizer, criterion,
                          epoch, device, interval, aux_weight)
        valid_acc = validate(student_model, valid_loader, config, device, distributed, device_ids)
        if valid_acc > best_valid_acc and main_util.is_main_process():
            print('Updating ckpt (Best top1 accuracy: {:.4f} -> {:.4f})'.format(best_valid_acc, valid_acc))
            best_valid_acc = valid_acc
            save_ckpt(student_model_without_ddp, epoch, best_valid_acc, ckpt_file_path, teacher_model_type)
        scheduler.step()

    del teacher_model
    del student_model


def run(args):
    distributed, device_ids = main_util.init_distributed_mode(args.world_size, args.dist_url)
    device = torch.device(args.device)
    if torch.cuda.is_available():
        cudnn.benchmark = True

    config = yaml_util.load_yaml_file(args.config)
    dataset_config = config['dataset']
    input_shape = config['input_shape']
    train_config = config['train']
    test_config = config['test']
    train_loader, valid_loader, test_loader =\
        general_util.get_data_loaders(dataset_config, batch_size=train_config['batch_size'],
                                      rough_size=train_config['rough_size'], reshape_size=input_shape[1:3],
                                      jpeg_quality=-1, test_batch_size=test_config['batch_size'],
                                      distributed=distributed)
    teacher_model_config = config['teacher_model']
    if not args.test_only:
        distill(train_loader, valid_loader, input_shape, args.aux, config, device, distributed, device_ids)

    org_model, teacher_model_type = mimic_util.get_org_model(teacher_model_config, device)
    mimic_model = mimic_util.get_mimic_model(config, org_model, teacher_model_type, teacher_model_config, device)
    mimic_model_without_dp = mimic_model.module if isinstance(mimic_model, DataParallel) else mimic_model
    file_util.save_pickle(mimic_model_without_dp, config['mimic_model']['ckpt'])
    if not args.student_only:
        evaluate(org_model.to(device), test_loader, device, title='[Original model]')
    
    if distributed:
        mimic_model = DistributedDataParallel(mimic_model_without_dp, device_ids=device_ids)
    evaluate(mimic_model, test_loader, device, title='[Mimic model]')


if __name__ == '__main__':
    parser = get_argparser()
    run(parser.parse_args())
