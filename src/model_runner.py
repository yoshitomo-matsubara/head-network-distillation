import argparse
import datetime
import time
import numpy as np
import torch
from torch import distributed as dist
from torch.backends import cudnn
from torch.nn import DataParallel
from torch.nn.parallel.distributed import DistributedDataParallel

from myutils.common import file_util, yaml_util
from myutils.pytorch import func_util
from structure.logger import MetricLogger, SmoothedValue
from utils import main_util, mimic_util, module_util


def get_argparser():
    argparser = argparse.ArgumentParser(description='PyTorch model runner')
    argparser.add_argument('--config', required=True, help='yaml file path')
    argparser.add_argument('--epoch', type=int, help='epoch (higher priority than config if set)')
    argparser.add_argument('--lr', type=float, help='learning rate (higher priority than config if set)')
    argparser.add_argument('-init', action='store_true', help='overwrite checkpoint')
    argparser.add_argument('-evaluate', action='store_true', help='evaluation option')
    # distributed training parameters
    argparser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    argparser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return argparser


def train_epoch(model, train_loader, optimizer, criterion, epoch, device, interval):
    model.train()
    metric_logger = MetricLogger(delimiter='  ')
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('img/s', SmoothedValue(window_size=10, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)
    for sample_batch, targets in metric_logger.log_every(train_loader, interval, header):
        start_time = time.time()
        sample_batch, targets = sample_batch.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(sample_batch)
        loss = sum((criterion(o, targets) for o in outputs)) if isinstance(outputs, tuple)\
            else criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        batch_size = sample_batch.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])
        metric_logger.meters['img/s'].update(batch_size / (time.time() - start_time))


def save_ckpt(model, acc, epoch, ckpt_file_path, model_type):
    print('Saving..')
    state = {
        'type': model_type,
        'model': model.state_dict(),
        'acc': acc,
        'epoch': epoch,
    }
    file_util.make_parent_dirs(ckpt_file_path)
    torch.save(state, ckpt_file_path)


def test(model, data_loader, device, interval=1000, split_name='Test'):
    num_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    model.eval()
    metric_logger = MetricLogger(delimiter='  ')
    header = '{}:'.format(split_name)
    proc_time_list = list()
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, interval, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            start_time = time.time()
            output = model(image)
            end_time = time.time()
            proc_time_list.append(end_time - start_time)
            acc1, acc5 = main_util.compute_accuracy(output, target, topk=(1, 5))
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
    print('Processing time [sec]: {} +- {}'.format(np.average(proc_time_list), np.std(proc_time_list)))
    torch.set_num_threads(num_threads)
    return metric_logger.acc1.global_avg


def validate(model, valid_loader, device):
    acc = test(model, valid_loader, device, split_name='Validation')
    return acc


def train(model, train_loader, valid_loader, best_valid_acc, criterion, device, distributed, device_ids, train_config,
          num_epochs, start_epoch, init_lr, ckpt_file_path, model_type):
    model_without_ddp = model
    if distributed:
        model = DistributedDataParallel(model_without_ddp, device_ids=device_ids)
    elif device.type == 'cuda':
        model = DataParallel(model_without_ddp)

    optim_config = train_config['optimizer']
    if init_lr is not None:
        optim_config['params']['lr'] = init_lr

    optimizer = func_util.get_optimizer(model, optim_config['type'], optim_config['params'])
    scheduler_config = train_config['scheduler']
    scheduler = func_util.get_scheduler(optimizer, scheduler_config['type'], scheduler_config['params'])
    interval = train_config['interval']
    if interval <= 0:
        num_batches = len(train_loader)
        interval = num_batches // 20 if num_batches >= 20 else 1

    end_epoch = start_epoch + train_config['epoch'] if num_epochs is None else start_epoch + num_epochs
    start_time = time.time()
    for epoch in range(start_epoch, end_epoch):
        if distributed:
            train_loader.sampler.set_epoch(epoch)

        train_epoch(model, train_loader, optimizer, criterion, epoch, device, interval)
        valid_acc = validate(model, valid_loader, device)
        if valid_acc > best_valid_acc and main_util.is_main_process():
            print('Updating ckpt (Best top1 accuracy: {:.4f} -> {:.4f})'.format(best_valid_acc, valid_acc))
            best_valid_acc = valid_acc
            save_ckpt(model_without_ddp, best_valid_acc, epoch, ckpt_file_path, model_type)
        scheduler.step()

    dist.barrier()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def run(args):
    distributed, device_ids = main_util.init_distributed_mode(args.world_size, args.dist_url)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        cudnn.benchmark = True

    print(args)
    config = yaml_util.load_yaml_file(args.config)
    train_loader, valid_loader, test_loader = main_util.get_data_loaders(config, distributed)
    if 'mimic_model' in config:
        model = mimic_util.get_mimic_model_easily(config, device)
        model_config = config['mimic_model']
    else:
        model = module_util.get_model(config, device)
        model_config = config['model']

    model_type, best_valid_acc, start_epoch, ckpt_file_path =\
        module_util.resume_from_ckpt(model, model_config, args.init)
    train_config = config['train']
    criterion_config = train_config['criterion']
    criterion = func_util.get_loss(criterion_config['type'], criterion_config['params'])
    if not args.evaluate:
        train(model, train_loader, valid_loader, best_valid_acc, criterion, device, distributed, device_ids,
              train_config, args.epoch, start_epoch, args.lr, ckpt_file_path, model_type)
    test(model, test_loader, device)


if __name__ == '__main__':
    parser = get_argparser()
    run(parser.parse_args())
