import argparse
import datetime
import time

import torch
from torch.backends import cudnn
from torch.nn import DataParallel
from torch.nn.parallel.distributed import DistributedDataParallel

from myutils.common import file_util, yaml_util
from myutils.pytorch import func_util, module_util
from structure.logger import MetricLogger, SmoothedValue
from utils import ae_util, main_util


def get_argparser():
    argparser = argparse.ArgumentParser(description='PyTorch autoencoder trainer')
    argparser.add_argument('--config', required=True, help='yaml file path')
    argparser.add_argument('--device', default='cuda', help='device')
    argparser.add_argument('-test_only', action='store_true', help='only test model')
    # distributed training parameters
    argparser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    argparser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return argparser


def resume_from_ckpt(ckpt_file_path, autoencoder):
    if not file_util.check_if_exists(ckpt_file_path):
        print('Autoencoder checkpoint was not found at {}'.format(ckpt_file_path))
        return 1, None

    print('Resuming from checkpoint..')
    checkpoint = torch.load(ckpt_file_path)
    state_dict = checkpoint['model']
    autoencoder.load_state_dict(state_dict)
    start_epoch = checkpoint['epoch']
    return start_epoch, checkpoint['best_value']


def train_epoch(autoencoder, head_model, train_loader, optimizer, criterion, epoch, device, interval):
    print('\nEpoch: {}, LR: {:.3E}'.format(epoch, optimizer.param_groups[0]['lr']))
    autoencoder.train()
    head_model.eval()
    metric_logger = MetricLogger(delimiter='  ')
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('img/s', SmoothedValue(window_size=10, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)
    for sample_batch, targets in metric_logger.log_every(train_loader, interval, header):
        start_time = time.time()
        sample_batch = sample_batch.to(device)
        optimizer.zero_grad()
        head_outputs = head_model(sample_batch)
        ae_outputs = autoencoder(head_outputs)
        loss = criterion(ae_outputs, head_outputs) if not isinstance(ae_outputs, tuple) else ae_outputs[1]
        loss.backward()
        optimizer.step()
        batch_size = sample_batch.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])
        metric_logger.meters['img/s'].update(batch_size / (time.time() - start_time))


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
    torch.set_num_threads(num_threads)
    return metric_logger.acc1.global_avg


def validate(ae_without_ddp, data_loader, config, device, distributed, device_ids):
    input_shape = config['input_shape']
    extended_model, model = ae_util.get_extended_model(ae_without_ddp, config, input_shape, device)
    if distributed:
        extended_model = DistributedDataParallel(extended_model, device_ids=device_ids)
    return evaluate(extended_model, data_loader, device, split_name='Validation')


def save_ckpt(autoencoder, epoch, best_avg_loss, ckpt_file_path, ae_type):
    print('Saving..')
    module = autoencoder.module if isinstance(autoencoder, (DistributedDataParallel, DataParallel)) else autoencoder
    state = {
        'type': ae_type,
        'model': module.state_dict(),
        'epoch': epoch + 1,
        'best_value': best_avg_loss
    }
    file_util.make_parent_dirs(ckpt_file_path)
    torch.save(state, ckpt_file_path)


def train(train_loader, valid_loader, input_shape, config, device, distributed, device_ids):
    ae_without_ddp, ae_type = ae_util.get_autoencoder(config, device)
    head_model = ae_util.get_head_model(config, input_shape, device)
    module_util.freeze_module_params(head_model)
    ckpt_file_path = config['autoencoder']['ckpt']
    start_epoch, best_valid_acc = resume_from_ckpt(ckpt_file_path, ae_without_ddp)
    if best_valid_acc is None:
        best_valid_acc = 0.0

    train_config = config['train']
    criterion_config = train_config['criterion']
    criterion = func_util.get_loss(criterion_config['type'], criterion_config['params'])
    optim_config = train_config['optimizer']
    optimizer = func_util.get_optimizer(ae_without_ddp, optim_config['type'], optim_config['params'])
    scheduler_config = train_config['scheduler']
    scheduler = func_util.get_scheduler(optimizer, scheduler_config['type'], scheduler_config['params'])
    interval = train_config['interval']
    if interval <= 0:
        num_batches = len(train_loader)
        interval = num_batches // 20 if num_batches >= 20 else 1

    autoencoder = ae_without_ddp
    if distributed:
        autoencoder = DistributedDataParallel(ae_without_ddp, device_ids=device_ids)
        head_model = DataParallel(head_model, device_ids=device_ids)
    elif device.type == 'cuda':
        autoencoder = DataParallel(ae_without_ddp)
        head_model = DataParallel(head_model)

    end_epoch = start_epoch + train_config['epoch']
    start_time = time.time()
    for epoch in range(start_epoch, end_epoch):
        if distributed:
            train_loader.sampler.set_epoch(epoch)

        train_epoch(autoencoder, head_model, train_loader, optimizer, criterion, epoch, device, interval)
        valid_acc = validate(ae_without_ddp, valid_loader, config, device, distributed, device_ids)
        if valid_acc < best_valid_acc and main_util.is_main_process():
            best_valid_acc = valid_acc
            save_ckpt(ae_without_ddp, epoch, best_valid_acc, ckpt_file_path, ae_type)
        scheduler.step()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    del head_model


def run(args):
    distributed, device_ids = main_util.init_distributed_mode(args.world_size, args.dist_url)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        cudnn.benchmark = True

    print(args)
    config = yaml_util.load_yaml_file(args.config)
    input_shape = config['input_shape']
    ckpt_file_path = config['autoencoder']['ckpt']
    train_loader, valid_loader, test_loader = main_util.get_data_loaders(config, distributed)
    if not args.evaluate:
        train(train_loader, valid_loader, input_shape, config, device, distributed, device_ids)

    autoencoder, _ = ae_util.get_autoencoder(config, device)
    resume_from_ckpt(ckpt_file_path, autoencoder)
    extended_model, model = ae_util.get_extended_model(autoencoder, config, input_shape, device)
    if device.type == 'cuda':
        extended_model = DistributedDataParallel(extended_model, device_ids=device_ids) if distributed \
            else DataParallel(extended_model)
    evaluate(extended_model, test_loader, device, title='[Mimic model]')


if __name__ == '__main__':
    parser = get_argparser()
    run(parser.parse_args())
