import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

from myutils.common import file_util, yaml_util
from myutils.pytorch import func_util
from utils import module_util
from utils.dataset import general_util
from models.autoencoder.input_ae import InputAutoencoder


def get_argparser():
    argparser = argparse.ArgumentParser(description='Autoencoder Trainer')
    argparser.add_argument('--config', required=True, help='yaml file path')
    argparser.add_argument('--epoch', type=int, help='epoch (higher priority than config if set)')
    argparser.add_argument('--lr', type=float, help='learning rate (higher priority than config if set)')
    argparser.add_argument('--gpu', type=int, help='gpu number')
    argparser.add_argument('-init', action='store_true', help='overwrite checkpoint')
    return argparser


def extract_head_model(model, input_shape, device, partition_idx):
    if partition_idx is None or partition_idx == 0:
        return None

    modules = list()
    module = model.module if isinstance(model, nn.DataParallel) else model
    module_util.extract_decomposable_modules(module, torch.rand(1, *input_shape).to(device), modules)
    return nn.Sequential(*modules[:partition_idx]).to(device)


def get_head_model(config, input_shape, device):
    org_model_config = config['org_model']
    model_config = yaml_util.load_yaml_file(org_model_config['config'])['model']
    if model_config['type'] == 'inception_v3':
        model_config['params']['aux_logits'] = False

    model = module_util.get_model(model_config, device)
    module_util.resume_from_ckpt(model_config['ckpt'], model, False)
    return extract_head_model(model, input_shape, device, org_model_config['partition_idx'])


def resume_from_ckpt(ckpt_file_path, autoencoder):
    if not file_util.check_if_exists(ckpt_file_path):
        print('Autoencoder checkpoint was not found at {}'.format(ckpt_file_path))
        return 1, 1e60

    print('Resuming from checkpoint..')
    checkpoint = torch.load(ckpt_file_path)
    state_dict = checkpoint['model']
    autoencoder.load_state_dict(state_dict)
    start_epoch = checkpoint['epoch']
    return start_epoch, checkpoint['best_avg_loss']


def get_autoencoder(config, device=None):
    autoencoder = None
    ae_config = config['autoencoder']
    ae_type = ae_config['type']
    if ae_type == 'input':
        autoencoder = InputAutoencoder(**ae_config['params'])

    if autoencoder is None:
        raise ValueError('ae_type `{}` is not expected'.format(ae_type))

    resume_from_ckpt(ae_config['ckpt'], autoencoder)
    if device is None:
        return autoencoder, ae_type

    autoencoder = autoencoder.to(device)
    return module_util.use_multiple_gpus_if_available(autoencoder, device), ae_type


def train(autoencoder, head_model, train_loader, optimizer, criterion, epoch, device, interval):
    print('\nEpoch: %d' % epoch)
    autoencoder.train()
    head_model.eval()
    train_loss = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        head_outputs = head_model(inputs) if head_model is not None else inputs
        ae_outputs = autoencoder(head_outputs)
        loss = criterion(ae_outputs, head_outputs)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        total += targets.size(0)
        if batch_idx > 0 and batch_idx % interval == 0:
            print('[{}/{} ({:.0f}%)]\tAvg Loss: {:.6f}'.format(batch_idx * len(inputs), len(train_loader.sampler),
                                                               100.0 * batch_idx / len(train_loader),
                                                               loss.item() / targets.size(0)))


def validate(autoencoder, head_model, valid_loader, criterion, device):
    autoencoder.eval()
    head_model.eval()
    valid_loss = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valid_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            head_outputs = head_model(inputs) if head_model is not None else inputs
            ae_outputs = autoencoder(head_outputs)
            loss = criterion(ae_outputs, head_outputs)
            valid_loss += loss.item()
            total += targets.size(0)

    avg_valid_loss = valid_loss / total
    print('Validation Loss: {:.6f}\tAvg Loss: {:.6f}'.format(valid_loss, avg_valid_loss))
    return avg_valid_loss


def save_ckpt(autoencoder, epoch, best_avg_loss, ckpt_file_path, ae_type):
    print('Saving..')
    module = autoencoder.module if isinstance(autoencoder, nn.DataParallel) else autoencoder
    state = {
        'type': ae_type,
        'model': module.state_dict(),
        'epoch': epoch + 1,
        'best_avg_loss': best_avg_loss
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
    head_model = get_head_model(config, input_shape, device)
    autoencoder, ae_type = get_autoencoder(config, device)
    ckpt_file_path = config['autoencoder']['ckpt']
    start_epoch, best_avg_loss = resume_from_ckpt(ckpt_file_path, autoencoder)
    if device.startswith('cuda'):
        head_model = nn.DataParallel(head_model)
        autoencoder = nn.DataParallel(autoencoder)

    train_config = config['train']
    train_loader, valid_loader, _ =\
        general_util.get_data_loaders(dataset_config, batch_size=train_config['batch_size'],
                                      reshape_size=input_shape[1:3], jpeg_quality=-1)
    criterion_config = train_config['criterion']
    criterion = func_util.get_loss(criterion_config['type'], criterion_config['params'])
    optim_config = train_config['optimizer']
    if args.lr is not None:
        optim_config['params']['lr'] = args.lr

    optimizer = func_util.get_optimizer(autoencoder, optim_config['type'], optim_config['params'])
    interval = train_config['interval']
    if interval <= 0:
        num_batches = len(train_loader)
        interval = num_batches // 100 if num_batches >= 100 else 1

    end_epoch = start_epoch + train_config['epoch'] if args.epoch is None else start_epoch + args.epoch
    for epoch in range(start_epoch, end_epoch):
        train(autoencoder, head_model, train_loader, optimizer, criterion, epoch, device, interval)
        avg_valid_loss = validate(autoencoder, head_model, valid_loader, criterion, device)
        if avg_valid_loss < best_avg_loss:
            best_avg_loss = avg_valid_loss
            save_ckpt(autoencoder, epoch, best_avg_loss, ckpt_file_path, ae_type)


if __name__ == '__main__':
    parser = get_argparser()
    run(parser.parse_args())
