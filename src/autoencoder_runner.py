import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

from models.autoencoder.input_ae import InputAutoencoder
from models.autoencoder.middle_ae import MiddleAutoencoder
from myutils.common import file_util, yaml_util
from myutils.pytorch import func_util
from utils import module_util
from utils.dataset import general_util


def get_argparser():
    argparser = argparse.ArgumentParser(description='Autoencoder Trainer')
    argparser.add_argument('--config', required=True, help='yaml file path')
    argparser.add_argument('--epoch', type=int, help='epoch (higher priority than config if set)')
    argparser.add_argument('--lr', type=float, help='learning rate (higher priority than config if set)')
    argparser.add_argument('--gpu', type=int, help='gpu number')
    argparser.add_argument('-init', action='store_true', help='overwrite checkpoint')
    argparser.add_argument('-evaluate', action='store_true', help='evaluation option')
    return argparser


def extract_head_model(model, input_shape, device, partition_idx):
    if partition_idx is None or partition_idx == 0:
        return nn.Sequential()

    modules = list()
    module = model.module if isinstance(model, nn.DataParallel) else model
    module_util.extract_decomposable_modules(module, torch.rand(1, *input_shape).to(device), modules)
    return nn.Sequential(*modules[:partition_idx]).to(device)


def get_head_model(config, input_shape, device):
    org_model_config = config['org_model']
    model_config = yaml_util.load_yaml_file(org_model_config['config'])
    sub_model_config = model_config['model']
    if sub_model_config['type'] == 'inception_v3':
        sub_model_config['params']['aux_logits'] = False

    model = module_util.get_model(model_config, device)
    module_util.resume_from_ckpt(model, sub_model_config, False)
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
    elif ae_type == 'middle':
        autoencoder = MiddleAutoencoder(**ae_config['params'])

    if autoencoder is None:
        raise ValueError('ae_type `{}` is not expected'.format(ae_type))

    if device is None:
        return autoencoder, ae_type

    autoencoder = autoencoder.to(device)
    return autoencoder, ae_type


def train(autoencoder, head_model, train_loader, optimizer, criterion, epoch, device, interval):
    print('\nEpoch: %d' % epoch)
    autoencoder.train()
    head_model.eval()
    train_loss = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        head_outputs = head_model(inputs)
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


def extend_model(autoencoder, model, input_shape, device, partition_idx):
    if partition_idx is None or partition_idx == 0:
        return nn.Sequential(autoencoder, model)

    modules = list()
    module = model.module if isinstance(model, nn.DataParallel) else model
    module_util.extract_decomposable_modules(module, torch.rand(1, *input_shape).to(device), modules)
    return nn.Sequential(*modules[:partition_idx], autoencoder, *modules[partition_idx:]).to(device)


def get_extended_model(autoencoder, config, input_shape, device):
    org_model_config = config['org_model']
    model_config = yaml_util.load_yaml_file(org_model_config['config'])
    sub_model_config = model_config['model']
    if sub_model_config['type'] == 'inception_v3':
        sub_model_config['params']['aux_logits'] = False

    model = module_util.get_model(model_config, device)
    module_util.resume_from_ckpt(model, sub_model_config, False)
    return extend_model(autoencoder, model, input_shape, device, org_model_config['partition_idx']), model


def predict(inputs, targets, model):
    preds = model(inputs)
    loss = nn.functional.cross_entropy(preds, targets)
    _, pred_labels = preds.max(1)
    correct_count = pred_labels.eq(targets).sum().item()
    return correct_count, loss.item()


def test(extended_model, org_model, test_loader, device):
    print('Testing..')
    extended_model.eval()
    org_model.eval()
    ext_correct_count = 0
    ext_test_loss = 0
    org_correct_count = 0
    org_test_loss = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)
            sub_correct_count, sub_test_loss = predict(inputs, targets, extended_model)
            ext_correct_count += sub_correct_count
            ext_test_loss += sub_test_loss
            sub_correct_count, sub_test_loss = predict(inputs, targets, org_model)
            org_correct_count += sub_correct_count
            org_test_loss += sub_test_loss

    ext_acc = 100.0 * ext_correct_count / total
    print('[Extended]\t\tAverage Loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        ext_correct_count / total, ext_test_loss, total, ext_acc))
    org_acc = 100.0 * org_correct_count / total
    print('[Original]\tAverage Loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        org_test_loss / total, org_correct_count, total, org_acc))
    return ext_acc, org_acc


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
    autoencoder, ae_type = get_autoencoder(config, device)
    ckpt_file_path = config['autoencoder']['ckpt']
    start_epoch, best_avg_loss = resume_from_ckpt(ckpt_file_path, autoencoder)
    train_config = config['train']
    train_loader, valid_loader, test_loader =\
        general_util.get_data_loaders(dataset_config, batch_size=train_config['batch_size'],
                                      reshape_size=input_shape[1:3], jpeg_quality=-1)
    criterion_config = train_config['criterion']
    criterion = func_util.get_loss(criterion_config['type'], criterion_config['params'])
    if not args.evaluate:
        head_model = get_head_model(config, input_shape, device)
        module_util.use_multiple_gpus_if_available(autoencoder, device)
        module_util.use_multiple_gpus_if_available(head_model, device)
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

        if isinstance(autoencoder, nn.DataParallel):
            autoencoder = autoencoder.module

        resume_from_ckpt(ckpt_file_path, autoencoder)
        head_model = None

    extended_model, model = get_extended_model(autoencoder, config, input_shape, device)
    module_util.use_multiple_gpus_if_available(extended_model, device)
    module_util.use_multiple_gpus_if_available(model, device)
    test(extended_model, model, test_loader, device)


if __name__ == '__main__':
    parser = get_argparser()
    run(parser.parse_args())
