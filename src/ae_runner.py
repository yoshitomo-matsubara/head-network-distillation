import argparse

import torch.utils.data
import yaml

import image_classifier
from myutils.common import file_util
from myutils.pytorch import func_util
from utils import module_util


def get_argparser():
    parser = argparse.ArgumentParser(description='Autoencoder Runner')
    parser.add_argument('--config', required=True, help='yaml file path')
    parser.add_argument('-init', action='store_true', help='overwrite checkpoint')
    parser.add_argument('-evaluate', action='store_true', help='evaluation option')
    return parser


def resume_from_ckpt(ae_model, ae_config, init):
    ckpt_file_path = ae_config['ckpt']
    if init or not file_util.check_if_exists(ckpt_file_path):
        return ae_config['type'], 1e60, 1, ckpt_file_path

    print('Resuming from checkpoint..')
    checkpoint = torch.load(ckpt_file_path)
    ae_model.load_state_dict(checkpoint['autoencoder'])
    ae_type = checkpoint['type']
    best_loss = checkpoint['loss']
    start_epoch = checkpoint['epoch']
    return ae_type, best_loss, start_epoch, ckpt_file_path


def train(ae_model, train_loader, optimizer, epoch, device, interval):
    ae_model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        loss = ae_model.loss_function(data)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.sampler),
                100.0 * batch_idx / len(train_loader), loss / len(data)))
    print('Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.sampler)))


def save_ckpt(ae_model, loss, epoch, ckpt_file_path, ae_type):
    print('Saving..')
    state = {
        'type': ae_type,
        'autoencoder': ae_model.state_dict(),
        'loss': loss,
        'epoch': epoch,
    }
    file_util.make_parent_dirs(ckpt_file_path)
    torch.save(state, ckpt_file_path)


def test(ae_model, test_loader, device, data_type='Test'):
    ae_model.eval()
    test_loss = 0
    for i, (data, _) in enumerate(test_loader):
        data = data.to(device)
        test_loss += ae_model.loss_function(data).item()

    test_loss /= len(test_loader.sampler)
    print('{} set loss: {:.4f}'.format(data_type, test_loss))
    return test_loss


def validate(ae_model, valid_loader, device, epoch, best_loss, ckpt_file_path, ae_type):
    loss = test(ae_model, valid_loader, device, data_type='Validation')
    if loss < best_loss:
        save_ckpt(ae_model, loss, epoch, ckpt_file_path, ae_type)
        best_loss = loss
    return best_loss


def run(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with open(args.config, 'r') as fp:
        config = yaml.load(fp)

    train_loader, valid_loader, test_loader = image_classifier.get_data_loaders(config)
    ae_model = module_util.get_autoencoder(device, config)
    ae_type, best_loss, start_epoch, ckpt_file_path = resume_from_ckpt(ae_model, config['autoencoder'], args.init)
    if not args.evaluate:
        train_config = config['train']
        optim_config = train_config['optimizer']
        optimizer = func_util.get_optimizer(ae_model, optim_config['type'], optim_config['params'])
        interval = train_config['interval']
        for epoch in range(start_epoch, start_epoch + train_config['epoch']):
            train(ae_model, train_loader, optimizer, epoch, device, interval)
            best_loss = validate(ae_model, valid_loader, torch, epoch, best_loss, ckpt_file_path, ae_type)
    test(ae_model, test_loader, device)


if __name__ == '__main__':
    parser = get_argparser()
    run(parser.parse_args())
