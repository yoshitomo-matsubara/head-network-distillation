import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

from myutils.common import file_util, yaml_util
from utils import mimic_util
from utils.dataset import general_util


def get_argparser():
    argparser = argparse.ArgumentParser(description='Mimic Tester')
    argparser.add_argument('--config', required=True, help='yaml file path')
    argparser.add_argument('--gpu', type=int, help='gpu number')
    return argparser


def predict(inputs, targets, model):
    preds = model(inputs)
    loss = nn.functional.cross_entropy(preds, targets)
    _, pred_labels = preds.max(1)
    correct_count = pred_labels.eq(targets).sum().item()
    return correct_count, loss.item()


def test(mimic_model, org_model, test_loader, device):
    print('Testing..')
    mimic_model.eval()
    org_model.eval()
    mimic_correct_count = 0
    mimic_test_loss = 0
    org_correct_count = 0
    org_test_loss = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)
            sub_correct_count, sub_test_loss = predict(inputs, targets, mimic_model)
            mimic_correct_count += sub_correct_count
            mimic_test_loss += sub_test_loss
            sub_correct_count, sub_test_loss = predict(inputs, targets, org_model)
            org_correct_count += sub_correct_count
            org_test_loss += sub_test_loss

    mimic_acc = 100.0 * mimic_correct_count / total
    print('[Mimic]\t\tAverage Loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        mimic_test_loss / total, mimic_correct_count, total, mimic_acc))
    org_acc = 100.0 * org_correct_count / total
    print('[Original]\tAverage Loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        org_test_loss / total, org_correct_count, total, org_acc))
    return mimic_acc, org_acc


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
        gpu_number = args.gpu
        if gpu_number is not None and gpu_number >= 0:
            device += ':' + str(gpu_number)

    config = yaml_util.load_yaml_file(args.config)
    teacher_model_config = config['teacher_model']
    org_model, teacher_model_type = mimic_util.get_org_model(teacher_model_config, device)
    mimic_model = mimic_util.get_mimic_model(config, org_model, teacher_model_type, teacher_model_config, device)
    dataset_config = config['dataset']
    _, _, test_loader =\
        general_util.get_data_loaders(dataset_config, batch_size=config['test']['batch_size'],
                                      rough_size=config['train']['rough_size'],
                                      reshape_size=tuple(config['input_shape'][1:3]), jpeg_quality=-1)
    test(mimic_model, org_model, test_loader, device)
    file_util.save_pickle(mimic_model, config['mimic_model']['ckpt'])


if __name__ == '__main__':
    parser = get_argparser()
    run(parser.parse_args())
