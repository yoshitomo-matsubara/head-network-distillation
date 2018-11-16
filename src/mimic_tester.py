import argparse

import torch
import torch.backends.cudnn as cudnn

import mimic_learner
from models.mimic.densenet_mimic import *
from models.mimic.inception_mimic import *
from models.mimic.resnet_mimic import *
from myutils.common import file_util, yaml_util
from utils import module_util
from utils.dataset import general_util


def get_argparser():
    argparser = argparse.ArgumentParser(description='Mimic Tester')
    argparser.add_argument('--config', required=True, help='yaml file path')
    argparser.add_argument('-init', action='store_true', help='overwrite checkpoint')
    return argparser


def load_student_model(student_config, teacher_model_type, device):
    student_model_config = student_config['student_model']
    student_model = mimic_learner.get_student_model(teacher_model_type, student_model_config)
    student_model = student_model.to(device)
    mimic_learner.resume_from_ckpt(student_model_config['ckpt'], student_model, True)
    return student_model


def get_org_model(teacher_model_config, device):
    teacher_config = yaml_util.load_yaml_file(teacher_model_config['config'])
    if teacher_config['model']['type'] == 'inception_v3':
        teacher_config['model']['params']['aux_logits'] = False

    model = module_util.get_model(teacher_config, device)
    model_config = teacher_config['model']
    mimic_learner.resume_from_ckpt(model_config['ckpt'], model)
    return model, model_config['type']


def get_mimic_model(config, org_model, teacher_model_type, teacher_model_config, device):
    student_model = load_student_model(config, teacher_model_type, device)
    org_modules = list()
    input_batch = torch.rand(config['input_shape']).unsqueeze(0).to(device)
    module_util.extract_decomposable_modules(org_model, input_batch, org_modules)
    end_idx = teacher_model_config['end_idx']
    mimic_modules = [student_model]
    mimic_modules.extend(org_modules[end_idx:])
    mimic_model_config = config['mimic_model']
    mimic_type = mimic_model_config['type']
    if mimic_type.startswith('densenet'):
        return DenseNetMimic(mimic_modules)
    elif mimic_type.startswith('inception'):
        return InceptionMimic(mimic_modules)
    elif mimic_type.startswith('resnet'):
        return ResNetMimic(mimic_modules)
    raise ValueError('mimic_type `{}` is not expected'.format(mimic_type))


def predict(inputs, targets, model):
    preds = model(inputs)
    loss = nn.functional.cross_entropy(preds, targets)
    _, pred_labels = preds.max(1)
    correct_count = pred_labels.eq(targets).sum().item()
    return correct_count, loss.item()


def test(mimic_model, org_model, test_loader, device):
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

    config = yaml_util.load_yaml_file(args.config)
    teacher_model_config = config['teacher_model']
    org_model, teacher_model_type = get_org_model(teacher_model_config, device)
    mimic_model = get_mimic_model(config, org_model, teacher_model_type, teacher_model_config, device)
    test_config = config['test']
    dataset_config = config['dataset']
    _, _, test_loader =\
        general_util.get_data_loaders(dataset_config['data'], batch_size=test_config['batch_size'], ae_model=None,
                                      reshape_size=tuple(config['input_shape'][1:3]), compression_quality=-1)
    test(mimic_model, org_model, test_loader, device)
    file_util.save_pickle(mimic_model, config['mimic_model']['ckpt'])


if __name__ == '__main__':
    parser = get_argparser()
    run(parser.parse_args())
