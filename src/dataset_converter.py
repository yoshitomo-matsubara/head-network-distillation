import argparse
import os
import random

from PIL import Image

from myutils.common import file_util


def get_argparser():
    parser = argparse.ArgumentParser(description='PyTorch dataset converter')
    parser.add_argument('--input', required=True, help='input dir path')
    parser.add_argument('--dataset', default='caltech', help='dataset type')
    parser.add_argument('--val', type=float, default=0.1, help='validation data rate')
    parser.add_argument('--test', type=float, default=0.1, help='test data rate')
    parser.add_argument('--output', required=True, help='output dir path')
    parser.add_argument('-rgb', action='store_true', help='option to ignore non-RGB image files')
    return parser


def write_converted_dataset(data_list, rgb_only, output_file_path, delimiter='\t'):
    file_util.make_parent_dirs(output_file_path)
    with open(output_file_path, 'w') as fp:
        for label_name, image_file_paths in data_list:
            for image_file_path in image_file_paths:
                if rgb_only:
                    img = Image.open(image_file_path)
                    if img.mode != 'RGB':
                        continue
                fp.write('{}{}{}\n'.format(image_file_path, delimiter, label_name))


def convert_caltech_dataset(input_dir_path, val_rate, test_rate, rgb_only, output_dir_path):
    sub_dir_path_list = file_util.get_dir_path_list(input_dir_path, is_sorted=True)
    dataset_dict = {'train': [], 'valid': [], 'test': []}
    for sub_dir_path in sub_dir_path_list:
        label_name = os.path.basename(sub_dir_path)
        image_file_paths = file_util.get_file_path_list(sub_dir_path, is_sorted=True)
        random.shuffle(image_file_paths)
        total_size = len(image_file_paths)
        train_end_idx = int(total_size * (1 - val_rate - test_rate))
        valid_end_idx = train_end_idx + int((total_size - train_end_idx) * val_rate / (val_rate + test_rate))
        dataset_dict['train'].append((label_name, image_file_paths[:train_end_idx]))
        dataset_dict['valid'].append((label_name, image_file_paths[train_end_idx:valid_end_idx]))
        dataset_dict['test'].append((label_name, image_file_paths[valid_end_idx:]))

    write_converted_dataset(dataset_dict['train'], rgb_only, os.path.join(output_dir_path, 'train.txt'))
    write_converted_dataset(dataset_dict['valid'], rgb_only, os.path.join(output_dir_path, 'valid.txt'))
    write_converted_dataset(dataset_dict['test'], rgb_only, os.path.join(output_dir_path, 'test.txt'))


def convert_imagenet_dataset(input_dir_path, output_dir_path):
    dataset_dict = dict()
    for key in ['train', 'val']:
        pair_list = list()
        sub_dir_path_list = file_util.get_dir_path_list(os.path.join(input_dir_path, key), is_sorted=True)
        for sub_dir_path in sub_dir_path_list:
            label_name = os.path.basename(sub_dir_path)
            image_file_paths = file_util.get_file_path_list(sub_dir_path, is_sorted=True)
            random.shuffle(image_file_paths)
            pair_list.append((label_name, image_file_paths))
        dataset_dict[key] = pair_list

    write_converted_dataset(dataset_dict['train'], False, os.path.join(output_dir_path, 'train.txt'))
    write_converted_dataset(dataset_dict['val'], False, os.path.join(output_dir_path, 'valid.txt'))


def run(args):
    input_dir_path = args.input
    dataset_type = args.dataset
    valid_rate = args.val
    test_rate = args.test
    output_dir_path = args.output
    rgb_only = args.rgb
    if dataset_type == 'caltech':
        convert_caltech_dataset(input_dir_path, valid_rate, test_rate, rgb_only, output_dir_path)
    elif dataset_type == 'imagenet':
        convert_imagenet_dataset(input_dir_path, output_dir_path)
    else:
        raise ValueError('dataset_type `{}` is not expected'.format(dataset_type))


if __name__ == '__main__':
    parser = get_argparser()
    run(parser.parse_args())
