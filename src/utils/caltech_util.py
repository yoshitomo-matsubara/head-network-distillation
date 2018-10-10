import os

import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms

from autoencoders import *
from structure.dataset import RgbImageDataset
from utils import data_util, file_util


def get_test_transformer(normalizer, compression_type, compressed_size_str, org_size=(180, 180), ae=None):
    normal_list = [transforms.ToTensor()]
    if ae is not None:
        normal_list.append(AETransformer(ae))

    if normalizer is not None:
        normal_list.append(normalizer)

    normal_transformer = transforms.Compose(normal_list)
    if compression_type is None or compressed_size_str is None:
        return normal_transformer

    hw = compressed_size_str.split(',')
    compressed_size = (int(hw[0]), int(hw[1]))
    if compression_type == 'base':
        comp_list = [transforms.Resize(compressed_size), transforms.Resize(org_size), transforms.ToTensor()]
        if normalizer is not None:
            comp_list.append(normalizer)
        return transforms.Compose(comp_list)
    return normal_transformer


def get_data_loaders(root_data_dir_path, batch_size=100, compression_type=None, compressed_size_str=None,
                     valid_rate=0.1, test_rate=0.1, random_seed=1, normalized=True, is_caltech256=False, ae=None,
                     reshape_size=(180, 180), compression_quality=0):
    dataset_name = '101' if not is_caltech256 else '256'
    data_dir_path = os.path.join(root_data_dir_path, dataset_name + '_ObjectCategories')
    if not os.path.exists(data_dir_path):
        ValueError('Could not find {} dataset at {}'.format('Caltech' + dataset_name, data_dir_path))

    sub_dir_path_list = file_util.get_dir_list(data_dir_path, is_sorted=True)
    file_path_lists = []
    for sub_dir_path in sub_dir_path_list:
        file_path_lists.append(file_util.get_file_list(sub_dir_path, is_sorted=True))

    train_file_path_lists, valid_file_path_lists, test_file_path_lists = [], [], []
    np.random.seed(random_seed)
    for file_path_list in file_path_lists:
        np.random.shuffle(file_path_list)
        sample_size = len(file_path_list)
        train_end_idx = int(sample_size * (1 - valid_rate - test_rate))
        valid_end_idx = train_end_idx + int(sample_size * valid_rate)
        train_file_path_lists.append(file_path_list[:train_end_idx])
        valid_file_path_lists.append(file_path_list[train_end_idx:valid_end_idx])
        test_file_path_lists.append(file_path_list[valid_end_idx:])

    train_dataset = RgbImageDataset(train_file_path_lists, reshape_size)
    normalizer = data_util.build_normalizer(train_dataset.load_all_data()) if normalized else None
    valid_comp_list = [transforms.ToTensor()]
    train_comp_list = [transforms.RandomHorizontalFlip(), transforms.ToTensor()]
    if normalizer is not None:
        valid_comp_list.append(normalizer)
        train_comp_list.append(normalizer)

    pin_memory = torch.cuda.is_available()

    train_transformer = transforms.Compose(train_comp_list)
    valid_transformer = transforms.Compose(valid_comp_list)
    train_dataset = RgbImageDataset(train_file_path_lists, reshape_size, train_transformer)
    valid_dataset = RgbImageDataset(valid_file_path_lists, reshape_size, valid_transformer)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=2, pin_memory=pin_memory)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=2, pin_memory=pin_memory)

    test_transformer = get_test_transformer(normalizer, compression_type, compressed_size_str, ae=ae)
    test_dataset = RgbImageDataset(test_file_path_lists, reshape_size, test_transformer, compression_quality)
    if 1 <= test_dataset.jpeg_quality <= 95:
        test_dataset.compute_compression_rate()
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True,
                                              num_workers=2, pin_memory=pin_memory)
    return train_loader, valid_loader, test_loader
