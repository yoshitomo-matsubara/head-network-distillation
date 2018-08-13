import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

from autoencoders import *
from utils import data_util


def get_train_and_valid_loaders(data_dir_path, batch_size, normalized, valid_rate, is_cifar100,
                                random_seed=1, shuffle=True):
    train_dataset = torchvision.datasets.CIFAR10(root=data_dir_path, train=True, download=True) if not is_cifar100\
        else torchvision.datasets.CIFAR100(root=data_dir_path, train=True, download=True)
    org_train_size = len(train_dataset)
    indices = list(range(org_train_size))
    train_end_idx = int(np.floor((1 - valid_rate) * org_train_size))
    normalizer = data_util.build_normalizer(train_dataset.train_data[:train_end_idx]) if normalized else None
    valid_comp_list = [transforms.ToTensor()]
    train_comp_list = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor()]
    if normalizer is not None:
        valid_comp_list.append(normalizer)
        train_comp_list.append(normalizer)

    train_transformer = transforms.Compose(train_comp_list)
    valid_transformer = transforms.Compose(valid_comp_list)
    train_dataset = torchvision.datasets.CIFAR10(root=data_dir_path, train=True,
                                                 download=True, transform=train_transformer) if not is_cifar100\
        else torchvision.datasets.CIFAR100(root=data_dir_path, train=True, download=True, transform=train_transformer)
    valid_dataset = torchvision.datasets.CIFAR10(root=data_dir_path, train=True,
                                                 download=True, transform=valid_transformer) if not is_cifar100\
        else torchvision.datasets.CIFAR100(root=data_dir_path, train=True, download=True, transform=valid_transformer)

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_indices, valid_indices = indices[:train_end_idx], indices[train_end_idx:]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(valid_indices)
    pin_memory = torch.cuda.is_available()
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                                               num_workers=2, pin_memory=pin_memory)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, sampler=valid_sampler,
                                               num_workers=2, pin_memory=pin_memory)
    return train_loader, valid_loader, normalizer


def get_test_transformer(normalizer, batch_size, compression_type, compressed_size_str, org_size=(32, 32), ae=None):
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


def get_data_loaders(data_dir_path, batch_size=128, compression_type=None, compressed_size_str=None,
                     valid_rate=0.1, normalized=True, is_cifar100=False, ae=None):
    train_loader, valid_loader, normalizer =\
        get_train_and_valid_loaders(data_dir_path, batch_size=batch_size, normalized=normalized,
                                    valid_rate=valid_rate, is_cifar100=is_cifar100)
    test_transformer = get_test_transformer(normalizer, compression_type, compressed_size_str, ae=ae)
    test_dataset =\
        torchvision.datasets.CIFAR10(root=data_dir_path, train=False, download=True, transform=test_transformer)\
            if not is_cifar100 else torchvision.datasets.CIFAR100(root=data_dir_path, train=False,
                                                                  download=True, transform=test_transformer)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2,
                                              pin_memory=torch.cuda.is_available())
    return train_loader, valid_loader, test_loader
